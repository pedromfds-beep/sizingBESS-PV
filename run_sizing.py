"""
Script Principal de Dimensionamento Ótimo de Sistema PV + BESS
==============================================================

Orquestra a pipeline analítica modular completa:
1. Obtenção do perfil de consumo (normativo E-REDES BTN ou CSV medido).
2. Simulação física da produção fotovoltaica (PVLIB com irradiação solar e temperatura).
3. Construção da curva horária do tarifário tri-horário português (ERSE).
4. Formulação e resolução do problema de otimização convexa (CVXPY + HiGHS).
5. Pós-processamento energético, ciclos Rainflow (ASTM E1049-85) e viabilidade económica.
6. Emissão de relatório executivo e exportação de gráficos e séries horárias.

Exemplos de Utilização (CLI):
-----------------------------
# 1. Simulação com parâmetros padrão de 2025 (E-REDES BTN_C 3500 kWh/ano, PV fixo 0.405 kWp)
python3 run_sizing.py

# 2. Otimização conjunta de PV e Bateria com venda de excedentes à rede (UPAC)
python3 run_sizing.py --optimize-pv --allow-export --export-price 0.05

# 3. Otimização com limite de telhado (ex: 3.0 kWp) e consumo anual personalizado
python3 run_sizing.py --optimize-pv --max-pv-kwp 3.0 --eredes-consumo 4000

# 4. Simulação com dados medidos de habitação em CSV
python3 run_sizing.py --load-source csv --consumption-csv "Load House 13.csv"
"""
import os
import argparse
import pandas as pd
import numpy as np

from config import SiteConfig, PVConfig, BESSConfig, EconomicsConfig, TariffConfig
from pv_engine import get_pv_timeseries_1kwp
from tariffs import create_trihourly_tariff_dataframe
from optimization import solve_sizing_optimization
from financial import analyze_financials
from plotting import plot_energy_balance, plot_accumulated_cash_flow, plot_soc_distribution
from PerfilConsumo import processar_perfil_btn


def load_consumption_csv(file_path: str, year: int = 2025) -> pd.Series:
    """
    Carrega o perfil de consumo horário a partir de um ficheiro CSV medido.
    Se a resolução for de 15 minutos, efetua o resample horário através da média.
    """
    print(f"-> A carregar perfil de consumo medido a partir de '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ficheiro de consumo '{file_path}' não encontrado.")

    df = pd.read_csv(file_path)
    col_date = 'DateTime' if 'DateTime' in df.columns else 'Timestamp'
    col_load = 'Consumption (kW)'

    df[col_date] = pd.to_datetime(df[col_date])
    df = df.set_index(col_date).sort_index()

    df_year = df[df.index.year == year]
    if df_year.empty:
        print(f"   Aviso: sem dados para o ano {year}. A utilizar todos os registos disponíveis.")
        df_year = df

    load_hourly = df_year[col_load].resample('h').mean().fillna(0.0)
    load_hourly.name = 'Load_kW'
    total_kwh = load_hourly.sum()
    print(f"   Consumo anual medido carregado: {total_kwh:.1f} kWh/ano | Pico: {load_hourly.max():.3f} kW")
    return load_hourly


def load_consumption_eredes(
    consumo_anual_kwh: float,
    perfil: str = 'BTN_C',
    sim_year: int = 2025,
    eredes_year: int = 2025
) -> pd.Series:
    """
    Gera o perfil de consumo horário com base no perfil normativo oficial da E-REDES,
    alinhado perfeitamente para o ano civil da simulação.
    """
    # Obter perfil a partir da cache ou do Excel oficial
    load_eredes = processar_perfil_btn(
        consumo_anual_kwh=consumo_anual_kwh,
        perfil=perfil,
        ano=eredes_year
    )

    std_idx = pd.date_range(
        start=f"{sim_year}-01-01 00:00:00",
        end=f"{sim_year}-12-31 23:00:00",
        freq='h'
    )

    # Re-indexar se o ano de simulação diferir do ano do ficheiro E-REDES
    if sim_year != eredes_year:
        load_eredes.index = load_eredes.index.map(
            lambda dt: dt.replace(year=sim_year)
        )

    # Eliminar duplicados decorrentes da mudança de hora legal de outono (DST)
    load_eredes = load_eredes[~load_eredes.index.duplicated(keep='first')]
    load_sim = load_eredes.reindex(std_idx).ffill().bfill()
    load_sim.name = 'Load_kW'
    return load_sim


def print_executive_report(opt: dict, fin: dict, pv_cfg: PVConfig, bess_cfg: BESSConfig, econ_cfg: EconomicsConfig):
    """
    Apresenta no terminal um relatório executivo estruturado com todos os
    indicadores técnicos, energéticos e financeiros da solução ótima obtida.
    """
    print("\n" + "=" * 72)
    print("        RELATÓRIO EXECUTIVO DE DIMENSIONAMENTO ÓTIMO PV + BESS")
    print("=" * 72)

    print("\n1. CONFIGURAÇÃO DA INSTALAÇÃO")
    print(f"   - Localização Geográfica:  Évora, Portugal (38.529° N, 8.010° O)")
    print(f"   - Sistema Fotovoltaico:    {pv_cfg.peak_power_kwp:.3f} kWp ({pv_cfg.system_type}, inclinação={pv_cfg.fixed_tilt}°)")
    print(f"   - Tecnologia da Bateria:   {bess_cfg.name} (Eficiência round-trip={bess_cfg.efficiency*100:.1f}%)")
    export_desc = f"Permitida ({econ_cfg.export_price_eur_per_kwh:.3f} €/kWh)" if econ_cfg.allow_export else "Não permitida (Zero Export)"
    print(f"   - Injeção na Rede:         {export_desc}")
    print(f"   - Vida Útil do Projeto:    {econ_cfg.project_lifetime_years} anos | Taxa de desconto: {econ_cfg.discount_rate*100:.1f}%")

    print("\n2. DIMENSIONAMENTO ÓTIMO (CVXPY / HiGHS)")
    if pv_cfg.optimize_capacity:
        print(f"   - Potência Ótima PV:       {opt['pv_capacity_kwp']:.2f} kWp (Otimizado)")
    else:
        print(f"   - Potência PV Instalada:   {opt['pv_capacity_kwp']:.2f} kWp (Fixo)")
    print(f"   - Capacidade Ótima BESS:   {opt['batt_capacity_kwh']:.2f} kWh")
    print(f"   - Potência Ótima BESS:     {opt['batt_power_kw']:.2f} kW")
    print(f"   - Taxa C-rate:             {opt['c_rate']:.2f} C")
    print(f"   - Tempo de Resolução:      {opt['solve_time_sec']:.2f} segundos")

    print("\n3. BALANÇO ENERGÉTICO ANUAL (kWh/ano)")
    print(f"   - Consumo Total da Casa:   {fin['total_load_kwh']:.1f} kWh")
    print(f"   - Produção Solar (PVLIB):  {fin['total_pv_kwh']:.1f} kWh")
    print(f"   - Importação da Rede:      {fin['total_grid_import_kwh']:.1f} kWh")
    print(f"   - Exportação para a Rede:  {fin['total_grid_export_kwh']:.1f} kWh")
    if fin['allow_export'] and fin['total_grid_export_kwh'] > 0:
        print(f"     * Receita de Venda:      +{fin['annual_export_revenue']:.2f} EUR/ano (a {fin['export_price_eur_per_kwh']:.3f} €/kWh)")
    print(f"   - Corte Solar (Curtail):   {fin['total_curtailed_kwh']:.1f} kWh")
    print(f"   - Carga da Bateria (Total):{fin['total_charge_kwh']:.1f} kWh")
    print(f"     * A partir de Solar:     {fin['charge_from_pv_kwh']:.1f} kWh ({(fin['charge_from_pv_kwh']/max(1e-3, fin['total_charge_kwh']))*100:.1f}%)")
    print(f"     * A partir da Rede:      {fin['charge_from_grid_kwh']:.1f} kWh ({(fin['charge_from_grid_kwh']/max(1e-3, fin['total_charge_kwh']))*100:.1f}%)")
    print(f"   - Taxa de Autoconsumo:     {fin['self_consumption_rate_pct']:.1f} %")
    print(f"   - Taxa de Autossuficiência:{fin['self_sufficiency_rate_pct']:.1f} %")

    print("\n4. DEGRADAÇÃO E CICLOS DA BATERIA (RAINFLOW)")
    print(f"   - Ciclos Equivalentes (EFC): {fin['annual_rainflow_efc']:.1f} ciclos/ano")
    print(f"   - Ciclos ao Longo da Vida:   {fin['lifetime_efc']:.0f} ciclos ({econ_cfg.project_lifetime_years} anos)")

    print("\n5. VIABILIDADE FINANCEIRA E INVESTIMENTO")
    print(f"   - CAPEX Fotovoltaico:      {opt['pv_capex']:.2f} EUR")
    print(f"   - CAPEX Bateria:           {opt['batt_capex']:.2f} EUR")
    print(f"   - CAPEX Inicial Total:     {opt['total_initial_capex']:.2f} EUR")
    print(f"   - Custo Anual Baseline:    {fin['baseline_annual_cost']:.2f} EUR/ano (Apenas Rede)")
    print(f"   - Custo Anual Sistema:     {fin['system_annual_opex']:.2f} EUR/ano (Operacional)")
    print(f"   - Poupança Anual Líquida:  {fin['annual_savings']:.2f} EUR/ano")
    if fin['payback_exact_years'] is not None:
        print(f"   - Período de Retorno:      {fin['payback_exact_years']:.2f} anos (Payback)")
    else:
        print(f"   - Período de Retorno:      Superior a {econ_cfg.project_lifetime_years} anos")
    print(f"   - Custo Total Anualizado:  {opt['total_annualized_cost']:.2f} EUR/ano (com amortização CRF)")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dimensionamento Ótimo de Sistema PV + BESS Residencial (Portugal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Fonte de Consumo ---
    grp_load = parser.add_argument_group('Perfil de Consumo')
    grp_load.add_argument("--load-source", choices=["eredes", "csv"], default="eredes",
                          help="Fonte do perfil de consumo: 'eredes' (normativo oficial) ou 'csv' (dados medidos)")
    grp_load.add_argument("--consumption-csv", type=str, default="Load House 13.csv",
                          help="[csv] Caminho para o ficheiro CSV com medições reais de consumo")
    grp_load.add_argument("--eredes-perfil", type=str, default="BTN_C",
                          choices=["BTN_A", "BTN_B", "BTN_C"],
                          help="[eredes] Classe de perfil BTN (BTN_C = habitação familiar típica)")
    grp_load.add_argument("--eredes-consumo", type=float, default=3500.0,
                          help="[eredes] Consumo anual total da instalação (kWh/ano) para escalar o perfil")
    grp_load.add_argument("--eredes-year", type=int, default=2025, choices=[2024, 2025, 2026],
                          help="[eredes] Ano do ficheiro de dados oficiais da E-REDES")

    # --- Simulação Temporal ---
    grp_sim = parser.add_argument_group('Simulação Temporal')
    grp_sim.add_argument("--year", type=int, default=2025, help="Ano civil de simulação (clima, calendário e tarifas)")

    # --- Sistema Fotovoltaico ---
    grp_pv = parser.add_argument_group('Sistema Fotovoltaico')
    grp_pv.add_argument("--optimize-pv", action="store_true",
                        help="Otimizar livremente a potência instalada de módulos PV (kWp)")
    grp_pv.add_argument("--max-pv-kwp", type=float, default=10.0,
                        help="[optimize-pv] Limite máximo de potência fotovoltaica admissível (kWp)")
    grp_pv.add_argument("--min-pv-kwp", type=float, default=0.0,
                        help="[optimize-pv] Limite mínimo de potência fotovoltaica (kWp)")
    grp_pv.add_argument("--pv-peak", type=float, default=0.405,
                        help="Potência fixa de pico de módulos se não otimizada (kWp)")
    grp_pv.add_argument("--pv-type", type=str, default="fixed",
                        choices=["fixed", "single_axis"], help="Tipo de estrutura PV (fixed ou seguidor 1-eixo)")

    # --- Rede e Venda de Excedentes ---
    grp_grid = parser.add_argument_group('Rede e Venda de Excedentes')
    grp_grid.add_argument("--allow-export", action="store_true",
                          help="Permitir injeção e remuneração de excedentes solares na rede (UPAC)")
    grp_grid.add_argument("--export-price", type=float, default=0.05,
                          help="Preço de remuneração de energia exportada (EUR/kWh)")
    grp_grid.add_argument("--max-grid-power", type=float, default=3.45,
                          help="Potência contratada/máxima de ligação com a rede elétrica (kW)")

    # --- Opções do Solucionador e Gráficos ---
    grp_solver = parser.add_argument_group('Opções de Execução')
    grp_solver.add_argument("--use-binary", action="store_true",
                            help="Forçar formulação MILP com variáveis binárias Big-M (mais lenta)")
    grp_solver.add_argument("--no-plots", action="store_true",
                            help="Desativar geração de ficheiros de imagem PNG dos gráficos")

    args = parser.parse_args()

    # 1. Instanciar Configurações
    site_cfg = SiteConfig(year=args.year)
    pv_cfg = PVConfig(
        system_type=args.pv_type,
        optimize_capacity=args.optimize_pv,
        min_power_kwp=args.min_pv_kwp,
        max_power_kwp=args.max_pv_kwp,
        peak_power_kwp=args.pv_peak
    )
    bess_cfg = BESSConfig()
    econ_cfg = EconomicsConfig(
        allow_export=args.allow_export,
        export_price_eur_per_kwh=args.export_price,
        max_grid_power_kw=args.max_grid_power
    )
    tariff_cfg = TariffConfig()

    # 2. Carregar perfil de Consumo da Habitação
    if args.load_source == "eredes":
        load_series = load_consumption_eredes(
            consumo_anual_kwh=args.eredes_consumo,
            perfil=args.eredes_perfil,
            sim_year=args.year,
            eredes_year=args.eredes_year
        )
    else:
        load_series = load_consumption_csv(args.consumption_csv, year=args.year)

    # 3. Gerar Produção Solar com PVLIB (Base unitária normalizada de 1 kWp)
    pv_1kwp_series = get_pv_timeseries_1kwp(site_cfg, pv_cfg)

    # 4. Gerar Curva Horária do Tarifário Tri-Horário
    tariff_df = create_trihourly_tariff_dataframe(args.year, tariff_cfg)

    # 5. Sincronização e Alinhamento no Índice Horário Padrão das 8760 Horas
    std_idx = pd.date_range(
        start=f"{args.year}-01-01 00:00:00",
        end=f"{args.year}-12-31 23:00:00",
        freq='h'
    )
    load_series = load_series.reindex(std_idx).ffill().bfill()
    pv_1kwp_series = pv_1kwp_series.reindex(std_idx).fillna(0.0)
    tariff_df = tariff_df.reindex(std_idx).ffill().bfill()

    load_kw = load_series
    pv_kw_1kwp = pv_1kwp_series
    prices = tariff_df['Price_EUR_per_kWh']

    # Custo de referência sem instalação (Grid-Only Baseline)
    baseline_energy_cost = (load_kw * prices).sum()

    # 6. Otimização Convexa (CVXPY + HiGHS)
    opt_results = solve_sizing_optimization(
        load_kw=load_kw,
        pv_kw_1kwp=pv_kw_1kwp,
        prices_eur_per_kwh=prices,
        pv_config=pv_cfg,
        bess_config=bess_cfg,
        econ_config=econ_cfg,
        use_binary_exclusivity=args.use_binary
    )
    opt_results['baseline_energy_cost'] = float(baseline_energy_cost)

    # 7. Análise Financeira, Rainflow e Balanço de Energia
    fin_results = analyze_financials(opt_results, bess_cfg, econ_cfg)

    # 8. Relatório Executivo no Terminal
    print_executive_report(opt_results, fin_results, pv_cfg, bess_cfg, econ_cfg)

    # 9. Geração dos Gráficos de Diagnóstico
    if not args.no_plots:
        print("-> A gerar gráficos analíticos...")
        plot_energy_balance(opt_results['profiles'])
        plot_accumulated_cash_flow(fin_results['accumulated_cash_flow'], fin_results['payback_exact_years'])
        plot_soc_distribution(opt_results['profiles'], opt_results['batt_capacity_kwh'])

    # 10. Gravação dos Resultados Detalhados em Ficheiro CSV
    load_tag = f"eredes_{args.eredes_perfil}_{int(args.eredes_consumo)}kwh" if args.load_source == "eredes" else "csv"
    results_csv = f"sizing_results_{load_tag}_{args.year}.csv"
    opt_results['profiles'].to_csv(results_csv)
    print(f"-> Perfis horários detalhados guardados com sucesso em '{results_csv}'\n")


if __name__ == "__main__":
    main()
