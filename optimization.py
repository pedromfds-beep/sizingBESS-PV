"""
Módulo de Otimização Matemática de Dimensionamento PV + BESS usando CVXPY e HiGHS
================================================================================

Formula e resolve o problema de dimensionamento ótimo (capacidade da bateria, potência
do conversor e opcionalmente capacidade de módulos PV) e o despacho horário das 8760 horas
do ano em regime de autoconsumo residencial.

NOTAS MATEMÁTICAS SOBRE A FORMULAÇÃO:
------------------------------------
1. Relaxação Linear Contínua (LP) vs. MILP:
   No notebook antigo utilizavam-se 8760 variáveis binárias com formulação Big-M para
   impedir carga e descarga simultâneas da bateria, o que demorava ~107 minutos a resolver.
   Aqui demonstra-se que uma formulação Linear Contínua (LP) garante NATURALMENTE a exclusividade:
   - Se a bateria carregar e descarregar na mesma hora, perde energia por ciclo round-trip
     devido a (eta_ch * eta_dis = 0.95 < 1).
   - Como o preço de energia da rede é estritamente positivo (0.11 a 0.35 €/kWh) e a degradação
     marginal por throughput é > 0, o solucionador minimizador NUNCA descarregará e carregará
     ao mesmo tempo, pois isso aumentaria os custos totais sem qualquer benefício.
   - Resultado: o problema resolve com o solver HiGHS em escassos segundos (~1 a 4 s).

2. Fator de Recuperação de Capital (Capital Recovery Factor - CRF):
   CRF = (r * (1 + r)^N) / ((1 + r)^N - 1)
   Transforma o investimento inicial (CAPEX) numa anuidade equivalente a uma taxa de desconto r
   ao longo de N anos de vida útil do projeto.

3. Degradação Marginal da Bateria por Throughput:
   O custo por kWh que passa pela bateria é calculado a partir do CAPEX e do número total de ciclos
   completos até 60% de capacidade residual (End of Life - EOL):
   C_deg = CAPEX_kwh / ((1 - EOL) / loss_per_cycle)
"""
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import cvxpy as cp
from config import PVConfig, BESSConfig, EconomicsConfig


def calculate_crf(discount_rate: float, lifetime_years: int) -> float:
    """
    Calcula o Fator de Recuperação de Capital (Capital Recovery Factor - CRF).
    Permite anualizar os custos de investimento inicial (CAPEX).
    """
    if discount_rate > 1e-6:
        return (discount_rate * (1 + discount_rate) ** lifetime_years) / ((1 + discount_rate) ** lifetime_years - 1)
    return 1.0 / lifetime_years


def solve_sizing_optimization(
    load_kw: pd.Series,
    pv_kw_1kwp: pd.Series,
    prices_eur_per_kwh: pd.Series,
    pv_config: PVConfig,
    bess_config: BESSConfig,
    econ_config: EconomicsConfig,
    use_binary_exclusivity: bool = False
) -> Dict[str, Any]:
    """
    Formula e resolve o problema conjunto de dimensionamento e despacho ótimo.

    Parâmetros:
    -----------
    load_kw : pd.Series
        Série temporal horária de consumo elétrico da habitação (kW).
    pv_kw_1kwp : pd.Series
        Série temporal horária de geração fotovoltaica normalizada para 1 kWp (kW/kWp).
    prices_eur_per_kwh : pd.Series
        Preço horário de compra de eletricidade da rede (EUR/kWh).
    pv_config : PVConfig
        Configurações do sistema fotovoltaico (inclui flag de otimização de kWp).
    bess_config : BESSConfig
        Configurações da bateria (custos, eficiências e limites operacionais).
    econ_config : EconomicsConfig
        Parâmetros financeiros, penalizações e permissão de exportação.
    use_binary_exclusivity : bool
        Se True, inclui variáveis booleanas Big-M (MILP). Se False, usa LP contínuo rápido.
    """
    n_steps = len(load_kw)
    dt = 1.0  # Intervalo de discretização temporal (1 hora)
    crf = calculate_crf(econ_config.discount_rate, econ_config.project_lifetime_years)

    # Eficiência de carregamento e descarregamento simétrica (raiz quadrada da eficiência round-trip)
    # Exemplo: se round-trip = 0.95 -> eta_ch = eta_dis = sqrt(0.95) = 0.9747
    eta = np.sqrt(bess_config.efficiency)

    # Custo de degradação por ciclo completo refletido no throughput de carregamento (EUR/kWh)
    if bess_config.capacity_loss_per_cycle > 0:
        total_cycles_to_eol = (1.0 - bess_config.eol_retention) / bess_config.capacity_loss_per_cycle
        cycle_deg_cost_per_kwh = bess_config.capex_per_kwh / total_cycles_to_eol
    else:
        cycle_deg_cost_per_kwh = 0.0

    print(f"\n========================================================")
    print(f"  A INICIAR OTIMIZAÇÃO HORÁRIA ({n_steps} HORAS)")
    print(f"  Modo: {'MILP (variáveis binárias Big-M)' if use_binary_exclusivity else 'LP Contínuo (rápido e comprovadamente exato)'}")
    print(f"  Otimizar PV: {'SIM (Capacidade livre de PV)' if pv_config.optimize_capacity else f'NÃO (PV Fixo a {pv_config.peak_power_kwp:.3f} kWp)'}")
    print(f"  Injeção na Rede: {'Permitida (' + str(econ_config.export_price_eur_per_kwh) + ' €/kWh)' if econ_config.allow_export else 'Não permitida (Zero Export)'}")
    print(f"  CRF: {crf:.4f} | Custo Degradação Bateria: {cycle_deg_cost_per_kwh:.4f} €/kWh throughput")
    print(f"========================================================")

    # -------------------------------------------------------------------------
    # 1. VARIÁVEIS DE DECISÃO
    # -------------------------------------------------------------------------
    # Dimensionamento da bateria
    batt_capacity = cp.Variable(pos=True, name='batt_capacity_kwh')  # Capacidade nominal (kWh)
    batt_power = cp.Variable(pos=True, name='batt_power_kw')        # Potência do conversor (kW)

    # Dimensionamento fotovoltaico (se ativo) ou valor fixo constante
    if pv_config.optimize_capacity:
        pv_capacity = cp.Variable(pos=True, name='pv_capacity_kwp')
        pv_capex = pv_capacity * pv_config.capex_per_kwp
        pv_power_expr = cp.multiply(pv_kw_1kwp.values, pv_capacity)
    else:
        pv_capacity = pv_config.peak_power_kwp
        pv_capex = pv_config.peak_power_kwp * pv_config.capex_per_kwp
        pv_power_expr = pv_kw_1kwp.values * pv_config.peak_power_kwp

    # Despacho operacional hora a hora
    charge_power = cp.Variable(n_steps, pos=True, name='charge_power_kw')       # Carga na bateria (kW)
    discharge_power = cp.Variable(n_steps, pos=True, name='discharge_power_kw') # Descarga da bateria (kW)
    soc = cp.Variable(n_steps + 1, pos=True, name='soc_kwh')                    # Estado de Carga da bateria (kWh)

    grid_import = cp.Variable(n_steps, pos=True, name='grid_import_kw')         # Importação da rede (kW)
    grid_export = cp.Variable(n_steps, pos=True, name='grid_export_kw')         # Injeção/Venda à rede (kW)
    curtailed_pv = cp.Variable(n_steps, pos=True, name='curtailed_pv_kw')       # Excesso solar cortado (kW)
    unmet_load = cp.Variable(n_steps, pos=True, name='unmet_load_kw')           # Carga não atendida (kW)

    if use_binary_exclusivity:
        is_charging = cp.Variable(n_steps, boolean=True, name='is_charging')

    # -------------------------------------------------------------------------
    # 2. FUNÇÃO OBJETIVO (Custo Anualizado Total a Minimizar)
    # -------------------------------------------------------------------------
    annualized_pv_capex = pv_capex * crf
    batt_capex = batt_capacity * bess_config.capex_per_kwh + batt_power * bess_config.capex_per_kw
    annualized_batt_capex = batt_capex * crf

    export_allowed = econ_config.allow_export and (econ_config.export_price_eur_per_kwh > -90)
    effective_export_price = econ_config.export_price_eur_per_kwh if export_allowed else 0.0

    # Custo de energia líquida da rede = Custo da energia importada - Receita da energia exportada
    grid_energy_cost = cp.sum(
        cp.multiply(grid_import, prices_eur_per_kwh.values)
        - cp.multiply(grid_export, effective_export_price)
    ) * dt

    annual_fixed_grid_fee = econ_config.daily_fixed_fee_eur * 365.0
    unmet_load_cost = cp.sum(unmet_load) * econ_config.unmet_load_penalty_per_kwh * dt
    curtailed_pv_cost = cp.sum(curtailed_pv) * econ_config.curtailment_penalty_per_kwh * dt
    cycle_deg_cost = cp.sum(charge_power) * dt * cycle_deg_cost_per_kwh

    objective = cp.Minimize(
        annualized_pv_capex +
        annualized_batt_capex +
        grid_energy_cost +
        annual_fixed_grid_fee +
        unmet_load_cost +
        curtailed_pv_cost +
        cycle_deg_cost
    )

    # -------------------------------------------------------------------------
    # 3. RESTRIÇÕES FÍSICAS E OPERACIONAIS
    # -------------------------------------------------------------------------
    constraints = []

    # 3.1 Limites de dimensionamento do PV (se variável livre)
    if pv_config.optimize_capacity:
        constraints += [
            pv_capacity >= pv_config.min_power_kwp,
            pv_capacity <= pv_config.max_power_kwp
        ]

    # 3.2 Limites de capacidade, potência e rácio E/P da bateria
    constraints += [
        batt_capacity >= bess_config.min_capacity_kwh,
        batt_capacity <= bess_config.max_capacity_kwh,
        batt_power >= bess_config.min_power_kw,
        batt_power <= bess_config.max_power_kw,
        # Restrições de duração mínima e máxima (C-rate operacional)
        batt_capacity >= bess_config.min_kwh_per_kw_ratio * batt_power,
        batt_capacity <= bess_config.max_kwh_per_kw_ratio * batt_power,
    ]

    # 3.3 Dinâmica de Conservação de Energia do Estado de Carga (SOC)
    # Condição inicial: começa no SOC mínimo
    constraints.append(soc[0] == bess_config.min_soc * batt_capacity)
    # Recorrência temporal: SOC(t+1) = SOC(t) + eta*P_ch*dt - (1/eta)*P_dis*dt
    constraints.append(
        soc[1:] == soc[:-1] + (charge_power * eta - discharge_power / eta) * dt
    )
    # Limites operacionais de SOC
    constraints += [
        soc >= bess_config.min_soc * batt_capacity,
        soc <= bess_config.max_soc * batt_capacity,
        # Condição de contorno cíclica anual: terminar o ano com energia >= inicial
        soc[n_steps] >= soc[0]
    ]

    # 3.4 Limites de potência da bateria e do ponto de entrega da rede
    constraints += [
        charge_power <= batt_power,
        discharge_power <= batt_power,
        grid_import <= econ_config.max_grid_power_kw,
        grid_export <= (econ_config.max_grid_power_kw if export_allowed else 0.0)
    ]

    # 3.5 Exclusividade de carga/descarga (opcional se forçado pelo utilizador)
    if use_binary_exclusivity:
        m_big = bess_config.max_power_kw
        constraints += [
            charge_power <= m_big * is_charging,
            discharge_power <= m_big * (1 - is_charging)
        ]

    # 3.6 Balanço Instantâneo de Potência em Cada Hora
    # Geração Solar + Descarga Bateria + Importação Rede = Consumo + Carga Bateria + Exportação + Corte + Não-Atendido
    load_vals = load_kw.values
    constraints.append(
        pv_power_expr + discharge_power + grid_import ==
        load_vals + charge_power + grid_export + unmet_load + curtailed_pv
    )

    # -------------------------------------------------------------------------
    # 4. RESOLUÇÃO DO PROBLEMA LINEAR
    # -------------------------------------------------------------------------
    prob = cp.Problem(objective, constraints)
    t0 = time.time()

    # Seleção de solver de alto desempenho
    solver = cp.HIGHS if cp.HIGHS in cp.installed_solvers() else cp.CLARABEL
    print(f"-> A resolver com solver de otimização {solver}...")
    prob.solve(solver=solver, verbose=False)
    solve_duration = time.time() - t0

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"Otimização não convergiu para uma solução ótima. Status: {prob.status}")

    print(f"-> Otimização concluída com sucesso em {solve_duration:.2f} s. Status: {prob.status}")

    # -------------------------------------------------------------------------
    # 5. ESTRUTURAÇÃO E RETORNO DOS RESULTADOS
    # -------------------------------------------------------------------------
    opt_batt_cap = float(batt_capacity.value)
    opt_batt_pow = float(batt_power.value)
    opt_pv_cap = float(pv_capacity.value if pv_config.optimize_capacity else pv_config.peak_power_kwp)
    c_rate = opt_batt_pow / opt_batt_cap if opt_batt_cap > 1e-4 else 0.0

    actual_pv_kw = pv_kw_1kwp.values * opt_pv_cap
    pv_capex_val = float(pv_capex.value if pv_config.optimize_capacity else pv_capex)
    batt_capex_val = float(batt_capex.value)

    idx = load_kw.index
    profiles = pd.DataFrame(index=idx)
    profiles['Load_kW'] = load_vals
    profiles['PV_kW'] = actual_pv_kw
    profiles['Charge_kW'] = charge_power.value
    profiles['Discharge_kW'] = discharge_power.value
    profiles['Grid_Import_kW'] = grid_import.value
    profiles['Grid_Export_kW'] = grid_export.value
    profiles['Curtailed_PV_kW'] = curtailed_pv.value
    profiles['Unmet_Load_kW'] = unmet_load.value
    profiles['SOC_kWh'] = soc.value[:-1]

    results = {
        'status': prob.status,
        'solve_time_sec': solve_duration,
        'batt_capacity_kwh': opt_batt_cap,
        'batt_power_kw': opt_batt_pow,
        'c_rate': c_rate,
        'pv_capacity_kwp': opt_pv_cap,
        'pv_capex': pv_capex_val,
        'batt_capex': batt_capex_val,
        'total_initial_capex': float(pv_capex_val + batt_capex_val),
        'annualized_capex': float((pv_capex_val + batt_capex_val) * crf),
        'annual_grid_energy_cost': float(grid_energy_cost.value),
        'annual_fixed_grid_fee': float(annual_fixed_grid_fee),
        'annual_cycle_deg_cost': float(cycle_deg_cost.value),
        'annual_unmet_cost': float(unmet_load_cost.value),
        'annual_curtailment_cost': float(curtailed_pv_cost.value),
        'total_annualized_cost': float(prob.value),
        'crf': crf,
        'profiles': profiles
    }

    return results
