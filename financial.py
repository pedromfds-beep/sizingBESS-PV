"""
Módulo de Análise Financeira, Balanço Energético e Degradação por Ciclos Rainflow
================================================================================

Implementa o pós-processamento analítico após a resolução da otimização:
1. Balanço Energético Completo (kWh): autoconsumo, autossuficiência e fontes de carga da bateria.
2. Contagem de Ciclos Rainflow (ASTM E1049-85):
   A degradação por ciclo depende da amplitude das oscilações de carga/descarga. O algoritmo
   Rainflow decompõe o perfil temporal não-periódico de SOC em ciclos equivalentes completos (EFC),
   permitindo estimar o tempo de vida físico útil da bateria até ao limite de 60% (EOL).
3. Avaliação Financeira e Cash Flow:
   - Comparação contra o cenário de referência (Grid-Only Baseline).
   - Fluxo de caixa anual e acumulado ao longo da vida do projeto.
   - Período de retorno simples (Payback) com interpolação sub-anual linear exata.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
import rainflow
from config import BESSConfig, EconomicsConfig


def analyze_financials(
    opt_results: Dict[str, Any],
    bess_config: BESSConfig,
    econ_config: EconomicsConfig
) -> Dict[str, Any]:
    """
    Executa a análise financeira completa, balanço de energia e degradação por fadiga (Rainflow).
    """
    profiles = opt_results['profiles']
    dt = 1.0  # resolução horária (1 h)

    # -------------------------------------------------------------------------
    # 1. BALANÇO ENERGÉTICO ANUAL (kWh)
    # -------------------------------------------------------------------------
    total_load_kwh = profiles['Load_kW'].sum() * dt
    total_pv_kwh = profiles['PV_kW'].sum() * dt
    total_grid_import_kwh = profiles['Grid_Import_kW'].sum() * dt
    total_grid_export_kwh = profiles['Grid_Export_kW'].sum() * dt
    total_charge_kwh = profiles['Charge_kW'].sum() * dt
    total_discharge_kwh = profiles['Discharge_kW'].sum() * dt
    total_curtailed_kwh = profiles['Curtailed_PV_kW'].sum() * dt
    total_unmet_kwh = profiles['Unmet_Load_kW'].sum() * dt

    # -------------------------------------------------------------------------
    # 2. DECOMPOSIÇÃO DAS FONTES DE CARGA DA BATERIA (Solar vs. Rede)
    # -------------------------------------------------------------------------
    # Prioridade física: a bateria carrega primeiro a partir do excedente solar (PV - Carga)
    excess_pv = (profiles['PV_kW'] - profiles['Load_kW']).clip(lower=0.0)
    charge_from_pv = np.minimum(profiles['Charge_kW'], excess_pv)
    remaining_charge = profiles['Charge_kW'] - charge_from_pv
    # O restante carregamento é efetuado a partir da rede (habitualmente no período de Vazio noturno)
    charge_from_grid = np.minimum(remaining_charge, profiles['Grid_Import_kW'])

    annual_charge_from_pv_kwh = charge_from_pv.sum() * dt
    annual_charge_from_grid_kwh = charge_from_grid.sum() * dt

    # -------------------------------------------------------------------------
    # 3. CONTAGEM DE CICLOS RAINFLOW (ASTM E1049-85)
    # -------------------------------------------------------------------------
    # O algoritmo Rainflow extrai amplitudes de ciclos de histerese do SOC (kWh)
    # EFC = soma ( amplitude_ciclo / capacidade_nominal * contagem )
    batt_cap = opt_results['batt_capacity_kwh']
    soc_series = profiles['SOC_kWh'].values
    annual_efc = 0.0

    if batt_cap > 1e-4 and len(soc_series) > 1:
        try:
            cycles = rainflow.extract_cycles(soc_series)
            for cycle_range, _, count, _, _ in cycles:
                annual_efc += (cycle_range / batt_cap) * count
        except Exception as e:
            print(f"Aviso no cálculo Rainflow: {e}")
            annual_efc = (total_discharge_kwh / batt_cap) if batt_cap > 0 else 0.0
    lifetime_efc = annual_efc * econ_config.project_lifetime_years

    # -------------------------------------------------------------------------
    # 4. CENÁRIO DE REFERÊNCIA (Grid Only Baseline - Sem PV nem Bateria)
    # -------------------------------------------------------------------------
    baseline_energy_cost = opt_results.get('baseline_energy_cost')
    if baseline_energy_cost is None:
        baseline_energy_cost = total_load_kwh * 0.18

    baseline_annual_cost = baseline_energy_cost + (econ_config.daily_fixed_fee_eur * 365.0)

    # Custo operacional anual líquido do sistema otimizado (OPEX de energia + termo fixo + degradação)
    system_annual_opex = (
        opt_results['annual_grid_energy_cost'] +
        opt_results['annual_fixed_grid_fee'] +
        opt_results['annual_cycle_deg_cost']
    )

    annual_savings = baseline_annual_cost - system_annual_opex
    total_capex = opt_results['total_initial_capex']

    # -------------------------------------------------------------------------
    # 5. FLUXO DE CAIXA E PAYBACK COM INTERPOLAÇÃO LINEAR
    # -------------------------------------------------------------------------
    lifetime_years = econ_config.project_lifetime_years
    # Ano 0: Desembolso inicial (-CAPEX). Anos 1..N: Poupança anual líquida
    cash_flow = [-total_capex] + [annual_savings] * lifetime_years
    accumulated_cash_flow = np.cumsum(cash_flow).tolist()

    payback_display = None
    for year_idx, acc_val in enumerate(accumulated_cash_flow):
        if acc_val >= 0 and payback_display is None and year_idx > 0:
            prev_acc = accumulated_cash_flow[year_idx - 1]
            curr_flow = cash_flow[year_idx]
            if curr_flow > 0 and prev_acc < 0:
                fraction = abs(prev_acc) / curr_flow
                payback_display = (year_idx - 1) + fraction
            else:
                payback_display = float(year_idx)

    # -------------------------------------------------------------------------
    # 6. MÉTRICAS DE AUTOCONSUMO E AUTOSSUFICIÊNCIA
    # -------------------------------------------------------------------------
    # Autoconsumo: fração da produção fotovoltaica consumida no local (direta ou via bateria)
    self_consumption_pv_kwh = total_pv_kwh - total_grid_export_kwh - total_curtailed_kwh
    self_consumption_rate = (self_consumption_pv_kwh / total_pv_kwh * 100.0) if total_pv_kwh > 0 else 0.0

    # Autossuficiência: fração do consumo da habitação coberta pelo sistema local
    self_sufficiency_rate = ((total_load_kwh - total_grid_import_kwh) / total_load_kwh * 100.0) if total_load_kwh > 0 else 0.0

    # Receita de exportação se injeção na rede for permitida
    export_allowed = econ_config.allow_export and (econ_config.export_price_eur_per_kwh > -90)
    effective_export_price = econ_config.export_price_eur_per_kwh if export_allowed else 0.0
    annual_export_revenue = total_grid_export_kwh * effective_export_price

    return {
        'total_load_kwh': total_load_kwh,
        'total_pv_kwh': total_pv_kwh,
        'total_grid_import_kwh': total_grid_import_kwh,
        'total_grid_export_kwh': total_grid_export_kwh,
        'annual_export_revenue': annual_export_revenue,
        'export_price_eur_per_kwh': effective_export_price,
        'allow_export': export_allowed,
        'total_charge_kwh': total_charge_kwh,
        'total_discharge_kwh': total_discharge_kwh,
        'total_curtailed_kwh': total_curtailed_kwh,
        'total_unmet_kwh': total_unmet_kwh,
        'charge_from_pv_kwh': annual_charge_from_pv_kwh,
        'charge_from_grid_kwh': annual_charge_from_grid_kwh,
        'self_consumption_rate_pct': self_consumption_rate,
        'self_sufficiency_rate_pct': self_sufficiency_rate,
        'annual_rainflow_efc': annual_efc,
        'lifetime_efc': lifetime_efc,
        'baseline_annual_cost': baseline_annual_cost,
        'system_annual_opex': system_annual_opex,
        'annual_savings': annual_savings,
        'simple_payback_years': total_capex / annual_savings if annual_savings > 0 else float('inf'),
        'payback_exact_years': payback_display,
        'cash_flow': cash_flow,
        'accumulated_cash_flow': accumulated_cash_flow
    }
