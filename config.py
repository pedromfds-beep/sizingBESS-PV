"""
Configuração do Sistema de Dimensionamento Ótimo PV + BESS (Autoconsumo Residencial)
====================================================================================
Centraliza todos os parâmetros físicos, técnicos, regulatórios e económicos
em dataclasses fortemente tipadas.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class SiteConfig:
    """Parâmetros geográficos e temporais da instalação."""
    latitude: float = 38.529       # Évora, Portugal (região com elevada irradiação solar DNI)
    longitude: float = -8.010
    altitude: float = 250.0        # Altitude em metros (relevante para pressão atmosférica e massa de ar)
    year: int = 2025               # Ano de referência temporal para simulação


@dataclass
class PVConfig:
    """Configurações e características técnicas do sistema fotovoltaico."""
    system_type: Literal['fixed', 'single_axis'] = 'fixed'
    optimize_capacity: bool = False  # Se True: otimiza o kWp ótimo conjuntamente com a bateria
    min_power_kwp: float = 0.0       # Limite mínimo de potência instalada (kWp)
    max_power_kwp: float = 10.0      # Limite máximo de potência instalada (telhado / ligação BT)
    peak_power_kwp: float = 0.405    # Potência instalada fixa se optimize_capacity=False (ex: 1 painel 405 W)

    # Parâmetros de Geometria:
    fixed_tilt: float = 37.0         # Inclinação ótima anual para a latitude do sul/centro de Portugal
    fixed_azimuth: float = 180.0     # 180° = Orientação a Sul geográfico verdadeiro
    tracker_tilt: float = 0.0        # Eixo horizontal se seguidor de 1 eixo (HSAT)
    tracker_azimuth: float = 180.0   # Orientação do eixo Norte-Sul
    tracker_max_angle: float = 60.0  # Rotação mecânica máxima (+-60°)
    tracker_backtrack: bool = True   # Algoritmo de backtracking para evitar sombreamento mútuo entre fiadas
    tracker_gcr: float = 0.35        # Ground Cover Ratio (largura do módulo / distância entre eixos)
    albedo: float = 0.20             # Reflectividade típica do solo envolvente (20% para solo/erva seca)

    # Parâmetros Físicos e Elétricos (PVWatts / Célula Silício Cristalino):
    gamma_pdc: float = -0.004        # Coeficiente térmico de potência (-0.40%/°C acima de 25°C em STC)
    system_losses: float = 0.14      # 14% de perdas globais AC (cabos, sujidade, inversor, tolerâncias)
    capex_per_kwp: float = 500.0     # Custo de investimento chave-na-mão por kWp (EUR/kWp)
    degradation_rate_year: float = 0.008  # Degradação anual linear da produção dos módulos (0.8%/ano)


@dataclass
class BESSConfig:
    """Parâmetros técnicos e económicos do Sistema de Armazenamento por Bateria (BESS)."""
    name: str = 'Li-ion'
    capex_per_kwh: float = 200.0     # Custo de investimento proporcional à capacidade (EUR/kWh)
    capex_per_kw: float = 250.0      # Custo de investimento proporcional à potência do inversor/PCS (EUR/kW)
    efficiency: float = 0.95         # Eficiência round-trip de ciclo completo (ida e volta: 95%)
    min_soc: float = 0.05            # Limite mínimo de segurança do Estado de Carga (5% SOC)
    max_soc: float = 0.95            # Limite máximo de segurança do Estado de Carga (95% SOC)
    min_capacity_kwh: float = 0.5    # Capacidade mínima admissível (kWh)
    max_capacity_kwh: float = 30.0   # Capacidade máxima admissível para BTN residencial (kWh)
    min_power_kw: float = 0.5        # Potência mínima do inversor da bateria (kW)
    max_power_kw: float = 10.0       # Potência máxima do inversor da bateria (kW)
    min_kwh_per_kw_ratio: float = 0.5   # Mínimo de 30 minutos de autonomia à potência nominal
    max_kwh_per_kw_ratio: float = 10.0  # Máximo de 10 horas de duração à potência nominal
    capacity_loss_per_cycle: float = 0.00002  # Perda de capacidade por ciclo completo (0.002%/ciclo)
    eol_retention: float = 0.60      # Critério de fim de vida útil da bateria (60% da capacidade nominal)


@dataclass
class EconomicsConfig:
    """Parâmetros financeiros, regulatórios e de ligação à rede elétrica."""
    project_lifetime_years: int = 15  # Horizonte temporal de análise económica do projeto (anos)
    discount_rate: float = 0.07      # Taxa de atualização / desconto financeiro anual (WACC: 7.0%)
    daily_fixed_fee_eur: float = 0.46  # Termo fixo diário da tarifa de eletricidade (~13.8 EUR/mês)
    allow_export: bool = False       # Se True: permite injeção de excedentes na rede com remuneração
    export_price_eur_per_kwh: float = 0.05  # Preço médio líquido de venda de excedentes (EUR/kWh)
    unmet_load_penalty_per_kwh: float = 100.0  # Penalização estrita para garantir 100% de fornecimento
    curtailment_penalty_per_kwh: float = 0.0    # Corte de excesso solar tem custo zero (dissipação segura)
    max_grid_power_kw: float = 3.45  # Potência contratada com o operador de rede (ex: 3.45 kVA em BTN)


@dataclass
class TariffConfig:
    """Preços da energia na tarifa tri-horária em Portugal (ERSE)."""
    peak_price: float = 0.35         # Período de Ponta (EUR/kWh) - horários de maior tensão na rede
    full_price: float = 0.18         # Período de Cheias (EUR/kWh) - horário diurno padrão
    off_peak_price: float = 0.11     # Período de Vazio (EUR/kWh) - madrugada e fins-de-semana
