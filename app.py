"""
Aplicação Web Streamlit: Dimensionamento Ótimo de PV + BESS Residencial (Portugal)
==================================================================================

Interface gráfica interativa para dimensionamento e análise técnico-económica de
sistemas fotovoltaicos com armazenamento por bateria para autoconsumo em Portugal.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SiteConfig, PVConfig, BESSConfig, EconomicsConfig, TariffConfig
from pv_engine import get_pv_timeseries_1kwp
from tariffs import create_trihourly_tariff_dataframe
from optimization import solve_sizing_optimization
from financial import analyze_financials
from PerfilConsumo import processar_perfil_btn
from run_sizing import load_consumption_eredes, load_consumption_csv

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dimensionamento PV + BESS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual moderno com CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 4px solid #1e88e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNÇÕES DE CACHE PARA DESEMPENHO INSTANTÂNEO
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_cached_pv_1kwp(year: int, system_type: str, tilt: float, azimuth: float):
    site = SiteConfig(year=year)
    pv = PVConfig(system_type=system_type, fixed_tilt=tilt, fixed_azimuth=azimuth)
    return get_pv_timeseries_1kwp(site, pv)


@st.cache_data(show_spinner=False)
def get_cached_tariff(year: int, peak: float, full: float, off_peak: float):
    cfg = TariffConfig(peak_price=peak, full_price=full, off_peak_price=off_peak)
    return create_trihourly_tariff_dataframe(year, cfg)


@st.cache_data(show_spinner=False)
def get_cached_load(source: str, annual_kwh: float, perfil: str, year: int, csv_path: str = "Load House 13.csv"):
    if source == "eredes":
        return load_consumption_eredes(consumo_anual_kwh=annual_kwh, perfil=perfil, sim_year=year, eredes_year=year)
    else:
        return load_consumption_csv(csv_path, year=year)


# ==============================================================================
# BARRA LATERAL - PARÂMETROS E CONFIGURAÇÃO
# ==============================================================================
st.sidebar.title("⚙️ Parâmetros do Sistema")

# --- 1. Perfil de Consumo ---
st.sidebar.subheader("🏠 1. Consumo da Habitação")
load_source = st.sidebar.radio(
    "Fonte dos Dados de Consumo:",
    ["Perfil Normativo E-REDES (Recomendado)", "Ficheiro CSV com Medições Reais"],
    index=0
)

sim_year = st.sidebar.selectbox("Ano de Simulação:", [2025, 2024, 2026], index=0)

if load_source.startswith("Perfil Normativo"):
    source_key = "eredes"
    eredes_perfil = st.sidebar.selectbox(
        "Classe de Perfil BTN (E-REDES):",
        ["BTN_C", "BTN_B", "BTN_A"],
        index=0,
        help="BTN_C: Habitação familiar típica (potência <= 13.8 kVA, consumo <= 7140 kWh/ano)"
    )
    annual_consumption_kwh = st.sidebar.number_input(
        "Consumo Anual Total (kWh/ano):",
        min_value=500.0,
        max_value=30000.0,
        value=3500.0,
        step=250.0,
        help="Consumo médio residencial em Portugal: ~3500 kWh/ano"
    )
    csv_file_path = "Load House 13.csv"
else:
    source_key = "csv"
    eredes_perfil = "BTN_C"
    annual_consumption_kwh = 3500.0
    csv_file_path = st.sidebar.text_input("Caminho do Ficheiro CSV:", value="Load House 13.csv")

# --- 2. Sistema Fotovoltaico (PV) ---
st.sidebar.subheader("☀️ 2. Sistema Fotovoltaico (PV)")
pv_mode = st.sidebar.radio(
    "Dimensionamento de PV:",
    ["Otimizar Potência Livremente (kWp)", "Fixar Potência Instalada (kWp)"],
    index=0
)

optimize_pv = pv_mode.startswith("Otimizar")
if optimize_pv:
    max_pv_kwp = st.sidebar.slider("Limite Máximo de Telhado (kWp):", 1.0, 15.0, 5.0, step=0.5)
    fixed_pv_kwp = 0.405
else:
    max_pv_kwp = 10.0
    fixed_pv_kwp = st.sidebar.number_input("Potência Fixa de PV (kWp):", min_value=0.2, max_value=20.0, value=1.62, step=0.405)

pv_structure = st.sidebar.selectbox("Tipo de Estrutura:", ["Estrutura Fixa (Sul 37°)", "Seguidor Solar 1-Eixo (HSAT)"], index=0)
pv_system_type = "single_axis" if "Seguidor" in pv_structure else "fixed"
pv_capex_kwp = st.sidebar.number_input("CAPEX Fotovoltaico (€/kWp):", min_value=200.0, max_value=1500.0, value=500.0, step=50.0)

# --- 3. Bateria (BESS) ---
st.sidebar.subheader("🔋 3. Bateria de Lítio (BESS)")
batt_capex_kwh = st.sidebar.number_input("CAPEX Capacidade (€/kWh):", min_value=100.0, max_value=800.0, value=200.0, step=25.0)
batt_capex_kw = st.sidebar.number_input("CAPEX Potência Conversor (€/kW):", min_value=50.0, max_value=600.0, value=250.0, step=25.0)
batt_roundtrip_eff = st.sidebar.slider("Eficiência Round-Trip (%):", 80, 99, 95, step=1) / 100.0

# --- 4. Rede e Excedentes ---
st.sidebar.subheader("🔌 4. Rede & Venda de Excedentes")
allow_grid_export = st.sidebar.checkbox("Permitir Injeção / Venda de Excedentes (UPAC)", value=True)
if allow_grid_export:
    export_price_eur = st.sidebar.number_input("Remuneração de Excedente (€/kWh):", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
else:
    export_price_eur = -100.0

grid_power_kw = st.sidebar.number_input("Potência Contratada (kW):", min_value=1.15, max_value=41.4, value=3.45, step=1.15)

# --- 5. Tarifário Elétrico ---
with st.sidebar.expander("Tarifário Tri-Horário (€/kWh)"):
    peak_price = st.number_input("Ponta (€/kWh):", value=0.35, step=0.01)
    full_price = st.number_input("Cheias (€/kWh):", value=0.18, step=0.01)
    off_peak_price = st.number_input("Vazio (€/kWh):", value=0.11, step=0.01)
    daily_fee = st.number_input("Termo Fixo (€/dia):", value=0.46, step=0.02)
    discount_rate = st.slider("Taxa de Desconto / WACC (%):", 2.0, 15.0, 7.0, step=0.5) / 100.0
    project_years = st.slider("Vida Útil do Projeto (anos):", 10, 25, 15, step=1)


# ==============================================================================
# BOTÃO DE EXECUÇÃO DA OTIMIZAÇÃO
# ==============================================================================
st.title("⚡ Dimensionamento Ótimo de PV + BESS Residencial")
st.markdown("Otimização técnico-económica com **PVLIB**, **Tarifário Tri-Horário Português (ERSE)**, **Perfis E-REDES** e resolução convexa via **CVXPY / HiGHS**.")

# Botão principal na sidebar para disparar a otimização
st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Executar Otimização", use_container_width=True, type="primary")

# Construir as configurações (necessárias tanto para a otimização como para os gráficos)
pv_cfg = PVConfig(
    system_type=pv_system_type,
    optimize_capacity=optimize_pv,
    max_power_kwp=max_pv_kwp,
    min_power_kwp=0.0,
    peak_power_kwp=fixed_pv_kwp,
    capex_per_kwp=pv_capex_kwp
)
bess_cfg = BESSConfig(
    capex_per_kwh=batt_capex_kwh,
    capex_per_kw=batt_capex_kw,
    efficiency=batt_roundtrip_eff
)
econ_cfg = EconomicsConfig(
    project_lifetime_years=project_years,
    discount_rate=discount_rate,
    daily_fixed_fee_eur=daily_fee,
    allow_export=allow_grid_export,
    export_price_eur_per_kwh=export_price_eur,
    max_grid_power_kw=grid_power_kw
)

# Executar a otimização quando o botão é pressionado; guardar resultados em session_state
if run_btn:
    with st.spinner("A carregar dados e a executar otimização horário-anual (8760 h)..."):
        # Carregar dados pré-calculados em cache
        pv_1kwp = get_cached_pv_1kwp(year=sim_year, system_type=pv_system_type, tilt=37.0, azimuth=180.0)
        tariff_df = get_cached_tariff(year=sim_year, peak=peak_price, full=full_price, off_peak=off_peak_price)
        load_series = get_cached_load(source=source_key, annual_kwh=annual_consumption_kwh, perfil=eredes_perfil, year=sim_year, csv_path=csv_file_path)

        # Sincronização temporal de 8760 horas
        std_idx = pd.date_range(start=f"{sim_year}-01-01 00:00:00", end=f"{sim_year}-12-31 23:00:00", freq='h')
        load_kw = load_series.reindex(std_idx).ffill().bfill()
        pv_kw_1kwp = pv_1kwp.reindex(std_idx).fillna(0.0)
        prices = tariff_df.reindex(std_idx).ffill().bfill()['Price_EUR_per_kWh']

        baseline_energy_cost = (load_kw * prices).sum()

        # Otimização com CVXPY + HiGHS
        opt_results = solve_sizing_optimization(
            load_kw=load_kw,
            pv_kw_1kwp=pv_kw_1kwp,
            prices_eur_per_kwh=prices,
            pv_config=pv_cfg,
            bess_config=bess_cfg,
            econ_config=econ_cfg,
            use_binary_exclusivity=False
        )
        opt_results['baseline_energy_cost'] = float(baseline_energy_cost)

        # Pós-processamento financeiro e Rainflow
        fin_results = analyze_financials(opt_results, bess_cfg, econ_cfg)

        # Guardar resultados no session_state para persistir entre re-runs
        st.session_state['opt_results'] = opt_results
        st.session_state['fin_results'] = fin_results
        st.session_state['optimize_pv'] = optimize_pv
        st.session_state['project_years'] = project_years

    st.success("✅ Otimização concluída com sucesso!")


# ==============================================================================
# VERIFICAÇÃO: mostrar resultados apenas se a otimização já foi executada
# ==============================================================================
if 'opt_results' not in st.session_state:
    # Estado inicial: app arranca sem resultados, com mensagem de boas-vindas
    st.info(
        "👋 **Bem-vindo!** Configure os parâmetros na barra lateral e pressione "
        "**🚀 Executar Otimização** para iniciar o dimensionamento."
    )
    st.stop()  # Interrompe aqui — nenhum gráfico é desenhado

# Recuperar resultados guardados no session_state
opt_results = st.session_state['opt_results']
fin_results = st.session_state['fin_results']
profiles = opt_results['profiles']
# Usar project_years da última otimização para consistência nos textos
_project_years = st.session_state.get('project_years', project_years)
_optimize_pv = st.session_state.get('optimize_pv', optimize_pv)

payback_txt = (
    f"{fin_results['payback_exact_years']:.2f} anos"
    if fin_results['payback_exact_years']
    else f"> {_project_years} anos"
)


# ==============================================================================
# PAINEL DE INDICADORES PRINCIPAIS (KPIs)
# ==============================================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pv_status_label = "(Otimizado)" if _optimize_pv else "(Fixo)"
    st.metric(
        label=f"☀️ Potência Solar PV {pv_status_label}",
        value=f"{opt_results['pv_capacity_kwp']:.2f} kWp",
        delta=f"{fin_results['total_pv_kwh']:.0f} kWh/ano"
    )

with col2:
    st.metric(
        label="🔋 Capacidade Bateria",
        value=f"{opt_results['batt_capacity_kwh']:.2f} kWh",
        delta=f"Potência: {opt_results['batt_power_kw']:.2f} kW"
    )

with col3:
    st.metric(
        label="💶 Investimento Inicial (CAPEX)",
        value=f"{opt_results['total_initial_capex']:.0f} €",
        delta=f"PV: {opt_results['pv_capex']:.0f} € | BESS: {opt_results['batt_capex']:.0f} €",
        delta_color="off"
    )

with col4:
    st.metric(
        label="💰 Poupança Anual Líquida",
        value=f"{fin_results['annual_savings']:.2f} €/ano",
        delta=f"Baseline: {fin_results['baseline_annual_cost']:.0f} €/ano"
    )

with col5:
    st.metric(
        label="⏱️ Retorno do Investimento (Payback)",
        value=payback_txt,
        delta=f"Autossuficiência: {fin_results['self_sufficiency_rate_pct']:.1f}%"
    )

st.markdown("---")

# ==============================================================================
# ABAS COM GRÁFICOS INTERATIVOS PLOTLY
# ==============================================================================
tab_power, tab_econ, tab_battery = st.tabs([
    "📈 Balanço de Potência Global",
    "💵 Viabilidade Económica & Cash Flow",
    "🔋 Dinâmica da Bateria (Potência & SoC)"
])

# ------------------------------------------------------------------------------
# ABA 1: BALANÇO DE POTÊNCIA
# ------------------------------------------------------------------------------
with tab_power:
    st.subheader("Balanço Horário de Potência (kW)")
    col_w1, col_w2 = st.columns([1, 3])
    with col_w1:
        preset_window = st.selectbox(
            "Período de Demonstração Rápida:",
            ["Semana de Primavera (Maio)", "Semana de Verão (Julho)", "Semana de Inverno (Janeiro)", "Personalizado (Dias do Ano)"],
            index=0
        )
        if preset_window == "Semana de Primavera (Maio)":
            start_hour, end_hour = 3200, 3200 + 168
        elif preset_window == "Semana de Verão (Julho)":
            start_hour, end_hour = 4800, 4800 + 168
        elif preset_window == "Semana de Inverno (Janeiro)":
            start_hour, end_hour = 200, 200 + 168
        else:
            start_hour = st.slider("Hora Inicial do Ano:", 0, 8760 - 168, 3000)
            end_hour = start_hour + st.slider("Duração da Janela (horas):", 24, 720, 168)

    sub_df = profiles.iloc[start_hour:end_hour]

    fig_power = go.Figure()

    # Fontes de energia em áreas empilhadas
    fig_power.add_trace(go.Scatter(
        x=sub_df.index, y=sub_df['PV_kW'],
        mode='lines', line=dict(width=0), fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.7)', name='Produção Solar (PV)'
    ))
    fig_power.add_trace(go.Scatter(
        x=sub_df.index, y=sub_df['PV_kW'] + sub_df['Discharge_kW'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(33, 150, 243, 0.7)', name='Descarga da Bateria'
    ))
    fig_power.add_trace(go.Scatter(
        x=sub_df.index, y=sub_df['PV_kW'] + sub_df['Discharge_kW'] + sub_df['Grid_Import_kW'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(244, 67, 54, 0.65)', name='Importação da Rede'
    ))

    # Curvas de consumo e fluxos de armazenamento
    fig_power.add_trace(go.Scatter(
        x=sub_df.index, y=sub_df['Load_kW'],
        mode='lines', line=dict(color='#1a237e', width=2.5), name='Consumo da Habitação'
    ))
    fig_power.add_trace(go.Scatter(
        x=sub_df.index, y=-sub_df['Charge_kW'],
        mode='lines', line=dict(color='#9c27b0', width=1.8, dash='dash'), name='Carregamento da Bateria (-)'
    ))
    if (sub_df['Grid_Export_kW'] > 1e-3).any():
        fig_power.add_trace(go.Scatter(
            x=sub_df.index, y=-sub_df['Grid_Export_kW'],
            mode='lines', line=dict(color='#ff9800', width=2, dash='dot'), name='Exportação / Venda à Rede (-)'
        ))

    fig_power.update_layout(
        title="Balanço Dinâmico de Energia (Fontes vs. Consumos)",
        xaxis_title="Data e Hora",
        yaxis_title="Potência (kW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=480
    )
    st.plotly_chart(fig_power, use_container_width=True)


# ------------------------------------------------------------------------------
# ABA 2: VIABILIDADE ECONÓMICA & CASH FLOW
# ------------------------------------------------------------------------------
with tab_econ:
    st.subheader("Análise Financeira ao Longo da Vida Útil")

    col_ec1, col_ec2 = st.columns([2, 1])

    with col_ec1:
        cash_flow_acc = fin_results['accumulated_cash_flow']
        years = list(range(len(cash_flow_acc)))
        bar_colors = ['#e53935' if v < 0 else '#43a047' for v in cash_flow_acc]

        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=years,
            y=cash_flow_acc,
            marker_color=bar_colors,
            text=[f"{v:.0f} €" for v in cash_flow_acc],
            textposition='outside',
            name="Cash Flow Acumulado"
        ))

        if fin_results['payback_exact_years']:
            pb_year = fin_results['payback_exact_years']
            fig_cf.add_vline(
                x=pb_year, line_width=2.5, line_dash="dash", line_color="#b71c1c",
                annotation_text=f"Payback: {pb_year:.2f} anos", annotation_position="top left"
            )

        fig_cf.update_layout(
            title="Fluxo de Caixa Acumulado Líquido (EUR)",
            xaxis_title="Ano do Projeto",
            yaxis_title="EUR Acumulado",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig_cf, use_container_width=True)

    with col_ec2:
        st.markdown("### Resumo Financeiro Anual")
        st.markdown(f"""
        - **Custo Baseline (Apenas Rede):** `{fin_results['baseline_annual_cost']:.2f} €/ano`
        - **Custo Operacional do Sistema (OPEX):** `{fin_results['system_annual_opex']:.2f} €/ano`
        - **Poupança Anual Líquida:** `+{fin_results['annual_savings']:.2f} €/ano`
        - **Receita Anual de Exportação:** `+{fin_results['annual_export_revenue']:.2f} €/ano`
        - **Custo Total Anualizado (c/ amortização CRF):** `{opt_results['total_annualized_cost']:.2f} €/ano`
        - **Período de Retorno Simples:** `{payback_txt}`
        """)

        # Gráfico Donut de Repartição do CAPEX
        fig_capex = go.Figure(data=[go.Pie(
            labels=['Fotovoltaico (PV)', 'Bateria (BESS)'],
            values=[opt_results['pv_capex'], opt_results['batt_capex']],
            hole=.4,
            marker_colors=['#ffb300', '#1e88e5']
        )])
        fig_capex.update_layout(title="Repartição do CAPEX Inicial", height=240, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_capex, use_container_width=True)


# ------------------------------------------------------------------------------
# ABA 3: ESTADO DE CARGA E CICLOS DA BATERIA
# ------------------------------------------------------------------------------
with tab_battery:
    st.subheader("Dinâmica Operacional da Bateria e Degradação")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        # Gráfico Donut: Origem da Carga da Bateria
        fig_charge_source = go.Figure(data=[go.Pie(
            labels=['Excedente Solar Fotovoltaico', 'Rede no Vazio Noturno'],
            values=[fin_results['charge_from_pv_kwh'], fin_results['charge_from_grid_kwh']],
            hole=.45,
            marker_colors=['#43a047', '#5c6bc0']
        )])
        fig_charge_source.update_layout(
            title=f"Fontes de Carregamento da Bateria (Total: {fin_results['total_charge_kwh']:.0f} kWh/ano)",
            height=320,
            template="plotly_white"
        )
        st.plotly_chart(fig_charge_source, use_container_width=True)

    with col_b2:
        st.markdown("### Métricas de Fadiga e Ciclagem (Rainflow ASTM E1049-85)")
        st.markdown(f"""
        - **Ciclos Completos Equivalentes (EFC):** `{fin_results['annual_rainflow_efc']:.1f} ciclos/ano`
        - **Ciclos ao Longo dos {_project_years} Anos:** `{fin_results['lifetime_efc']:.0f} ciclos`
        - **Taxa de Descarga Operacional (C-rate):** `{opt_results['c_rate']:.2f} C`
        - **Critério de Fim de Vida Útil (EOL):** `60% de retenção de capacidade`
        - **Custo Marginal de Degradação:** `{bess_cfg.capex_per_kwh / ((1.0 - bess_cfg.eol_retention) / bess_cfg.capacity_loss_per_cycle):.4f} €/kWh throughput`
        """)

    # Amostra Temporal: Potência de Carga/Descarga e Evolução do SoC
    if opt_results['batt_capacity_kwh'] > 1e-3:
        st.markdown("---")
        st.subheader("Amostra Dinâmica da Bateria: Carga/Descarga e SoC")

        col_bs1, col_bs2 = st.columns([1, 2])
        with col_bs1:
            bess_preset = st.selectbox(
                "Período de Demonstração da Bateria:",
                ["Semana de Primavera (Maio)", "Semana de Verão (Julho)", "Semana de Inverno (Janeiro)", "Personalizado (Dias do Ano)"],
                index=0,
                key="bess_sample_preset"
            )
            if bess_preset == "Semana de Primavera (Maio)":
                b_start, b_end = 3200, 3200 + 168
            elif bess_preset == "Semana de Verão (Julho)":
                b_start, b_end = 4800, 4800 + 168
            elif bess_preset == "Semana de Inverno (Janeiro)":
                b_start, b_end = 200, 200 + 168
            else:
                b_start = st.slider("Hora Inicial do Ano:", 0, 8760 - 168, 3000, key="b_start_sl")
                b_end = b_start + st.slider("Duração da Janela (horas):", 24, 720, 168, key="b_dur_sl")

        sub_bess = profiles.iloc[b_start:b_end]
        batt_cap = opt_results['batt_capacity_kwh']
        sub_soc_pct = (sub_bess['SOC_kWh'] / batt_cap) * 100.0

        # Gráfico 1: Variação da Potência de Carga e Descarga da Bateria
        fig_power_bess = go.Figure()
        fig_power_bess.add_trace(go.Scatter(
            x=sub_bess.index,
            y=sub_bess['Charge_kW'],
            mode='lines',
            line=dict(color='#2e7d32', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 125, 50, 0.25)',
            name='Potência de Carga (kW)'
        ))
        fig_power_bess.add_trace(go.Scatter(
            x=sub_bess.index,
            y=sub_bess['Discharge_kW'],
            mode='lines',
            line=dict(color='#ef6c00', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 108, 0, 0.25)',
            name='Potência de Descarga (kW)'
        ))
        fig_power_bess.update_layout(
            title="Variação da Potência da Bateria: Carga e Descarga (kW)",
            xaxis_title="Data e Hora",
            yaxis_title="Potência (kW)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=370
        )
        st.plotly_chart(fig_power_bess, use_container_width=True)

        # Gráfico 2: Estado de Carga (SoC % e Energia Armazenada em kWh)
        fig_soc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_soc.add_trace(
            go.Scatter(
                x=sub_bess.index,
                y=sub_soc_pct,
                mode='lines',
                line=dict(color='#1565c0', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(21, 101, 192, 0.15)',
                name='Estado de Carga (SoC %)'
            ),
            secondary_y=False
        )
        fig_soc.add_trace(
            go.Scatter(
                x=sub_bess.index,
                y=sub_bess['SOC_kWh'],
                mode='lines',
                line=dict(color='#00838f', width=1.5, dash='dot'),
                name='Energia Armazenada (kWh)'
            ),
            secondary_y=True
        )

        min_pct = bess_cfg.min_soc * 100.0
        max_pct = bess_cfg.max_soc * 100.0

        fig_soc.add_hline(
            y=min_pct, line_dash="dash", line_color="#d32f2f", line_width=1.5,
            annotation_text=f"SoC Mínimo ({min_pct:.0f}%)", annotation_position="bottom right",
            secondary_y=False
        )
        fig_soc.add_hline(
            y=max_pct, line_dash="dash", line_color="#2e7d32", line_width=1.5,
            annotation_text=f"SoC Máximo ({max_pct:.0f}%)", annotation_position="top right",
            secondary_y=False
        )

        fig_soc.update_xaxes(title_text="Data e Hora")
        fig_soc.update_yaxes(title_text="Estado de Carga (SoC %)", range=[0, 105], secondary_y=False)
        fig_soc.update_yaxes(title_text="Energia Armazenada (kWh)", range=[0, batt_cap * 1.05], secondary_y=True)

        fig_soc.update_layout(
            title="Evolução do Estado de Carga (SoC % e Energia Armazenada)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=370
        )
        st.plotly_chart(fig_soc, use_container_width=True)
    else:
        st.info("💡 Nenhuma bateria foi dimensionada/instalada no sistema ótimo atual (Capacidade = 0 kWh).")
#Pedro Miguel Sousa, pedro.fortunas.sousa@uevora.pt, pedromfds@gmail.com

