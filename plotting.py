"""
Módulo de Visualização Gráfica do Sistema PV + BESS
===================================================

Gera gráficos de alta qualidade para análise do comportamento operacional e económico:
1. Balanço de Potência Instantâneo (kW) numa janela temporal amostral representativa.
2. Fluxo de Caixa Acumulado (EUR) ao longo do tempo de vida com ponto de Payback destacado.
3. Histogramas de Distribuição do Estado de Carga (SOC %) comparando Inverno e Verão.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_energy_balance(profiles: pd.DataFrame, window_hours: int = 96, output_file: str = "energy_balance_sample.png"):
    """
    Gera um gráfico de balanço de potência empilhado numa janela temporal amostral.
    Mostra as fontes de energia (PV, Descarga da Bateria, Importação da Rede) e
    os consumos (Carga da habitação, Carga da Bateria e Injeção na Rede).
    """
    n_total = len(profiles)
    # Selecionar uma janela temporal representativa (ex: início de maio com boa alternância solar/carga)
    start_idx = min(3000, max(0, n_total - window_hours))
    end_idx = start_idx + min(window_hours, n_total)
    sub = profiles.iloc[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Fontes de energia (empilhadas na área positiva)
    ax.stackplot(
        sub.index,
        sub['PV_kW'].values,
        sub['Discharge_kW'].values,
        sub['Grid_Import_kW'].values,
        labels=['Produção Fotovoltaica (PV)', 'Descarga da Bateria (BESS)', 'Importação da Rede'],
        colors=['#4caf50', '#2196f3', '#f44336'],
        alpha=0.75
    )

    # Curvas de consumo e fluxos de armazenamento
    ax.plot(sub.index, sub['Load_kW'], label='Consumo da Habitação', color='#1a237e', linewidth=2.2)
    ax.plot(sub.index, -sub['Charge_kW'], label='Carregamento da Bateria (-)', color='#9c27b0', linestyle='--', linewidth=1.5)

    if (sub['Grid_Export_kW'] > 1e-3).any():
        ax.plot(sub.index, -sub['Grid_Export_kW'], label='Exportação / Venda à Rede (-)', color='#ff9800', linestyle=':', linewidth=1.8)

    ax.set_ylabel('Potência (kW)', fontsize=11)
    ax.set_title(f'Balanço de Potência da Habitação (Amostra de {window_hours} horas consecutivas)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.22, 1.0), fontsize=9)
    plt.xticks(rotation=25)
    plt.tight_layout()

    fig.savefig(output_file, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Gráfico guardado em '{output_file}'")


def plot_accumulated_cash_flow(accumulated_cash_flow: list, payback_years: float, output_file: str = "accumulated_cash_flow.png"):
    """
    Gera o gráfico de barras do fluxo de caixa acumulado, destacando o momento de Payback.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(range(len(accumulated_cash_flow)))
    colors = ['#e53935' if val < 0 else '#43a047' for val in accumulated_cash_flow]

    bars = ax.bar(years, accumulated_cash_flow, color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)

    ax.set_xlabel('Ano do Projeto', fontsize=11)
    ax.set_ylabel('Fluxo de Caixa Acumulado Líquido (EUR)', fontsize=11)
    ax.set_title('Fluxo de Caixa Acumulado e Retorno do Investimento (Payback)', fontsize=12, fontweight='bold')
    ax.set_xticks(years)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Linha vertical a marcar o momento de break-even
    if payback_years is not None and payback_years < len(accumulated_cash_flow):
        ax.axvline(
            x=payback_years,
            color='#b71c1c',
            linestyle='--',
            linewidth=2.0,
            label=f'Payback Simples: {payback_years:.2f} anos'
        )
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Gráfico guardado em '{output_file}'")


def plot_soc_distribution(profiles: pd.DataFrame, batt_capacity_kwh: float, output_file: str = "soc_distribution.png"):
    """
    Gera histogramas comparativos da distribuição do Estado de Carga (SOC %) entre Verão e Inverno.
    """
    if batt_capacity_kwh <= 1e-4:
        return

    soc_pct = (profiles['SOC_kWh'] / batt_capacity_kwh) * 100.0

    # Meses quentes com maior irradiação solar vs. meses frios
    summer_mask = profiles.index.month.isin([5, 6, 7, 8, 9])
    winter_mask = ~summer_mask

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].hist(soc_pct[winter_mask], bins=35, color='#00838f', edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribuição de SOC - Inverno / Primavera', fontsize=11)
    axes[0].set_xlabel('Estado de Carga (SOC %)', fontsize=10)
    axes[0].set_ylabel('Frequência (Horas)', fontsize=10)
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(soc_pct[summer_mask], bins=35, color='#f57c00', edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribuição de SOC - Verão', fontsize=11)
    axes[1].set_xlabel('Estado de Carga (SOC %)', fontsize=10)
    axes[1].grid(True, alpha=0.4)

    fig.suptitle('Distribuição Estatística do Estado de Carga da Bateria (SOC)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Gráfico guardado em '{output_file}'")
