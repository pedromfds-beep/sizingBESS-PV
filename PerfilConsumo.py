"""
Simulação de Perfis de Consumo BTN com Base nos Dados Oficiais da E-REDES
========================================================================

Este módulo extrai e processa os perfis normalizados de Baixa Tensão Normal (BTN)
publicados anualmente pelo operador de rede de distribuição português (E-REDES),
ao abrigo das diretivas regulamentares da ERSE.

CLASSES DE PERFIL BTN:
----------------------
1. 'BTN_A':
   - Potência Contratada: Superior a 13.8 kVA.
   - Destinatários: Moradias de grande dimensão, explorações rurais ou pequenos comércios.

2. 'BTN_B':
   - Potência Contratada: Inferior ou igual a 13.8 kVA.
   - Consumo Anual: Superior a 7.140 kWh/ano (utilizadores intensivos).

3. 'BTN_C' (Habitação Familiar Típica):
   - Potência Contratada: Inferior ou igual a 13.8 kVA (3.45, 4.6, 6.9 ou 10.35 kVA).
   - Consumo Anual: Inferior ou igual a 7.140 kWh/ano.
   - Representa mais de 85% dos consumidores domésticos em Portugal.
   - Bandas de consumo: DB (1.000–2.500 kWh), DC (2.500–5.000 kWh), DD (5.000–15.000 kWh).
"""
import os
import pandas as pd

# Mapeamento de ficheiros e sheets para os anos disponíveis
_FICHEIROS_EREDES = {
    2024: {
        'ficheiro': 'Perfil_Consumo_Injeção_E-REDES_2024.xlsx',
        'sheet': 'Perfis Consumo e Injeção',
    },
    2025: {
        'ficheiro': 'Perfil_Consumo_Injeção_E-REDES_2025.xlsx',
        'sheet': '2025',
    },
    2026: {
        'ficheiro': 'Perfil_Consumo_Injeção_E-REDES_2026.xlsx',
        'sheet': '2026',
    },
}

# Índices das colunas no Excel oficial (0-based)
_COLUNAS_BTN = {
    'BTN_A': 4,
    'BTN_B': 5,
    'BTN_C': 6,
}


def processar_perfil_btn(
    consumo_anual_kwh: float,
    perfil: str = 'BTN_C',
    ano: int = 2025,
    caminho_ficheiro: str = None,
    base_dir: str = None
) -> pd.Series:
    """
    Gera uma série horária de potência média de consumo (kW) calibrada para um consumo anual.

    Otimização de Desempenho:
    -------------------------
    Para evitar ler o ficheiro Excel pesado (~5 MB) em todas as execuções, o perfil normalizado
    (com soma unitária = 1.0) é gravado em cache num ficheiro CSV local:
    `eredes_normalized_{perfil}_{ano}.csv`. Nas execuções seguintes, a leitura é quase instantânea (~5 ms).

    Parâmetros:
    -----------
    consumo_anual_kwh : float
        Consumo anual total da habitação em kWh (ex: 3500 kWh/ano).
    perfil : str
        Classe BTN: 'BTN_A', 'BTN_B' ou 'BTN_C' (omissão).
    ano : int
        Ano do perfil oficial (2024, 2025 ou 2026).
    caminho_ficheiro : str, opcional
        Caminho explícito para o Excel (opcional).
    base_dir : str, opcional
        Diretório base onde se encontram os ficheiros.
    """
    perfil_upper = perfil.upper()
    if perfil_upper not in _COLUNAS_BTN:
        raise ValueError(f"Perfil '{perfil}' inválido. Escolha entre: {list(_COLUNAS_BTN.keys())}")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Ficheiro de cache rápido para os coeficientes normalizados
    cache_csv = os.path.join(base_dir, f"eredes_normalized_{perfil_upper}_{ano}.csv")
    if os.path.exists(cache_csv):
        print(f"-> A carregar perfil normalizado E-REDES {perfil_upper} ({ano}) a partir da cache rápida...")
        df_cache = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
        # Multiplicar o vetor normalizado horário pelo consumo anual desejado (kW médio por hora = kWh/h)
        load_hourly = df_cache['normalized_weight'] * consumo_anual_kwh
        load_hourly.name = 'Load_kW'
        print(f"   Consumo anual E-REDES ({perfil_upper}): {load_hourly.sum():.1f} kWh/ano | "
              f"Pico: {load_hourly.max():.3f} kW")
        return load_hourly

    # Se não existir cache, lê o Excel bruto oficial da E-REDES
    if caminho_ficheiro is None:
        if ano not in _FICHEIROS_EREDES:
            raise ValueError(f"Ano {ano} não suportado. Anos disponíveis: {list(_FICHEIROS_EREDES.keys())}")
        cfg = _FICHEIROS_EREDES[ano]
        caminho_ficheiro = os.path.join(base_dir, cfg['ficheiro'])
        sheet_name = cfg['sheet']
    else:
        import openpyxl
        wb = openpyxl.load_workbook(caminho_ficheiro, read_only=True)
        sheet_name = wb.sheetnames[0]
        wb.close()

    if not os.path.exists(caminho_ficheiro):
        raise FileNotFoundError(f"Ficheiro E-REDES não encontrado: '{caminho_ficheiro}'")

    col_idx = _COLUNAS_BTN[perfil_upper]
    print(f"-> A processar ficheiro Excel E-REDES ({os.path.basename(caminho_ficheiro)})...")

    # Extração de dados (linhas 0 a 3 são cabeçalhos institucionais)
    df_raw = pd.read_excel(caminho_ficheiro, sheet_name=sheet_name, header=None)
    dados = df_raw.iloc[4:].copy()

    df = pd.DataFrame()
    df['data_base'] = pd.to_datetime(dados.iloc[:, 1], errors='coerce')

    # Limpeza da string horária (remover sufixos 'a'/'b' de mudança de hora de outono)
    df['hora_str'] = (
        dados.iloc[:, 2]
        .astype(str)
        .str.strip()
        .str.replace(r'[ab]$', '', regex=True)
    )
    df['coeficiente'] = pd.to_numeric(dados.iloc[:, col_idx], errors='coerce')
    df = df.dropna(subset=['data_base', 'coeficiente']).copy()

    # Tratamento da hora '24:00' (marcador do fim do dia -> 00:00 do dia seguinte)
    def parse_dt(row):
        d = row['data_base']
        h = row['hora_str']
        if h in ('24:00', '24:00:00'):
            return (d + pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' 00:00'
        return d.strftime('%Y-%m-%d') + ' ' + h

    df['datetime'] = pd.to_datetime(df.apply(parse_dt, axis=1))
    df = df.set_index('datetime').sort_index()

    # Normalização estrita: a soma de todos os pesos anuais é igual a 1.0
    soma_pesos = df['coeficiente'].sum()
    df['fator_normalizado'] = df['coeficiente'] / soma_pesos

    # Converter de energia quarto-horária (15 min) para potência média por hora (kW)
    # 15 min kWh -> kW instantâneo = kWh * 4. A média horária de 4 intervalos de 15 min dá a potência média horária
    df['potencia_kw_15min'] = df['fator_normalizado'] * 4.0
    load_hourly_normalized = df['potencia_kw_15min'].resample('h').mean()
    # Ajuste fino para garantir que a soma das 8760 horas é exatamente 1.0
    load_hourly_normalized = load_hourly_normalized / load_hourly_normalized.sum()

    # Guardar cache leve em CSV para que futuras execuções demorem 5 ms
    df_cache_out = pd.DataFrame({'normalized_weight': load_hourly_normalized})
    df_cache_out.to_csv(cache_csv)
    print(f"-> Perfil normalizado guardado em cache rápida: '{os.path.basename(cache_csv)}'")

    # Escalar pelo consumo anual
    load_hourly = load_hourly_normalized * consumo_anual_kwh
    load_hourly.name = 'Load_kW'

    print(f"   Consumo anual E-REDES ({perfil_upper}): {load_hourly.sum():.1f} kWh/ano | "
          f"Pico: {load_hourly.max():.3f} kW | Pontos: {len(load_hourly)}")
    return load_hourly
