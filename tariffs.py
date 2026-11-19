"""
Modelação do Tarifário Elétrico Tri-Horário em Portugal (Regulamentação ERSE)
=============================================================================

Implementa a estrutura horária da tarifa tri-horária em Baixa Tensão Normal (BTN),
respeitando a calendarização oficial da Entidade Reguladora dos Serviços Energéticos (ERSE):
1. Três períodos tarifários:
   - Ponta (Peak): Horas de ponta do consumo nacional, preço mais elevado.
   - Cheias (Full): Período diurno normal de atividade económica.
   - Vazio (Off-Peak): Horas noturnas e madrugadas, preço mais reduzido.
2. Diferenciação Sazonal (Inverno vs. Verão):
   - Início do Verão: Último domingo de março (avanço de 1 hora no relógio).
   - Início do Inverno: Último domingo de outubro (recuo de 1 hora no relógio).
3. Fins-de-semana e Feriados Nacionais:
   - Sábados e Domingos: Todo o dia é faturado no período de Vazio (Off-Peak).
   - Feriados Nacionais Obrigatórios: Tratam-se como domingo (100% Vazio).
"""
import calendar
import pandas as pd
from config import TariffConfig


def create_trihourly_tariff_dataframe(year: int, config: TariffConfig = None) -> pd.DataFrame:
    """
    Gera o perfil horário anual com a classificação do período e o preço respetivo (EUR/kWh).

    Parâmetros:
    -----------
    year : int
        Ano civil da análise (ex: 2025).
    config : TariffConfig, opcional
        Valores dos preços de cada período (€/kWh).
    """
    if config is None:
        config = TariffConfig()

    prices = {
        'Peak': config.peak_price,
        'Full': config.full_price,
        'Off_Peak': config.off_peak_price
    }

    # 1. Obtenção dinâmica dos feriados nacionais em Portugal (inclui feriados móveis como Páscoa e Corpo de Deus)
    try:
        import holidays as pyholidays
        pt_holidays = pyholidays.Portugal(years=year)
        holidays_set = set(pt_holidays.keys())
    except Exception:
        # Fallback de segurança se o pacote 'holidays' não estiver disponível
        holidays_set = {
            f'{year}-01-01', f'{year}-04-18', f'{year}-04-20', f'{year}-04-25', f'{year}-05-01',
            f'{year}-06-10', f'{year}-06-19', f'{year}-08-15', f'{year}-10-05', f'{year}-11-01',
            f'{year}-12-01', f'{year}-12-08', f'{year}-12-25'
        }

    # 2. Determinação exata das datas de mudança de hora legal em Portugal (último domingo de março e outubro)
    march_last_day = pd.Timestamp(year, 3, calendar.monthrange(year, 3)[1])
    summer_start = march_last_day - pd.Timedelta(days=(march_last_day.dayofweek - 6) % 7) + pd.Timedelta(hours=1)

    october_last_day = pd.Timestamp(year, 10, calendar.monthrange(year, 10)[1])
    winter_start = october_last_day - pd.Timedelta(days=(october_last_day.dayofweek - 6) % 7) + pd.Timedelta(hours=1)

    # 3. Construção do índice horário contínuo das 8760 horas do ano
    hourly_index = pd.date_range(start=f'{year}-01-01 00:00:00', end=f'{year}-12-31 23:00:00', freq='h')
    df = pd.DataFrame(index=hourly_index)
    df.index.name = 'DateTime'

    df['Day_of_Week'] = df.index.dayofweek  # 0=Segunda, 4=Sexta, 5=Sábado, 6=Domingo
    df['Holiday'] = df.index.strftime('%Y-%m-%d').isin(holidays_set)
    df['Hour'] = df.index.hour

    # Definição sazonal
    df['Season'] = 'Winter'
    df.loc[(df.index >= summer_start) & (df.index < winter_start), 'Season'] = 'Summer'

    # Por defeito, todo o período é Vazio (Off-Peak)
    df['Period'] = 'Off_Peak'

    # Apenas os dias úteis que não sejam feriados nacionais têm períodos de Ponta e Cheia
    weekday_mask = ~df['Day_of_Week'].isin([5, 6]) & ~df['Holiday']

    # 4. Regras de Inverno (Horário Legal de Inverno - Dias Úteis):
    # - Ponta:  09:00 - 11:00 e 17:00 - 20:00
    # - Cheias: 07:00 - 09:00, 11:00 - 17:00 e 20:00 - 22:00
    winter_mask = weekday_mask & (df['Season'] == 'Winter')
    df.loc[winter_mask & (((df['Hour'] >= 9) & (df['Hour'] < 11)) | ((df['Hour'] >= 17) & (df['Hour'] < 20))), 'Period'] = 'Peak'
    df.loc[winter_mask & (((df['Hour'] >= 7) & (df['Hour'] < 9)) | ((df['Hour'] >= 11) & (df['Hour'] < 17)) | ((df['Hour'] >= 20) & (df['Hour'] < 22))), 'Period'] = 'Full'

    # 5. Regras de Verão (Horário Legal de Verão - Dias Úteis):
    # - Ponta:  10:00 - 13:00 e 18:00 - 21:00
    # - Cheias: 08:00 - 10:00, 13:00 - 18:00 e 21:00 - 22:00
    summer_mask = weekday_mask & (df['Season'] == 'Summer')
    df.loc[summer_mask & (((df['Hour'] >= 10) & (df['Hour'] < 13)) | ((df['Hour'] >= 18) & (df['Hour'] < 21))), 'Period'] = 'Peak'
    df.loc[summer_mask & (((df['Hour'] >= 8) & (df['Hour'] < 10)) | ((df['Hour'] >= 13) & (df['Hour'] < 18)) | ((df['Hour'] >= 21) & (df['Hour'] < 22))), 'Period'] = 'Full'

    # Mapeamento do preço em EUR/kWh
    df['Price_EUR_per_kWh'] = df['Period'].map(prices)
    return df[['Season', 'Period', 'Price_EUR_per_kWh']]
