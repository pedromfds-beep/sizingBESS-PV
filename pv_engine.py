"""
Módulo de Simulação Fotovoltaica com PVLIB e Previsões Meteorológicas
=====================================================================

Implementa a cadeia física completa de simulação de produção solar:
1. Posição Solar Astronómica (algoritmo NREL SPA / PSA via pvlib).
2. Geometria Dinâmica (Estrutura Fixa com tilt/azimuth ou Seguidor 1-Eixo HSAT com Backtracking).
3. Transposição de Irradiação no Plano dos Módulos (POA) usando o modelo anisotrópico de Hay-Davies.
4. Modelação Térmica da Célula com o modelo empírico de Faiman (arrefecimento pelo vento).
5. Conversão de Potência DC e AC com o modelo NREL PVWatts e perdas de sistema.
"""
import os
import requests
import pandas as pd
import numpy as np
import pvlib
from config import SiteConfig, PVConfig


def fetch_historical_weather(site: SiteConfig) -> pd.DataFrame:
    """
    Descarrega ou carrega da cache local os dados meteorológicos horários do Open-Meteo Archive:
    - Temperatura ambiente a 2 metros (°C)
    - Velocidade do vento a 10 metros (convertida de km/h para m/s)
    - Irradiação Horizontal Global - GHI (W/m²)
    - Irradiação Direta Normal - DNI (W/m²)
    - Irradiação Difusa Horizontal - DHI (W/m²)
    """
    cache_csv = f"weather_history_{site.year}.csv"
    if os.path.exists(cache_csv):
        print(f"-> A carregar meteorologia histórica a partir de '{cache_csv}'...")
        df = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
        return df

    print(f"-> A descarregar meteorologia de {site.year} para ({site.latitude}, {site.longitude}) via Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": f"{site.year}-01-01",
        "end_date": f"{site.year}-12-31",
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "shortwave_radiation",       # GHI (W/m²)
            "direct_normal_irradiance",   # DNI (W/m²)
            "diffuse_radiation",          # DHI (W/m²)
        ],
        "timezone": "UTC"
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data['hourly'])
    df['ds'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('ds', inplace=True)

    df['temp_air'] = df['temperature_2m']
    df['wind_speed'] = df['wind_speed_10m'] / 3.6  # km/h -> m/s
    df['ghi'] = df['shortwave_radiation'].fillna(0).clip(lower=0)
    df['dni'] = df['direct_normal_irradiance'].fillna(0).clip(lower=0)
    df['dhi'] = df['diffuse_radiation'].fillna(0).clip(lower=0)

    # Guarda cache em formato leve
    df[['temp_air', 'wind_speed', 'ghi', 'dni', 'dhi']].to_csv(cache_csv)
    return df


def simulate_pv_generation(weather_df: pd.DataFrame, site: SiteConfig, pv: PVConfig) -> pd.Series:
    """
    Simula a produção elétrica AC horária (kW) da instalação fotovoltaica.

    Cadeia Física de Modelação:
    ---------------------------
    1. Posição Solar: Zénite aparente, azimute e elevação solar para cada hora.
    2. Geometria: Ângulo de inclinação e azimute da superfície dos módulos.
       - No caso de seguidor 1-eixo com backtracking: roda até +-60° e recua
         ao nascer/pôr do sol para evitar que fiadas projetem sombras nas vizinhas.
    3. POA (Plane Of Array): Modelo de Hay-Davies considera a irradiação direta,
       difusa circumsolar, difusa isotrópica do céu e difusa refletida pelo solo (albedo).
    4. Temperatura de Célula: Modelo Faiman T_cell = T_air + POA / (u0 + u1 * wind_speed).
    5. PVWatts DC: P_dc = P_peak * (POA / 1000) * [1 + gamma * (T_cell - 25°C)].
    6. Potência AC: P_ac = P_dc * (1 - perdas_globais_sistema).
    """
    times = weather_df.index
    print(f"-> A simular produção fotovoltaica ({pv.peak_power_kwp:.3f} kWp, sistema={pv.system_type})...")

    # 1. Posição Solar Astronómica
    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude
    )
    solar_zenith = solpos['apparent_zenith']
    solar_azimuth = solpos['azimuth']
    solar_elevation = solpos['apparent_elevation']

    # 2. Geometria da Superfície dos Módulos
    if pv.system_type == 'single_axis':
        tracking_geom = pvlib.tracking.singleaxis(
            apparent_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            axis_tilt=pv.tracker_tilt,
            axis_azimuth=pv.tracker_azimuth,
            max_angle=pv.tracker_max_angle,
            backtrack=pv.tracker_backtrack,
            gcr=pv.tracker_gcr
        )
        surface_tilt = tracking_geom['surface_tilt'].fillna(0.0)
        surface_azimuth = tracking_geom['surface_azimuth'].fillna(pv.tracker_azimuth)
    else:
        surface_tilt = pd.Series(pv.fixed_tilt, index=times)
        surface_azimuth = pd.Series(pv.fixed_azimuth, index=times)

    # 3. Transposição de Irradiação para o Plano Inclinado (POA - Hay-Davies)
    dni_extra = pvlib.irradiance.get_extra_radiation(times.dayofyear).values
    poa_components = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=weather_df['dni'],
        ghi=weather_df['ghi'],
        dhi=weather_df['dhi'],
        dni_extra=dni_extra,
        albedo=pv.albedo,
        model='haydavies'
    )
    poa_global = poa_components['poa_global'].fillna(0).clip(lower=0)

    # 4. Modelação Térmica da Célula Fotovoltaica (Faiman)
    t_cell = pvlib.temperature.faiman(
        poa_global=poa_global,
        temp_air=weather_df['temp_air'],
        wind_speed=weather_df['wind_speed'],
        u0=25.0,
        u1=6.84
    )

    # 5. Conversão de Potência DC (PVWatts DC com Derate Térmico)
    peak_power_w = pv.peak_power_kwp * 1000.0
    p_dc_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_global,
        temp_cell=t_cell,
        pdc0=peak_power_w,
        gamma_pdc=pv.gamma_pdc,
        temp_ref=25.0
    ).clip(lower=0)

    # 6. Conversão para Potência AC com Perdas Globais do Sistema (inversor, cablagem, sujidade)
    p_ac_w = (p_dc_w * (1.0 - pv.system_losses)).clip(lower=0)

    # Corte noturno estrito (evitar potências residuais com sol abaixo do horizonte)
    night_mask = (solar_elevation <= 0) | (poa_global <= 0)
    p_ac_w[night_mask] = 0.0

    # Converter de Watts para kW
    pv_power_kw = (p_ac_w / 1000.0)
    pv_power_kw.name = 'PV_Generation_kW'
    pv_power_kw.index = pv_power_kw.index.tz_localize(None)

    total_kwh = pv_power_kw.sum()
    print(f"-> Produção solar anual calculada pelo PVLIB: {total_kwh:.2f} kWh/ano (Pico={pv_power_kw.max():.3f} kW)")
    return pv_power_kw


def get_pv_timeseries_1kwp(site: SiteConfig, pv: PVConfig) -> pd.Series:
    """
    Retorna a série horária de produção fotovoltaica normalizada para 1 kWp instalada (kW/kWp).
    Permite escalar ou otimizar linearmente qualquer potência de pico de forma instantânea.
    """
    cache_csv = f"pv_generation_pvlib_1kWp_{site.year}_{pv.system_type}.csv"
    if os.path.exists(cache_csv):
        print(f"-> A carregar produção PV (1 kWp base) a partir de '{cache_csv}'...")
        df = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
        return df['PV_Generation_1kWp']

    # Simular para 1 kWp unitário
    pv_unit = PVConfig(
        system_type=pv.system_type,
        peak_power_kwp=1.0,
        fixed_tilt=pv.fixed_tilt,
        fixed_azimuth=pv.fixed_azimuth,
        tracker_tilt=pv.tracker_tilt,
        tracker_azimuth=pv.tracker_azimuth,
        tracker_max_angle=pv.tracker_max_angle,
        tracker_backtrack=pv.tracker_backtrack,
        tracker_gcr=pv.tracker_gcr,
        albedo=pv.albedo,
        gamma_pdc=pv.gamma_pdc,
        system_losses=pv.system_losses
    )
    weather_df = fetch_historical_weather(site)
    pv_series = simulate_pv_generation(weather_df, site, pv_unit)
    pv_series.name = 'PV_Generation_1kWp'
    pv_series.to_frame().to_csv(cache_csv)
    return pv_series


def get_pv_timeseries(site: SiteConfig, pv: PVConfig) -> pd.Series:
    """
    Retorna a série horária de produção fotovoltaica em kW para a potência configurada.
    """
    pv_1kwp = get_pv_timeseries_1kwp(site, pv)
    pv_scaled = pv_1kwp * pv.peak_power_kwp
    pv_scaled.name = 'PV_Generation_kW'
    return pv_scaled
