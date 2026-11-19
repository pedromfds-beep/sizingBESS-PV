# sizingBESS-PV
Dimensionamento ótimo de um sistema solar fotovoltaico (PV) com bateria (BESS) para autoconsumo residencial em Portugal.

Gera o perfil horário da casa usando perfis normativos E-REDES (BTN_A/B/C) ou CSV real. Modela a produção PV hora a hora com PVLIB (irradiação Open-Meteo, modelo térmico Faiman, PVWatts). Aplica tarifas tri-horárias ERSE (Ponta/Cheias/Vazio) com calendário de feriados nacionais. Resolve um problema de programação linear convexa (CVXPY + HiGHS) que decide simultaneamente a potência PV, capacidade/potência da bateria e os fluxos horários (carga, descarga, importação, exportação) que minimizam o custo anualizado. Calcula poupança anual, payback, cash flow acumulado e degradação da bateria (Rainflow ASTM E1049-85).

Endereço da aplicação: https://sizingbess-pv.streamlit.app/

