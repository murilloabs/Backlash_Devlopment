# %%
# ==============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES DE DIRETÓRIOS
# ==============================================================================
import sys
# Adicione aqui os caminhos para a pasta do ROSS e a pasta onde está a classe Backlash
sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\ross_dev_backlash\ross')
sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\teste_backlash_gemini')

import os
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ross as rs


# Importa a classe Backlash e a função de FFT nativas da sua implementação
from backlash import Backlash, compute_dfft

# Identifica o diretório de execução para salvar os resultados nativos
try:
    main_file = sys.modules['__main__'].__file__
    diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
except AttributeError:
    # Fallback para ambientes interativos (como Jupyter Notebook)
    diretorio_execucao = os.getcwd()

OUTPUT_PLOT_DIR = os.path.join(diretorio_execucao, "validation_plots")
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

print(f"Diretório base de execução: {diretorio_execucao}")
print(f"Diretório dos gráficos gerados: {OUTPUT_PLOT_DIR}")


# %%
# ==============================================================================
# FUNÇÕES DE PLOTAGEM (SEM ARQUIVOS DE REFERÊNCIA)
# ==============================================================================
def save_plot_formats(fig, base_filename):
    """Salva a figura gerada em HTML e tenta salvar em PDF."""
    html_path = os.path.join(OUTPUT_PLOT_DIR, base_filename)
    pdf_path = os.path.join(OUTPUT_PLOT_DIR, base_filename.replace('.html', '.pdf'))
    
    fig.write_html(html_path)
    try:
        fig.write_image(pdf_path, format='pdf')
    except Exception:
        pass # Silencia erros caso a biblioteca 'kaleido' não esteja instalada

def plot_time_response(sim_x, sim_y, title, x_label, y_label, output_filename, 
                       x_range=None, sim_scale_y=1.0, constant_ref=None):
    """Plota a resposta no domínio do tempo, com opção de adicionar a linha do valor constante."""
    fig = go.Figure()

    # Aplica o zoom na janela de tempo especificada (se houver)
    if x_range:
        mask = (sim_x >= x_range[0]) & (sim_x <= x_range[1])
        x_plot = sim_x[mask]
        y_plot = sim_y[mask] * sim_scale_y
    else:
        x_plot = sim_x
        y_plot = sim_y * sim_scale_y

    # Curva da simulação
    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Modelo Implementado', 
                             line=dict(color='#0072B2', width=2.5)))

    # Linha de referência (ex: distância nominal, folga estática, etc.)
    if constant_ref is not None:
        fig.add_trace(go.Scatter(x=[x_plot[0], x_plot[-1]], y=[constant_ref, constant_ref], 
                                 mode='lines', name='Valor de referência (Constante)', 
                                 line=dict(color='#000000', width=2, dash='dash')))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    if x_range: fig.update_xaxes(range=x_range)

    save_plot_formats(fig, output_filename)

def plot_fft(sim_time, sim_signal, title, x_label, y_label, output_filename, sim_scale_y=1.0, x_range=None):
    """Calcula e plota a FFT usando a função nativa do seu backlash.py."""
    fig = go.Figure()

    # Corta o transiente inicial (primeira metade) para pegar o regime permanente
    meio = len(sim_time) // 2
    freq_hz, amp = compute_dfft(sim_signal[meio:], sim_time[meio:], freq_unit="Hz", window="hann")
    
    freq_khz = freq_hz / 1000.0  
    amp_scaled = amp * sim_scale_y

    fig.add_trace(go.Scatter(x=freq_khz, y=amp_scaled, mode='lines', name='Modelo Implementado (FFT)', 
                             line=dict(color='#0072B2', width=2)))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    if x_range: fig.update_xaxes(range=x_range)

    save_plot_formats(fig, output_filename)

def plot_poincare(pasta_saida_simulacao, title, x_label, y_label, output_filename):
    """Lê o CSV do Mapa de Poincaré gerado nativamente pela simulação e o plota."""
    fig = go.Figure()
    poincare_csv = os.path.join(pasta_saida_simulacao, "nao_linear_dados_poincare.csv")
    
    if os.path.exists(poincare_csv):
        df_sim = pd.read_csv(poincare_csv)
        p_delta_um = df_sim["Delta_um"].values
        p_delta_dot_mms = df_sim["dDelta_dt_mm_s"].values
        
        fig.add_trace(go.Scatter(x=p_delta_um, y=p_delta_dot_mms, mode='markers', name='Modelo Implementado (Poincaré)', 
                                 marker=dict(color='#0072B2', size=8, symbol='circle-open', line_width=2, opacity=0.9)))
    else:
        print(f"AVISO: Arquivo nativo do Poincaré não encontrado em {poincare_csv}")

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))

    save_plot_formats(fig, output_filename)


# %%
# ==============================================================================
# CONSTRUÇÃO DO MODELO FÍSICO DO ROTOR
# ==============================================================================
def build_multirotor():
    """Constrói o modelo físico do multirotor (Alta e Baixa Velocidade)"""
    helix_angle = np.radians(30.9749)
    pr_ang = np.radians(20)
    damping_factor = 1

    # ------------------ Baixa Velocidade ------------------
    steel_ls = rs.Material(name="AISI_4140", rho=7850 * 0.992887, E=2e11, Poisson=0.3)
    L_aux   = np.array([0.03056235, 0.06356968, 0.0599022 , 0.0403423 , 0.02811736,
           0.04645477, 0.04767726, 0.0207824 , 0.0391198 , 0.03789731,
           0.02444988, 0.02444988, 0.0391198 , 0.03789731, 0.02200489,
           0.04767726, 0.04767726, 0.02689487, 0.03300733, 0.01100244,
           0.09413203, 0.09168704, 0.02322738, 0.00244499, 0.085574572])
    o_d_aux = np.array([0.18072289, 0.08433735, 0.08433735, 0.19036145, 0.19036145,
           0.19036145, 0.19036145, 0.19036145, 0.78072289, 0.78072289,
           0.76144578, 0.76144578, 0.78072289, 0.78072289, 0.19036145,
           0.19036145, 0.19036145, 0.19277108, 0.19518072, 0.21686747,
           0.19036145, 0.19277108, 0.38795181, 0.29879518, 0.195180723])

    L, o_d = L_aux[::-1], o_d_aux[::-1]
    i_d = np.zeros_like(L)
    shaft_gear_ls = [rs.ShaftElement(L=L[i], idl=i_d[i], odl=o_d[i], material=steel_ls, shear_effects=True, rotary_inertia=True, gyroscopic=True) for i in range(len(L))]

    steel = rs.Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)
    gear_low_speed = [rs.GearElementTVMS(n=14, material=steel, width=156.01e-3, bore_diameter=189.9e-3, module=(788.44e-3/169), n_teeth=169, pr_angle=pr_ang, helix_angle=helix_angle)]

    speed_low_blind = np.array([410, 820, 1231, 1641, 2051, 2461, 2871, 3282])*np.pi/30
    kxx_low_blind = np.array([1.26e+08, 1.41e+08, 1.69e+08, 1.82e+08, 1.99e+08, 2.16e+08, 2.30e+08, 2.41e+08])
    kxy_low_blind = np.array([4.33e+08, 3.27e+08, 3.00e+08, 2.85e+08, 2.81e+08, 2.82e+08, 2.85e+08, 2.86e+08])
    kyx_low_blind = np.array([-4.98e+08, -5.77e+08, -6.53e+08, -6.86e+08, -7.25e+08, -7.62e+08, -7.95e+08, -8.22e+08])
    kyy_low_blind = np.array([1.90e+09, 1.40e+09, 1.20e+09, 1.10e+09, 1.05e+09, 1.01e+09, 9.87e+08, 9.78e+08])
    cxx_low_blind = np.array([3.04e+06, 1.93e+06, 1.59e+06, 1.37e+06, 1.25e+06, 1.18e+06, 1.11e+06, 1.04e+06])*damping_factor
    cxy_low_blind = np.array([4.22e+06, 9.47e+05, 2.20e+05, -9.41e+04, -2.24e+05, -2.99e+05, -3.35e+05, -3.51e+05])*damping_factor
    cyx_low_blind = np.array([8.52e+05, -4.68e+05, -6.70e+05, -6.04e+05, -5.79e+05, -5.56e+05, -5.25e+05, -4.77e+05])*damping_factor
    cyy_low_blind = np.array([5.13e+07, 2.30e+07, 1.48e+07, 1.11e+07, 8.88e+06, 7.49e+06, 6.49e+06, 5.75e+06])*damping_factor

    speed_low_ext = np.array([410, 820, 1231, 1641, 2051, 2461, 2871, 3282])*np.pi/30
    kxx_low_ext = np.array([1.70e+08, 1.93e+08, 2.03e+08, 2.24e+08, 2.43e+08, 2.62e+08, 2.81e+08, 2.96e+08])
    kxy_low_ext = np.array([4.37e+08, 3.32e+08, 2.93e+08, 2.81e+08, 2.76e+08, 2.80e+08, 2.84e+08, 2.88e+08])
    kyx_low_ext = np.array([-7.33e+08, -7.69e+08, -7.75e+08, -8.14e+08, -8.47e+08, -8.82e+08, -9.16e+08, -9.44e+08])
    kyy_low_ext = np.array([2.38e+09, 1.72e+09, 1.48e+09, 1.34e+09, 1.26e+09, 1.20e+09, 1.17e+09, 1.14e+09])
    cxx_low_ext = np.array([3.27e+06, 2.20e+06, 1.72e+06, 1.52e+06, 1.37e+06, 1.27e+06, 1.21e+06, 1.15e+06])*damping_factor
    cxy_low_ext = np.array([2.71e+06, 1.18e+06, -4.18e+05, -5.37e+05, -5.75e+05, -5.83e+05, -5.92e+05, -5.77e+05])*damping_factor
    cyx_low_ext = np.array([-1.53e+06, -1.71e+06, -1.24e+06, -1.09e+06, -9.43e+05, -8.49e+05, -7.92e+05, -7.23e+05])*damping_factor
    cyy_low_ext = np.array([6.18e+07, 2.73e+07, 1.74e+07, 1.28e+07, 1.02e+07, 8.48e+06, 7.33e+06, 6.45e+06])*damping_factor

    n = [9, 19]
    kxx, kyy, kxy, kyx = [kxx_low_ext, kxx_low_blind], [kyy_low_ext, kyy_low_blind], [kxy_low_ext, kxy_low_blind], [kyx_low_ext, kyx_low_blind]
    cxx, cyy, cxy, cyx = [cxx_low_ext, cxx_low_blind], [cyy_low_ext, cyy_low_blind], [cxy_low_ext, cxy_low_blind], [cyx_low_ext, cyx_low_blind]
    freq = [speed_low_ext, speed_low_blind]

    bearing_ls_gearbox = [rs.BearingElement(n=n[i], kxx=kxx[i], kyy=kyy[i], kxy=kxy[i], kyx=kyx[i], cxx=cxx[i], cyy=cyy[i], cxy=cxy[i], cyx=cyx[i], frequency=freq[i]) for i in range(2)]
    rotor_ls = rs.Rotor(shaft_gear_ls, gear_low_speed, bearing_ls_gearbox)

    # ------------------ Alta Velocidade ------------------
    steel_hs = rs.Material(name="AISI_4140", rho=7850 * 1.016, E=2e11, Poisson=0.3)
    L_aux   = np.array([0.03870968, 0.02822581, 0.02580645, 0.03387097, 0.03629032, 0.03870968, 0.03870968, 0.01935484, 0.01048387, 0.03951613, 0.03870968, 0.02419355, 0.0233871 , 0.03870968, 0.03870968, 0.01048387, 0.02903226, 0.03870968, 0.04112903, 0.025     , 0.01451613, 0.00967742, 0.03306452, 0.03145161, 0.01532258, 0.03064516, 0.04112903, 0.01209677, 0.00887097])
    o_d_aux = np.array([0.0899, 0.07377049, 0.07377049, 0.08032787, 0.08032787, 0.08032787, 0.08032787, 0.08032787, 0.10655738, 0.12459016, 0.12459016, 0.10819672, 0.10655738, 0.12459016, 0.12459016, 0.10655738, 0.08032787, 0.07868852, 0.08032787, 0.07868852, 0.07868852, 0.10655738, 0.07868852, 0.07868852, 0.05737705, 0.05737705, 0.05737705, 0.08032787, 0.08032787])

    L, o_d = L_aux[::-1], o_d_aux[::-1]
    i_d = np.zeros_like(L)
    shaft_gear_hs = [rs.ShaftElement(L=L[i], idl=i_d[i], odl=o_d[i], material=steel_hs, shear_effects=True, rotary_inertia=True, gyroscopic=True) for i in range(len(L))]

    gear_high_speed = [rs.GearElementTVMS(n=17, material=steel, width=156.01e-3, bore_diameter=60e-3, module=(125.96e-3/27), n_teeth=27, pr_angle=pr_ang, helix_angle=helix_angle)]

    speed_hs_blind = np.array([2567, 5134, 7702, 10269, 12836, 15403, 17970, 20538])*np.pi/30
    kxx_hs_blind = np.array([8.16e+08, 6.55e+08, 6.24e+08, 6.22e+08, 6.29e+08, 6.40e+08, 6.54e+08, 6.68e+08])
    kyx_hs_blind = np.array([-1.10e+09, -8.37e+08, -7.70e+08, -7.52e+08, -7.53e+08, -7.64e+08, -7.79e+08, -7.96e+08])
    kxy_hs_blind = np.array([-2.75e+08, -5.44e+07, 4.11e+07, 8.72e+07, 1.15e+08, 1.27e+08, 1.32e+08, 1.35e+08])
    kyy_hs_blind = np.array([7.92e+08, 4.72e+08, 3.70e+08, 3.27e+08, 3.05e+08, 2.92e+08, 2.83e+08, 2.77e+08])
    cxx_hs_blind = np.array([3.52e+06, 1.65e+06, 1.12e+06, 8.53e+05, 6.94e+05, 5.77e+05, 4.92e+05, 4.29e+05])*damping_factor
    cxy_hs_blind = np.array([-3.37e+06, -1.41e+06, -9.05e+05, -6.65e+05, -5.27e+05, -4.35e+05, -3.70e+05, -3.22e+05])*damping_factor
    cyx_hs_blind = np.array([-3.95e+06, -1.61e+06, -9.80e+05, -6.94e+05, -5.35e+05, -4.36e+05, -3.69e+05, -3.21e+05])*damping_factor
    cyy_hs_blind = np.array([4.77e+06, 1.98e+06, 1.26e+06, 9.35e+05, 7.54e+05, 6.39e+05, 5.58e+05, 4.99e+05])*damping_factor

    speed_hs_ext = np.array([2567, 5134, 7702, 10269, 12836, 15403, 17970, 20538])*np.pi/30
    kxx_hs_ext = np.array([7.97e+08, 6.49e+08, 6.17e+08, 6.14e+08, 6.21e+08, 6.33e+08, 6.46e+08, 6.61e+08])
    kxy_hs_ext = np.array([-2.72e+08, -3.35e+07, 4.72e+07, 8.99e+07, 1.13e+08, 1.23e+08, 1.26e+08, 1.29e+08])
    kyx_hs_ext = np.array([-1.06e+09, -8.12e+08, -7.49e+08, -7.33e+08, -7.35e+08, -7.46e+08, -7.62e+08, -7.81e+08])
    kyy_hs_ext = np.array([7.57e+08, 4.43e+08, 3.53e+08, 3.14e+08, 2.94e+08, 2.82e+08, 2.74e+08, 2.69e+08])
    cxx_hs_ext = np.array([3.39e+06, 1.67e+06, 1.11e+06, 8.42e+05, 6.76e+05, 5.62e+05, 4.78e+05, 4.18e+05])*damping_factor
    cxy_hs_ext = np.array([-3.20e+06, -1.41e+06, -8.83e+05, -6.49e+05, -5.11e+05, -4.22e+05, -3.59e+05, -3.13e+05])*damping_factor
    cyx_hs_ext = np.array([-3.78e+06, -1.58e+06, -9.49e+05, -6.71e+05, -5.18e+05, -4.22e+05, -3.58e+05, -3.12e+05])*damping_factor
    cyy_hs_ext = np.array([4.53e+06, 1.92e+06, 1.21e+06, 9.05e+05, 7.31e+05, 6.21e+05, 5.43e+05, 4.86e+05])*damping_factor

    n = [11, 23]
    kxx, kyy, kxy, kyx = [kxx_hs_ext, kxx_hs_blind], [kyy_hs_ext, kyy_hs_blind], [kxy_hs_ext, kxy_hs_blind], [kyx_hs_ext, kyx_hs_blind]
    cxx, cyy, cxy, cyx = [cxx_hs_ext, cxx_hs_blind], [cyy_hs_ext, cyy_hs_blind], [cxy_hs_ext, cxy_hs_blind], [cyx_hs_ext, cyx_hs_blind]
    freq = [speed_hs_ext, speed_hs_blind]

    bearing_hs_gearbox = [rs.BearingElement(n=n[i], kxx=kxx[i], kyy=kyy[i], kxy=kxy[i], kyx=kyx[i], cxx=cxx[i], cyy=cyy[i], cxy=cxy[i], cyx=cyx[i], frequency=freq[i]) for i in range(2)]
    rotor_hs = rs.Rotor(shaft_gear_hs, gear_high_speed, bearing_hs_gearbox)

    multirotor = rs.MultiRotor(rotor_hs, rotor_ls, coupled_nodes=(17, 14), update_mesh_stiffness=True, position="above", orientation_angle=0)
    
    return multirotor


# %%
# ==============================================================================
# FUNÇÃO DE EXECUÇÃO DA SIMULAÇÃO
# ==============================================================================
def run_simulation_at_speed(speed_rpm, sim_time_seconds, integrador="internal_newmark"):
    print(f"\n[{speed_rpm} RPM] Construindo o modelo e calculando número de ciclos...")
    
    z1 = 27
    pd_gear = 457.20e-3                 
    alpha_0_rad = np.radians(20.0)
    b0 = ((0.305+0.559)/2)*1e-3                               
    err_amp = 0                   

    speed_rad_s = speed_rpm * np.pi / 30
    omega_m = speed_rad_s           
    Tm = 2 * np.pi / omega_m      

    n_cicles = int(np.ceil(sim_time_seconds / Tm))
    cut_cicles = 0 
    
    multirotor = build_multirotor()

    unb_node = [int(e.n) for e in multirotor.disk_elements if isinstance(e, rs.GearElement)]
    unb_magnitude = [81.48e-6, 11491.25e-6]
    unb_phase = [0.0, 0.0]

    backlash = Backlash(
        multirotor, speed_rad_s, b0=b0, error_amp=err_amp, gear_mesh_stiffness=None,
        num_points_cicle=60000, n_cicles=n_cicles, cut_cicles=cut_cicles,
        use_multirotor_coupling_stiffness=False, compute_contact_ratio=True, mesh_damping_ratio=0.07
    )

    _, _, _ = backlash._get_or_create_stiffness_table(square_varying_stiffness=False, n_poits=500)

    F = np.zeros((len(backlash.time), multirotor.ndof))

    print(f"[{speed_rpm} RPM] Iniciando Integração Não-Linear ({n_cicles} ciclos até ~{sim_time_seconds}s)...")

    gamma = 0.5
    beta = (1/4) * (gamma + 0.5)**2
    
    backlash.run_dynamic_backlash(
        unb_node=unb_node, unb_magnitude=unb_magnitude, unb_phase=unb_phase,
        integration_method=integrador, gamma=gamma, beta=beta, tol=1e-6,
        sigma=1e5, smooth_operator=False, add_force=F
    )
    
    pasta_base_nativa = os.path.join(diretorio_execucao, f"resultados_engrenamento_{speed_rpm}rpm")
    pasta_saida = backlash.save_results(unb_node, unb_magnitude, unb_phase, integrador, output_dir=pasta_base_nativa)
    
    # Dashboards e Mapas de Poincaré nativos
    caminho_dash = os.path.join(pasta_saida, "painel_grafico.html")
    backlash.plot_dashboard(freq_unit="rpm", decimation=5, save_path=caminho_dash)

    wm = speed_rad_s * z1
    Tm1 = (2.0 * np.pi) / wm  
    t_max = backlash.time[-1]
    n_periods = int(t_max / Tm1)
    discard_periods = int(n_periods * 0.975)
    backlash.plot_poincare_map(is_linear=False, save_dir=pasta_saida, discard_periods=discard_periods)
    
    # Captura o índice do grau de liberdade x1 (do primeiro nó de engrenagem)
    idx_x1 = unb_node[0] * multirotor.number_dof + 0
    cr0 = multirotor.mesh.contact_ratio

    return backlash, idx_x1, pd_gear, b0, alpha_0_rad, cr0, pasta_saida

# ==============================================================================
# FUNÇÃO AUXILIAR PARA GERAR TODOS OS GRÁFICOS
# ==============================================================================
def gerar_graficos_resultados(bk, idx_x1, pd_gear, b0, alfa0_rad, cr0, pasta_saida, speed_rpm, tempo_sim_total):
    """
    Gera todos os gráficos extraindo os dados da classe Backlash.
    Um zoom na janela de tempo (últimos 0.2s) é aplicado nos gráficos temporais.
    """
    print(f"\n[{speed_rpm} RPM] Gerando gráficos individuais...")

    t = bk.time
    x1 = bk.time_response.yout[:, idx_x1]
    delta = bk.backlash_results['delta']
    alfa_deg = np.degrees(bk.backlash_results['alfa'])
    d = bk.backlash_results['d']
    bt = bk.backlash_results['bt']
    contact_ratio = bk.backlash_results['contact_ratio']
    Fm = bk.backlash_results['Fm']

    # Constantes de Referência
    d0_ref_mm = pd_gear * 1000.0                 
    alfa0_ref_deg = np.degrees(alfa0_rad)        
    b0_ref_um = b0 * 1e6                         
    cr0_ref = cr0                                

    # Define uma janela de tempo para visualização (zoom no final da simulação para ver o regime)
    t_min = max(0, tempo_sim_total - 0.2)
    t_window = [t_min, tempo_sim_total]

    # --- 1. Respostas no Tempo com Linhas Constantes ---
    plot_time_response(t, d, f"Distância entre centros (d) a {speed_rpm} RPM", "Tempo t /s", "Distância entre centros d /mm", f"Res_{speed_rpm}RPM_CenterDistance.html", t_window, 1000.0, constant_ref=d0_ref_mm)
    plot_time_response(t, alfa_deg, f"Ângulo de pressão (α) a {speed_rpm} RPM", "Tempo t /s", "Ângulo de pressão α /(°)", f"Res_{speed_rpm}RPM_PressureAngle.html", t_window, 1.0, constant_ref=alfa0_ref_deg)
    plot_time_response(t, bt, f"Folga (bt) a {speed_rpm} RPM", "Tempo t /s", "Folga bt /μm", f"Res_{speed_rpm}RPM_Backlash.html", t_window, 1e6, constant_ref=b0_ref_um)
    plot_time_response(t, contact_ratio, f"Razão de contato a {speed_rpm} RPM", "Tempo t /s", "Razão de contato mp", f"Res_{speed_rpm}RPM_ContactRatio.html", t_window, 1.0, constant_ref=cr0_ref)

    # --- 2. Vibração (Deslocamentos) e Força Dinâmica ---
    plot_time_response(t, x1, f"Resposta de vibração x1 a {speed_rpm} RPM", "Tempo t /s", "Deslocamento x1 /μm", f"Res_{speed_rpm}RPM_Vibration_x1.html", t_window, 1e6)
    plot_time_response(t, delta, f"Resposta de vibração δ (DTE) a {speed_rpm} RPM", "Tempo t /s", "DTE δ /μm", f"Res_{speed_rpm}RPM_Vibration_DTE.html", t_window, 1e6)
    plot_time_response(t, Fm, f"Resposta no tempo da DMF a {speed_rpm} RPM", "Tempo t /s", "DMF /kN", f"Res_{speed_rpm}RPM_DMF.html", t_window, 1e-3)

    # --- 3. FFTs (Espectros de Frequência) ---
    plot_fft(t, x1, f"Espectro FFT de x1 a {speed_rpm} RPM", "Frequência f /kHz", "Amplitude x1 /μm", f"Res_{speed_rpm}RPM_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 8])
    plot_fft(t, delta, f"Espectro FFT de δ a {speed_rpm} RPM", "Frequência f /kHz", "Amplitude δ /μm", f"Res_{speed_rpm}RPM_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 8])

    # --- 4. Mapa de Poincaré ---
    plot_poincare(pasta_saida, f"Mapa de Poincaré de δ a {speed_rpm} RPM", "δ /μm", "dδ/dt /(mm/s)", f"Res_{speed_rpm}RPM_Poincare.html")


# %%
# ==============================================================================
# EXECUÇÃO DAS SIMULAÇÕES (1953 RPM e 12225 RPM)
# ==============================================================================
if __name__ == "__main__":
    
    # Define o tempo de simulação (3 segundos para ambas)
    TEMPO_SIMULACAO = 3.0

    # ---------------------------------------------------------
    # 1. Simulação para 1953 RPM
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" INICIANDO SIMULAÇÃO 1: 1953 RPM")
    print("="*60)
    bk_1953, idx_x1_1953, pd_gear_1953, b0_1953, alfa0_rad_1953, cr0_1953, pasta_1953 = run_simulation_at_speed(speed_rpm=1953, sim_time_seconds=TEMPO_SIMULACAO)
    
    # Gera os gráficos da primeira simulação
    gerar_graficos_resultados(bk_1953, idx_x1_1953, pd_gear_1953, b0_1953, alfa0_rad_1953, cr0_1953, pasta_1953, speed_rpm=1953, tempo_sim_total=TEMPO_SIMULACAO)


    # ---------------------------------------------------------
    # 2. Simulação para 12225 RPM
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" INICIANDO SIMULAÇÃO 2: 12225 RPM")
    print("="*60)
    bk_12225, idx_x1_12225, pd_gear_12225, b0_12225, alfa0_rad_12225, cr0_12225, pasta_12225 = run_simulation_at_speed(speed_rpm=12225, sim_time_seconds=TEMPO_SIMULACAO)
    
    # Gera os gráficos da segunda simulação
    gerar_graficos_resultados(bk_12225, idx_x1_12225, pd_gear_12225, b0_12225, alfa0_rad_12225, cr0_12225, pasta_12225, speed_rpm=12225, tempo_sim_total=TEMPO_SIMULACAO)

    print("\n" + "="*60)
    print(" ✅ TODAS AS SIMULAÇÕES FINALIZADAS COM SUCESSO!")
    print(f" Todos os gráficos e dados estão em:\n -> {OUTPUT_PLOT_DIR}")
    print("="*60)