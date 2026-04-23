# %%
# ==============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES DE DIRETÓRIOS
# ==============================================================================
import sys
# sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\ross_dev_backlash\ross')
# sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\teste_backlash_gemini')

# Configuração dos diretórios locais (ajuste se necessário)
sys.path.append(r'C:\Users\M\Documents\Mestrado\ENGRENAMENTO\ross')
sys.path.append(r'C:\Users\M\Documents\Mestrado\ENGRENAMENTO\teste\teste-gemini')

import os
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ross as rs





# Importa a classe Backlash e a função de FFT nativa da sua implementação
from backlash import Backlash, compute_dfft

# Diretório onde os CSVs extraídos do artigo estão salvos
# CSV_DIR = r"F:\Mestrado\Validation\paper"
CSV_DIR = r"C:\Users\M\Documents\Mestrado\ENGRENAMENTO\teste\teste-gemini\validation\paper"

# Diretório onde os gráficos HTML comparativos da validação serão salvos
try:
    main_file = sys.modules['__main__'].__file__
    diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
except AttributeError:
    # Fallback para ambientes interativos (como Jupyter Notebook)
    diretorio_execucao = os.getcwd()

OUTPUT_PLOT_DIR = os.path.join(diretorio_execucao, "validation_plots")
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

print(f"Diretório de saída dos gráficos comparativos: {OUTPUT_PLOT_DIR}")

# Paleta de Cores Segura para Daltonismo (Colorblind-friendly palette)
COLOR_SIM = '#0072B2'   # Azul forte (Simulação)
COLOR_PAPER = '#D55E00' # Vermelhão/Laranja (Pontos do Artigo)
COLOR_REF = '#000000'   # Preto (Linha de Referência Constante)

# %%
# ==============================================================================
# FUNÇÕES AUXILIARES PARA VALIDAÇÃO E PLOTAGENS COMUNS
# ==============================================================================

def load_paper_data(filename):
    """Lê o CSV extraído do artigo. Assumimos: Coluna 0 = X, Coluna 1 = Y."""
    filepath = os.path.join(CSV_DIR, filename)
    if not os.path.exists(filepath):
        print(f"AVISO: Arquivo do paper não encontrado: {filepath}. Plotará apenas a simulação.")
        return None, None
    try:
        df = pd.read_csv(filepath, header=None)
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except Exception as e:
        print(f"Erro ao ler {filename}: {e}")
        return None, None

def save_plot_formats(fig, base_filename):
    """Salva a figura gerada em HTML e tenta salvar em PDF."""
    html_path = os.path.join(OUTPUT_PLOT_DIR, base_filename)
    pdf_path = os.path.join(OUTPUT_PLOT_DIR, base_filename.replace('.html', '.pdf'))
    
    # Salva o HTML interativo
    fig.write_html(html_path)
    
    # Tenta salvar o PDF estático (Requer 'pip install kaleido')
    try:
        fig.write_image(pdf_path, format='pdf')
    except ValueError as ve:
        print(f"Aviso ao salvar PDF ({base_filename}): {ve}. Certifique-se de instalar 'kaleido'.")
    except Exception:
        pass # Silencia outros erros menores de exportação

def plot_validation(sim_x, sim_y, csv_filename, title, x_label, y_label, output_filename, 
                    x_range=None, sim_scale_y=1.0, paper_scale_x=1.0, paper_scale_y=1.0, 
                    step_sim=1, step_paper=1, constant_ref=None):
    """Plota a resposta no Tempo."""
    fig = go.Figure()

    if x_range:
        mask = (sim_x >= x_range[0]) & (sim_x <= x_range[1])
        x_plot = sim_x[mask][::step_sim]
        y_plot = (sim_y[mask] * sim_scale_y)[::step_sim]
    else:
        x_plot = sim_x[::step_sim]
        y_plot = (sim_y * sim_scale_y)[::step_sim]

    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Modelo Implementado', 
                             line=dict(color=COLOR_SIM, width=2.5)))

    p_x, p_y = load_paper_data(csv_filename)
    if p_x is not None and p_y is not None:
        fig.add_trace(go.Scatter(x=(p_x * paper_scale_x)[::step_paper], y=(p_y * paper_scale_y)[::step_paper], 
                                 mode='markers', name='Yi et al. (2019)', 
                                 marker=dict(color=COLOR_PAPER, size=5, symbol='circle', opacity=0.85)))

    if constant_ref is not None:
        fig.add_trace(go.Scatter(x=[x_plot[0], x_plot[-1]], y=[constant_ref, constant_ref], 
                                 mode='lines', name='Valor de referência (Constante)', 
                                 line=dict(color=COLOR_REF, width=2, dash='dash')))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    if x_range: fig.update_xaxes(range=x_range)

    save_plot_formats(fig, output_filename)

def plot_fft_validation(sim_time, sim_signal, csv_filename, title, x_label, y_label, output_filename, 
                        sim_scale_y=1.0, x_range=None, paper_scale_x=1.0, paper_scale_y=1.0, step_paper=1):
    """Calcula a FFT usando a função NATIVA 'compute_dfft' da classe Backlash."""
    fig = go.Figure()

    # Corta o transiente inicial (primeira metade) e usa a função do backlash.py
    meio = len(sim_time) // 2
    freq_hz, amp = compute_dfft(sim_signal[meio:], sim_time[meio:], freq_unit="Hz", window="hann")
    
    freq_khz = freq_hz / 1000.0  
    amp_scaled = amp * sim_scale_y

    fig.add_trace(go.Scatter(x=freq_khz, y=amp_scaled, mode='lines', name='Modelo Implementado (FFT)', 
                             line=dict(color=COLOR_SIM, width=2)))

    p_x, p_y = load_paper_data(csv_filename)
    if p_x is not None and p_y is not None:
        fig.add_trace(go.Scatter(x=(p_x * paper_scale_x)[::step_paper], y=(p_y * paper_scale_y)[::step_paper], 
                                 mode='markers', name='Yi et al. (2019)', 
                                 marker=dict(color=COLOR_PAPER, size=4, opacity=0.85)))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    if x_range: fig.update_xaxes(range=x_range)

    save_plot_formats(fig, output_filename)

def plot_poincare_validation(pasta_saida_simulacao, csv_filename_paper, title, x_label, y_label, output_filename, 
                             paper_scale_x=1.0, paper_scale_y=1.0):
    """
    Lê o CSV do Mapa de Poincaré exportado NATIVAMENTE pela classe Backlash 
    e o compara com os pontos extraídos do artigo.
    """
    fig = go.Figure()

    # Caminho do arquivo exportado nativamente pela sua classe Backlash
    poincare_csv = os.path.join(pasta_saida_simulacao, "nao_linear_dados_poincare.csv")
    
    if os.path.exists(poincare_csv):
        df_sim = pd.read_csv(poincare_csv)
        # Extrai os dados que já foram amostrados e calculados no Backlash.py
        p_delta_um = df_sim["Delta_um"].values
        p_delta_dot_mms = df_sim["dDelta_dt_mm_s"].values
        
        fig.add_trace(go.Scatter(x=p_delta_um, y=p_delta_dot_mms, mode='markers', name='Modelo Implementado (Poincaré)', 
                                 marker=dict(color=COLOR_SIM, size=8, symbol='circle-open', line_width=2, opacity=0.9)))
    else:
        print(f"AVISO: Arquivo nativo do Poincaré não encontrado em {poincare_csv}")

    # Pontos do Artigo
    p_x, p_y = load_paper_data(csv_filename_paper)
    if p_x is not None and p_y is not None:
        fig.add_trace(go.Scatter(x=p_x * paper_scale_x, y=p_y * paper_scale_y, mode='markers', name='Yi et al. (2019)', 
                                 marker=dict(color=COLOR_PAPER, size=7, symbol='x', opacity=0.9)))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))

    save_plot_formats(fig, output_filename)

# %%
# ==============================================================================
# FUNÇÃO MESTRE PARA CONSTRUIR O MODELO E RODAR A SIMULAÇÃO POR ROTAÇÃO
# ==============================================================================

def run_simulation_at_speed(speed_rpm, sim_time_seconds, integrador="internal_newmark"):
    print(f"\n[{speed_rpm} RPM] Construindo o modelo e calculando n_cicles...")
    
    z1 = z2 = 20
    m_n = 0.01                           
    pd_gear = m_n * z1                   
    alpha_0_rad = np.radians(20.0)
    width = 0.030
    b0 = 50e-6                           
    err_amp = 20e-6                      
    m_gear = 6.57                        
    J_gear = 0.0365                      
    k_brg = 1.0e8                        
    c_brg = 512.64

    ks = 3.6228e8                        # Rigidez contato simples (N/m)
    kd = 6.5072e8                        # Rigidez contato duplo (N/m)                       

    T10, T1a = 300.0, 100.0
    T20, T2a = 300.0, 100.0

    speed_rad_s = speed_rpm * np.pi / 30
    omega_m = speed_rad_s           
    Tm = 2 * np.pi / (omega_m)      

    n_cicles = int(np.ceil(sim_time_seconds / Tm))
    cut_cicles = 0 
    
    steel = rs.Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)
    steel_stiff = rs.Material(name="Steel_Stiff", rho=0.01, E=1e15, Poisson=0.3)
    
    shaft1 = [rs.ShaftElement(L=0.0001, idl=0.0, odl=0.0001, material=steel_stiff, n=0)]
    brg1 = rs.BearingElement(n=0, kxx=k_brg, kyy=k_brg, cxx=c_brg, cyy=c_brg)
    
    gear1 = rs.GearElementTVMS(
        n=0, material=steel, width=width, bore_diameter=np.sqrt(pd_gear**2-(4*m_gear)/(np.pi*width*steel.rho)), 
        module=m_n, n_teeth=z1, pr_angle=alpha_0_rad, helix_angle=0,
        addendum_coeff=1, tip_clearance_coeff=0.25
    )

    gear1.m = m_gear
    gear1.Ip = J_gear
    gear1.Id = 0.0001*J_gear / 2

    rotor1 = rs.Rotor(shaft_elements=shaft1, disk_elements=[gear1], bearing_elements=[brg1])
    rotor2 = copy.deepcopy(rotor1)

    multirotor = rs.MultiRotor(
        driving_rotor=rotor1, driven_rotor=rotor2, coupled_nodes=(0,0),
        update_mesh_stiffness=True, square_varying_stiffness=True,
        square_stiffness_amplitude_ratio=0.275, orientation_angle=0.0, position="above"
    )

    unb_node = [int(e.n) for e in multirotor.disk_elements if isinstance(e, rs.GearElement)]
    unb_magnitude = [0.0, 0.0]
    unb_phase = [0.0, 0.0]

    backlash = Backlash(
        multirotor, speed_rad_s, b0=b0, error_amp=err_amp, gear_mesh_stiffness=None,
        num_points_cicle=6000, n_cicles=n_cicles, cut_cicles=cut_cicles,
        use_multirotor_coupling_stiffness=False, compute_contact_ratio=True, mesh_damping_ratio=0.07
    )

    _, _, _ = backlash._get_or_create_stiffness_table(square_varying_stiffness=True, kd=kd, ks=ks, n_poits = 1000)

    w1 = speed_rad_s
    w2 = multirotor.mesh.gear_ratio * w1
    F = np.zeros((len(backlash.time), multirotor.ndof))
    F[:, unb_node[0] * multirotor.number_dof + 5] = T10 + T1a * np.sin(w1 * backlash.time)
    F[:, unb_node[1] * multirotor.number_dof + 5] = T20 + T2a * np.sin(w2 * backlash.time)

    print(f"[{speed_rpm} RPM] Iniciando Integração Não-Linear ({n_cicles} ciclos até {sim_time_seconds}s)...")

    # integrador = "ode15s"

    gamma = 0.5
    beta = (1/4) * (gamma + 0.5)**2
    
    backlash.run_dynamic_backlash(
        unb_node=unb_node, unb_magnitude=unb_magnitude, unb_phase=unb_phase,
        integration_method=integrador, gamma=gamma, beta=beta, tol=1e-6,
        sigma=1e5, smooth_operator=False, add_force=F
    )
    
    pasta_base_nativa = os.path.join(diretorio_execucao, f"resultados_engrenamento_{speed_rpm}rpm")
    print(f"\nSalvando dados nativos em: {pasta_base_nativa}")
    
    pasta_saida = backlash.save_results(unb_node, unb_magnitude, unb_phase, integrador, output_dir=pasta_base_nativa)
    
    # Executa a geração de Dashboards e CSVs da própria classe Backlash
    caminho_dash = os.path.join(pasta_saida, "painel_grafico.html")
    backlash.plot_dashboard(freq_unit="rpm", decimation=5, save_path=caminho_dash)

    wm = speed_rad_s * z1
    # Período exato de 1 ciclo de engrenamento (segundos)
    Tm1 = (2.0 * np.pi) / wm  
    t_max = backlash.time[-1]
    n_periods = int(t_max / Tm1)
    discard_periods = int(n_periods * 0.975)

    backlash.plot_poincare_map(is_linear=False, save_dir=pasta_saida, discard_periods=discard_periods)
    
    idx_x1 = unb_node[0] * multirotor.number_dof + 0
    
    # Retorna também pasta_saida para a função de plot de Poincaré conseguir achar o CSV
    return backlash, idx_x1, pd_gear, b0, alpha_0_rad, multirotor.mesh.contact_ratio, pasta_saida

# %%
# ==============================================================================
# VALIDAÇÃO FIGURAS 7 e 9: ROTAÇÃO BAIXA (1000 RPM)
# ==============================================================================
bk_1000, idx_x1_1000, pd_gear, b0, alfa0_rad, cr0, pasta_saida_1000 = run_simulation_at_speed(speed_rpm=1000, sim_time_seconds=2.65)

t_1000 = bk_1000.time
x1_1000 = bk_1000.time_response.yout[:, idx_x1_1000]
delta_1000 = bk_1000.backlash_results['delta']
alfa_1000_deg = np.degrees(bk_1000.backlash_results['alfa'])

step_sim = 1
step_paper = 1

d0_ref_mm = pd_gear * 1000.0                 
alfa0_ref_deg = np.degrees(alfa0_rad)        
b0_ref_um = b0 * 1e6                         
cr0_ref = cr0                                

# --- Respostas no Tempo (Fig 7 - Com Linhas de Referência) ---
plot_validation(t_1000, bk_1000.backlash_results['d'], "7a.csv", "Fig 7(a) - Center Distance (d) @ 1000 RPM", "Time t/s", "Center distance d /mm", "Fig_7a_CenterDistance.html", [2.4, 2.6], 1000.0, step_sim=step_sim, step_paper=step_paper, constant_ref=d0_ref_mm)
plot_validation(t_1000, alfa_1000_deg, "7b.csv", "Fig 7(b) - Pressure Angle (α) @ 1000 RPM", "Time t/s", "Pressure angle α /(°)", "Fig_7b_PressureAngle.html", [2.4, 2.6], 1.0, step_sim=step_sim, step_paper=step_paper, constant_ref=alfa0_ref_deg)
plot_validation(t_1000, bk_1000.backlash_results['bt'], "7c.csv", "Fig 7(c) - Backlash (bt) @ 1000 RPM", "Time t/s", "Backlash bt /μm", "Fig_7c_Backlash.html", [2.4, 2.6], 1e6, step_sim=step_sim, step_paper=step_paper, constant_ref=b0_ref_um)
plot_validation(t_1000, bk_1000.backlash_results['contact_ratio'], "7d.csv", "Fig 7(d) - Contact Ratio (mp) @ 1000 RPM", "Time t/s", "Contact ratio mp", "Fig_7d_ContactRatio.html", [2.4, 2.6], 1.0, step_sim=step_sim, step_paper=step_paper, constant_ref=cr0_ref)

# --- Respostas no Tempo (Fig 9) ---
plot_validation(t_1000, x1_1000, "9a.csv", "Fig 9(a) - Vibration Response x1 @ 1000 RPM", "Time t/s", "Displacement x1 /μm", "Fig_9a_Vibration_x1.html", [2.4, 2.52], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_1000, delta_1000, "9c.csv", "Fig 9(c) - Vibration Response δ (DTE) @ 1000 RPM", "Time t/s", "DTE δ /μm", "Fig_9c_Vibration_DTE.html", [2.4, 2.52], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_1000, bk_1000.backlash_results['Fm'], "9f.csv", "Fig 9(f) - Time Response of DMF @ 1000 RPM", "Time t/s", "DMF /kN", "Fig_9f_DMF.html", [2.4, 2.52], 1e-3, step_sim=step_sim, step_paper=step_paper)

# --- FFTs (Fig 9) ---
plot_fft_validation(t_1000, x1_1000, "9b.csv", "Fig 9(b) - FFT Spectrum of x1 @ 1000 RPM", "Frequency f /kHz", "Amplitude x1 /μm", "Fig_9b_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 0.8], step_paper=step_paper)
plot_fft_validation(t_1000, delta_1000, "9d.csv", "Fig 9(d) - FFT Spectrum of δ @ 1000 RPM", "Frequency f /kHz", "Amplitude δ /μm", "Fig_9d_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 6], step_paper=step_paper)

# --- Mapa de Poincaré NATIVO DA CLASSE (Fig 9) ---
plot_poincare_validation(pasta_saida_1000, "9e.csv", "Fig 9(e) - Poincaré Map of δ @ 1000 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_9e_Poincare.html")


# %%
# ==============================================================================
# VALIDAÇÃO FIGURA 10: ROTAÇÃO MÉDIA (3000 RPM)
# ==============================================================================
bk_3000, idx_x1_3000, pd_gear, b0, alfa0_rad, cr0, pasta_saida_3000 = run_simulation_at_speed(speed_rpm=3000, sim_time_seconds=0.95)

t_3000 = bk_3000.time
x1_3000 = bk_3000.time_response.yout[:, idx_x1_3000]
delta_3000 = bk_3000.backlash_results['delta']

# --- Respostas no Tempo ---
plot_validation(t_3000, x1_3000, "10a.csv", "Fig 10(a) - Vibration Response x1 @ 3000 RPM", "Time t/s", "Displacement x1 /μm", "Fig_10a_Vibration_x1.html", [0.8, 0.85], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_3000, delta_3000, "10c.csv", "Fig 10(c) - Vibration Response δ (DTE) @ 3000 RPM", "Time t/s", "DTE δ /μm", "Fig_10c_Vibration_DTE.html", [0.8, 0.85], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_3000, bk_3000.backlash_results['Fm'], "10f.csv", "Fig 10(f) - Time Response of DMF @ 3000 RPM", "Time t/s", "DMF /kN", "Fig_10f_DMF.html", [0.8, 0.85], 1e-3, step_sim=step_sim, step_paper=step_paper)

# --- FFTs ---
plot_fft_validation(t_3000, x1_3000, "10b.csv", "Fig 10(b) - FFT Spectrum of x1 @ 3000 RPM", "Frequency f /kHz", "Amplitude x1 /μm", "Fig_10b_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 4], step_paper=step_paper)
plot_fft_validation(t_3000, delta_3000, "10d.csv", "Fig 10(d) - FFT Spectrum of δ @ 3000 RPM", "Frequency f /kHz", "Amplitude δ /μm", "Fig_10d_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 7], step_paper=step_paper)

# --- Mapa de Poincaré NATIVO DA CLASSE ---
plot_poincare_validation(pasta_saida_3000, "10e.csv", "Fig 10(e) - Poincaré Map of δ @ 3000 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_10e_Poincare.html")


# %%
# ==============================================================================
# VALIDAÇÃO FIGURA 11: ROTAÇÃO ALTA (4500 RPM)
# ==============================================================================
bk_4500, idx_x1_4500, pd_gear, b0, alfa0_rad, cr0, pasta_saida_4500 = run_simulation_at_speed(speed_rpm=4500, sim_time_seconds=0.65)

t_4500 = bk_4500.time
x1_4500 = bk_4500.time_response.yout[:, idx_x1_4500]
delta_4500 = bk_4500.backlash_results['delta']

# --- Respostas no Tempo ---
plot_validation(t_4500, x1_4500, "11a.csv", "Fig 11(a) - Vibration Response x1 @ 4500 RPM", "Time t/s", "Displacement x1 /μm", "Fig_11a_Vibration_x1.html", [0.55, 0.58], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_4500, delta_4500, "11c.csv", "Fig 11(c) - Vibration Response δ (DTE) @ 4500 RPM", "Time t/s", "DTE δ /μm", "Fig_11c_Vibration_DTE.html", [0.55, 0.58], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_4500, bk_4500.backlash_results['Fm'], "11f.csv", "Fig 11(f) - Time Response of DMF @ 4500 RPM", "Time t/s", "DMF /kN", "Fig_11f_DMF.html", [0.55, 0.58], 1e-3, step_sim=step_sim, step_paper=step_paper)

# --- FFTs ---
plot_fft_validation(t_4500, x1_4500, "11b.csv", "Fig 11(b) - FFT Spectrum of x1 @ 4500 RPM", "Frequency f /kHz", "Amplitude x1 /μm", "Fig_11b_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 3], step_paper=step_paper)
plot_fft_validation(t_4500, delta_4500, "11d.csv", "Fig 11(d) - FFT Spectrum of δ @ 4500 RPM", "Frequency f /kHz", "Amplitude δ /μm", "Fig_11d_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 7], step_paper=step_paper)

# --- Mapa de Poincaré NATIVO DA CLASSE ---
plot_poincare_validation(pasta_saida_4500, "11e.csv", "Fig 11(e) - Poincaré Map of δ @ 4500 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_11e_Poincare.html")

# %%
# ==============================================================================
# VALIDAÇÃO FIGURA 12: ROTAÇÃO MUITO ALTA (6000 RPM)
# ==============================================================================
bk_6000, idx_x1_6000, pd_gear_6000, b0_6000, alfa0_rad_6000, cr0_6000, pasta_saida_6000 = run_simulation_at_speed(speed_rpm=6000, sim_time_seconds=0.5)

t_6000 = bk_6000.time
delta_6000 = bk_6000.backlash_results['delta']

# --- Respostas no Tempo ---
# Fig 12(a) - Vibration Response δ (DTE)
plot_validation(t_6000, delta_6000, "12a.csv", "Fig 12(a) - Vibration Response δ (DTE) @ 6000 RPM", "Time t/s", "DTE δ /μm", "Fig_12a_Vibration_DTE.html", [0.4, 0.425], 1e6, step_sim=step_sim, step_paper=step_paper)

# Fig 12(d) - Time Response of DMF
plot_validation(t_6000, bk_6000.backlash_results['Fm'], "12d.csv", "Fig 12(d) - Time Response of DMF @ 6000 RPM", "Time t/s", "DMF /kN", "Fig_12d_DMF.html", [0.4, 0.425], 1e-3, step_sim=step_sim, step_paper=step_paper)

# --- FFTs ---
# Fig 12(b) - FFT Spectrum of δ
plot_fft_validation(t_6000, delta_6000, "12b.csv", "Fig 12(b) - FFT Spectrum of δ @ 6000 RPM", "Frequency f /kHz", "Amplitude δ /μm", "Fig_12b_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 8], step_paper=step_paper)

# --- Mapa de Poincaré NATIVO DA CLASSE ---
# Fig 12(c) - Poincaré Map of δ
plot_poincare_validation(pasta_saida_6000, "12c.csv", "Fig 12(c) - Poincaré Map of δ @ 6000 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_12c_Poincare.html")