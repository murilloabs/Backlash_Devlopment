import sys
import os  # <-- ADICIONADO PARA LIDAR COM OS DIRETÓRIOS
sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\ross_dev_backlash\ross')
sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\teste_backlash_gemini')

# sys.path.append('C:\\Users\\M\\Documents\\Mestrado\\ENGRENAMENTO\\ross')
# sys.path.append('C:\\Users\\M\\Documents\\Mestrado\\ENGRENAMENTO\\teste\\teste-gemini')
import numpy as np
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ross as rs
from tqdm import tqdm
from backlash import Backlash # Certifique-se de que o arquivo backlash.py está na mesma pasta

def build_multirotor():
    """Constrói o modelo físico do multirotor (Alta e Baixa Velocidade)"""
    
    # Parâmetros definidos
    helix_angle = np.radians(30.9749)
    pr_ang = np.radians(20)
    damping_factor = 1

    # ==========================================
    # Modelagem do Rotor de Baixa Velocidade
    # ==========================================
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

    L = L_aux[::-1]
    o_d = o_d_aux[::-1]
    i_d = np.zeros_like(L)
    N = len(L)

    shaft_gear_ls = [
        rs.ShaftElement(
            L=L[i],
            idl=i_d[i],
            odl=o_d[i],
            material=steel_ls,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(N)
    ]

    # Creating Gear low Speed
    steel = rs.Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)
    n      = [14]
    width  = np.array([156.01])*1e-3
    bore_i_d = np.array([189.9])*1e-3
    pitch_diameter = np.array([788.44])*1e-3
    pressure_angle = np.array([pr_ang])
    n_teeth = np.array([169])
    helix = np.array([helix_angle])
    N = len(n)

    gear_low_speed = [
        rs.GearElementTVMS(
            n=n[i],
            material=steel,
            width=width[i],
            bore_diameter=bore_i_d[i],
            module=pitch_diameter[i]/n_teeth[i],
            n_teeth=n_teeth[i],
            pr_angle=pressure_angle[i],
            helix_angle = helix[i],
        )
        for i in range(N)
    ]

    # Creating Bearing Low Speed Rotor
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

    n   = [9, 19]
    tag = ['Gearbox A', 'Gearbox B']
    kxx = [kxx_low_ext, kxx_low_blind]
    kyy = [kyy_low_ext, kyy_low_blind]
    kxy = [kxy_low_ext, kxy_low_blind]
    kyx = [kyx_low_ext, kyx_low_blind]
    cxx = [cxx_low_ext, cxx_low_blind]
    cyy = [cyy_low_ext, cyy_low_blind]
    cxy = [cxy_low_ext, cxy_low_blind]
    cyx = [cyx_low_ext, cyx_low_blind]
    freq = [speed_low_ext, speed_low_blind]
    N = len(n)

    bearing_ls_gearbox = [
        rs.BearingElement(
            n=n[i],
            kxx = kxx[i],
            kyy = kyy[i],
            kxy = kxy[i],
            kyx = kyx[i],
            cxx = cxx[i],
            cyy = cyy[i],
            cxy = cxy[i],
            cyx = cyx[i],
            tag = tag[i],
            frequency = freq[i]
        )
        for i in range(N)
    ]

    rotor_ls = rs.Rotor(shaft_gear_ls, gear_low_speed, bearing_ls_gearbox)

    # ==========================================
    # Modelagem do Rotor de Alta Velocidade
    # ==========================================
    steel_hs = rs.Material(name="AISI_4140", rho=7850 * 1.016, E=2e11, Poisson=0.3)

    L_aux   = np.array([0.03870968, 0.02822581, 0.02580645, 0.03387097, 0.03629032,
           0.03870968, 0.03870968, 0.01935484, 0.01048387, 0.03951613,
           0.03870968, 0.02419355, 0.0233871 , 0.03870968, 0.03870968,
           0.01048387, 0.02903226, 0.03870968, 0.04112903, 0.025     ,
           0.01451613, 0.00967742, 0.03306452, 0.03145161, 0.01532258,
           0.03064516, 0.04112903, 0.01209677, 0.00887097])

    o_d_aux = np.array([0.0899, 0.07377049, 0.07377049, 0.08032787, 0.08032787,
           0.08032787, 0.08032787, 0.08032787, 0.10655738, 0.12459016,
           0.12459016, 0.10819672, 0.10655738, 0.12459016, 0.12459016,
           0.10655738, 0.08032787, 0.07868852, 0.08032787, 0.07868852,
           0.07868852, 0.10655738, 0.07868852, 0.07868852, 0.05737705,
           0.05737705, 0.05737705, 0.08032787, 0.08032787])

    L = L_aux[::-1]
    o_d = o_d_aux[::-1]
    i_d = np.zeros_like(L)
    N = len(L)
    
    shaft_gear_hs = [
        rs.ShaftElement(
            L=L[i],
            idl=i_d[i],
            odl=o_d[i],
            material=steel_hs,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(N)
    ]

    # Creating Gear High Speed
    n      = [17]
    width  = np.array([156.01])*1e-3
    bore_i_d_hs = np.array([60])*1e-3
    pitch_diameter = np.array([125.96])*1e-3
    pressure_angle = np.array([pr_ang])
    n_teeth = np.array([27])
    helix = np.array([helix_angle])
    N = len(n)

    gear_high_speed = [
        rs.GearElementTVMS(
            n=n[i],
            material=steel,
            width=width[i],
            bore_diameter=bore_i_d_hs[i],
            module=pitch_diameter[i]/n_teeth[i],
            n_teeth=n_teeth[i],
            pr_angle=pressure_angle[i],
            helix_angle = helix[i],
        )
        for i in range(N)
    ]

    # Creating Bearing high Speed Rotor
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

    n   = [11, 23]
    kxx = [kxx_hs_ext, kxx_hs_blind]
    kyy = [kyy_hs_ext, kyy_hs_blind]
    kxy = [kxy_hs_ext, kxy_hs_blind]
    kyx = [kyx_hs_ext, kyx_hs_blind]
    cxx = [cxx_hs_ext, cxx_hs_blind]
    cyy = [cyy_hs_ext, cyy_hs_blind]
    cxy = [cxy_hs_ext, cxy_hs_blind]
    cyx = [cyx_hs_ext, cyx_hs_blind]
    freq = [speed_hs_ext, speed_hs_blind]
    N = len(n)

    bearing_hs_gearbox = [
        rs.BearingElement(
            n=n[i],
            kxx = kxx[i],
            kyy = kyy[i],
            kxy = kxy[i],
            kyx = kyx[i],
            cxx = cxx[i],
            cyy = cyy[i],
            cxy = cxy[i],
            cyx = cyx[i],
            frequency = freq[i]
        )
        for i in range(N)
    ]

    rotor_hs = rs.Rotor(shaft_gear_hs, gear_high_speed, bearing_hs_gearbox)

    # ==========================================
    # Determinação do Multi Rotor
    # ==========================================
    n_gear_ls = next(e.n for e in rotor_ls.disk_elements if isinstance(e, rs.GearElement))
    n_gear_hs = next(e.n for e in rotor_hs.disk_elements if isinstance(e, rs.GearElement))
    coupled_nodes = (n_gear_hs, n_gear_ls)

    multirotor = rs.MultiRotor(
        rotor_hs, 
        rotor_ls, 
        coupled_nodes=coupled_nodes, 
        update_mesh_stiffness=True,
        position="above",
        orientation_angle=0 * np.pi/180,
    )
    
    return multirotor

def varredura_bifurcacao(rpm_min=1, rpm_max=13000, num_steps=500):
    multirotor = build_multirotor()
    unb_node = [int(e.n) for e in multirotor.disk_elements if isinstance(e, rs.GearElement)]
    
    # Parâmetros fixos do seu modelo
    b0 = ((0.305+0.559)/2)*1e-3                           
    err_amp = 0 

    z1 = 27


    # Configuração de convergência da bifurcação
    # 150 ciclos totais, cortamos os primeiros 100. Sobram 50 ciclos limpos de regime permanente.
    n_cicles_sim = 30 #300 
    cut_cicles_sim = 15 #250 

    bif_speed = []
    bif_disp = []

    speeds_rpm = np.linspace(rpm_min, rpm_max, num_steps)

    # 1. INICIALIZA OS ESTADOS A FRIO PARA A PRIMEIRA RPM
    estado_y, estado_ydot, estado_y2dot = None, None, None
    
    for speed_rpm in tqdm(speeds_rpm, desc="Gerando Diagrama"):
        speed_rad_s = speed_rpm * np.pi / 30
        
        # 1. Instancia o Backlash para a velocidade atual
        # Reduzimos num_points_cicle para 2000 para a varredura ficar rápida
        backlash = Backlash(
            multirotor, speed_rad_s, b0=b0, error_amp=err_amp, gear_mesh_stiffness=None,
            num_points_cicle=60000, n_cicles=n_cicles_sim, cut_cicles=cut_cicles_sim,
            use_multirotor_coupling_stiffness=False, compute_contact_ratio=True, mesh_damping_ratio=0.07
        )

        _, _, _ = backlash._get_or_create_stiffness_table(square_varying_stiffness=False, n_poits = 500)

        # 2. Recalcula as forças (Torques) baseadas no novo vetor de tempo do backlash
        w1 = speed_rad_s
        w2 = multirotor.mesh.gear_ratio * w1
        F = np.zeros((len(backlash.time), multirotor.ndof))
        # F[:, unb_node[0] * multirotor.number_dof + 5] = T10 + T1a * np.sin(w1 * backlash.time)
        # F[:, unb_node[1] * multirotor.number_dof + 5] = T20 + T2a * np.sin(w2 * backlash.time)

        # forca peso
        # m_gear = 6.57  
        # F[:, unb_node[0] * multirotor.number_dof + 2] = np.ones_like(backlash.time)*(-m_gear * 9.81)
        # F[:, unb_node[1] * multirotor.number_dof + 2] = np.ones_like(backlash.time)*(-m_gear * 9.81)

        gamma = 0.5
        beta = (1/4) * (gamma + 0.5)**2

        unb_magnitude = [81.48e-6, 11491.25e-6]

        # 2. EXECUTA A INTEGRAÇÃO PASSANDO OS ESTADOS ANTERIORES
        backlash.run_dynamic_backlash(
            unb_node=unb_node, unb_magnitude=unb_magnitude, unb_phase=[0.0, 0.0],
            integration_method="internal_newmark", gamma=gamma, beta=beta, tol=1e-6,
            sigma=1e5, smooth_operator=False, add_force=F,
            ramp_fraction=0.0, # <-- Mantenha 0.0 na bifurcação, pois já estamos usando o Sweep
            y_init=estado_y, ydot_init=estado_ydot, y2dot_init=estado_y2dot # <-- INJEÇÃO
        )
        
        # 3. CAPTURA OS ESTADOS FINAIS DESTA RPM PARA USAR NA PRÓXIMA!
        # estado_y, estado_ydot, estado_y2dot = backlash.final_states

        # 4. Amostragem de Poincaré
        # Frequência de engrenamento (Mesh Frequency)
        wm = speed_rad_s * z1 
        Tm = 2 * np.pi / wm
        
        # O vetor de tempo do response já vem com os transientes cortados (devido ao cut_cicles)
        t_final = backlash.time
        
        # Cria vetor de instantes espaçados exatamente por Tm
        t_poincare = np.arange(t_final[0], t_final[-1], Tm)
        
        # Extrai o deslocamento x1 (Grau de liberdade 0 do nó unb_node[0])
        idx_x1 = unb_node[0] * multirotor.number_dof + 0
        x1_signal = backlash.time_response.yout[:, idx_x1]
        
        # Interpola o sinal contínuo para pegar os pontos exatos
        pts_poincare = np.interp(t_poincare, t_final, x1_signal)
        
        # Armazena os dados
        krpm = speed_rpm / 1000.0
        for pt in pts_poincare:
            bif_speed.append(krpm)
            bif_disp.append(pt * 1e6) # Converte para micrometros para o plot

    return np.array(bif_speed), np.array(bif_disp)

def plotar_diagrama(x_data, y_data, filename_base="diagrama_bifurcacao"):
    """
    Gera o gráfico estilo artigo em PDF e uma versão interativa em HTML.
    """
    print(f"\nSalvando gráficos... Aguarde.")

    # --- INÍCIO DA ADIÇÃO: Captura o diretório onde o arquivo atual está ---
    try:
        diretorio_base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        diretorio_base = os.getcwd()
    # --- FIM DA ADIÇÃO ---
    
    # =========================================================================
    # 1. VERSÃO PDF (MATPLOTLIB) - Foco em Qualidade para Publicação
    # =========================================================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(x_data, y_data, s=0.5, c='magenta', alpha=0.6, edgecolors='none')
    
    ax.set_xlim(0, 50)
    
    ymin, ymax = np.min(y_data), np.max(y_data)
    margem = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margem, ymax + margem)
    
    # --- GRÁFICOS EM PORTUGUÊS (MATPLOTLIB) ---
    ax.set_xlabel(r'$\mathbf{Velocidade\ de\ rotação\ \mathit{n}_1 / (kr/min)}$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{Deslocamento\ \mathit{x}_1 / \mu m}$', fontsize=14, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(direction='in', width=1.5, length=6, labelsize=12, top=True, right=True)

    plt.tight_layout()
    
    # Salva em PDF (Vetorizado, sem perda de qualidade)
    # --- ALTERAÇÃO AQUI: Salva no diretorio_base ---
    pdf_path = os.path.join(diretorio_base, f"{filename_base}.pdf")
    plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f" -> PDF salvo com sucesso: '{pdf_path}'")
    
    # Opcional: Salvar em PNG também para visualização rápida
    # plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    
    plt.close() # Fecha a figura para liberar memória

    # =========================================================================
    # 2. VERSÃO HTML (PLOTLY) - Foco em Análise Interativa (Zoom, Hover)
    # =========================================================================
    # Usamos Scattergl (WebGL) porque diagramas de bifurcação têm MUITOS pontos 
    # e o Scatter normal faria o seu navegador travar.
    fig_html = go.Figure()
    
    fig_html.add_trace(go.Scattergl(
        x=x_data, 
        y=y_data, 
        mode='markers',
        marker=dict(size=3, color='magenta', opacity=0.5),
        name='Pontos de Poincaré' # <-- Traduzido
    ))
    
    # --- GRÁFICOS EM PORTUGUÊS (PLOTLY) ---
    fig_html.update_layout(
        title="Diagrama de Bifurcação - Análise Interativa",
        xaxis_title="Velocidade de rotação n1 / (kr/min)",
        yaxis_title="Deslocamento x1 / μm",
        template="plotly_white",
        width=1200,
        height=700,
        hovermode="closest"
    )
    
    # Linha pontilhada só pra marcar o início do caos (ajuste o x=5.1 para o seu caso real)
    fig_html.add_vline(x=5.1, line_dash="dash", line_color="gray", annotation_text="Início do Caos")
    
    # --- ALTERAÇÃO AQUI: Salva no diretorio_base ---
    html_path = os.path.join(diretorio_base, f"{filename_base}_interativo.html")
    fig_html.write_html(html_path, include_plotlyjs="cdn") # cdn deixa o arquivo menor
    print(f" -> HTML interativo salvo com sucesso: '{html_path}'\n")

if __name__ == "__main__":
    print("\n Iniciando simulações... Prepare um café ☕")
    
    # Ajuste num_steps para a "resolução" desejada. 
    # Sugestão: comece com 50 para testar se funciona rápido. Depois aumente para 300 para o gráfico final de artigo.
    velocidades, deslocamentos = varredura_bifurcacao(rpm_min=1, rpm_max=50000, num_steps=1000)
    
    print("Finalizado! Gerando gráfico...")
    plotar_diagrama(velocidades, deslocamentos)

    miau=1