import sys
import os  # <-- ADICIONADO PARA LIDAR COM OS DIRETÓRIOS
# sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\ross_dev_backlash\ross')
# sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\teste_backlash_gemini')
sys.path.append('C:\\Users\\M\\Documents\\Mestrado\\ENGRENAMENTO\\ross')
sys.path.append('C:\\Users\\M\\Documents\\Mestrado\\ENGRENAMENTO\\teste\\teste-gemini')
import numpy as np
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ross as rs
from tqdm import tqdm
from backlash import Backlash # Certifique-se de que o arquivo backlash.py está na mesma pasta

def build_multirotor():
    """Constrói o modelo físico do rotor (Geometria não muda com a velocidade)"""
    z1 = z2 = 20
    m_n = 0.01                             
    pd_gear = m_n * z1                   
    alpha_0_rad = np.radians(20.0)
    width = 0.030
    m_gear = 6.57                        
    J_gear = 0.0365                      
    k_brg = 1.0e8                        
    c_brg = 512.64                      
    
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
    return multirotor

def varredura_bifurcacao(rpm_min=1000, rpm_max=8000, num_steps=300):
    multirotor = build_multirotor()
    unb_node = [int(e.n) for e in multirotor.disk_elements if isinstance(e, rs.GearElement)]
    
    # Parâmetros fixos do seu modelo
    b0 = 50e-6                           
    err_amp = 20e-6 
    T10, T1a = 300.0, 100.0
    T20, T2a = 300.0, 100.0
    z1 = 20

    ks = 3.6228e8                        # Rigidez contato simples (N/m)
    kd = 6.5072e8                        # Rigidez contato duplo (N/m)   

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
            num_points_cicle=6000, n_cicles=n_cicles_sim, cut_cicles=cut_cicles_sim,
            use_multirotor_coupling_stiffness=False, compute_contact_ratio=True, mesh_damping_ratio=0.07
        )

        _, _, _ = backlash._get_or_create_stiffness_table(square_varying_stiffness=True, kd=kd, ks=ks, n_poits = 1000)

        # 2. Recalcula as forças (Torques) baseadas no novo vetor de tempo do backlash
        w1 = speed_rad_s
        w2 = multirotor.mesh.gear_ratio * w1
        F = np.zeros((len(backlash.time), multirotor.ndof))
        F[:, unb_node[0] * multirotor.number_dof + 5] = T10 + T1a * np.sin(w1 * backlash.time)
        F[:, unb_node[1] * multirotor.number_dof + 5] = T20 + T2a * np.sin(w2 * backlash.time)

        gamma = 0.5
        beta = (1/4) * (gamma + 0.5)**2

        # 2. EXECUTA A INTEGRAÇÃO PASSANDO OS ESTADOS ANTERIORES
        backlash.run_dynamic_backlash(
            unb_node=unb_node, unb_magnitude=[0.0, 0.0], unb_phase=[0.0, 0.0],
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
    
    ax.set_xlabel(r'$\mathbf{Rotational\ speed\ \mathit{n}_1 / (kr/min)}$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\mathbf{Displacement\ \mathit{x}_1 / \mu m}$', fontsize=14, fontweight='bold')
    
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
        name='Poincaré Points'
    ))
    
    fig_html.update_layout(
        title="Diagrama de Bifurcação - Análise Interativa",
        xaxis_title="Rotational speed n1 / (kr/min)",
        yaxis_title="Displacement x1 / μm",
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

    aa = 1