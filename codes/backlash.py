import sys
# sys.path.append('C:\\Users\\Murillo\\OneDrive - Universidade Federal de Uberlândia\\Área de Trabalho\\Mestrado\\ENGRENAMENTO\\Implementacao\\ross')
sys.path.append('C:\\Users\\Murillo\\OneDrive - Universidade Federal de Uberlândia\\Área de Trabalho\\Mestrado\\ENGRENAMENTO\\Implementacao\\ross_dev_backlash\\ross')

#home
# sys.path.append('C:\\Users\\M\\Documents\\Mestrado\\ENGRENAMENTO\\ross')

import shutil  # <-- BIBLIOTECA PARA LER O TAMANHO DO TERMINAL
import os
import csv
import datetime
import numpy as np
from copy import deepcopy as copy
import time
import ross as rs
from ross.results import TimeResponseResults
from numba import njit, objmode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid




# Tenta importar o tqdm para a barra de progresso no terminal. 
# Se não estiver instalado, cria uma função "falsa" para não quebrar o código.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

__all__ = ["Backlash", "compute_dfft"]

# =============================================================================
# BLOCO: FERRAMENTAS DE PÓS-PROCESSAMENTO (SINAIS E DADOS)
# =============================================================================

def compute_dfft(x, t, freq_unit="Hz", window="hann"):
    """
    Calcula a FFT nativamente com NumPy (já é ultra-otimizado em C nativo do NumPy).
    Não usamos Numba aqui pois Numba não suporta np.fft e essa função roda 
    apenas no final para plotagem, não afetando o tempo da simulação.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dt = t[1] - t[0]
    N = len(x)
    
    # 1. Remove a componente DC (Valor médio)
    x_centered = x - np.mean(x)
        
    # 2. Janela de Hann
    correction = 1.0
    if window == "hann":
        w = np.hanning(N)
        x_centered = x_centered * w
        correction = 1.0 / np.mean(w) # Compensa a perda de energia da janela

    # 3. FFT (Usando rfft pois o sinal é real: é 2x mais rápido e já corta a metade espelhada)
    X_full = np.fft.rfft(x_centered)
    
    # Monta o vetor de frequências
    freq = np.fft.rfftfreq(N, d=dt)
    amplitude = (2.0 / N) * np.abs(X_full) * correction
    
    # Aplica a conversão de unidades solicitada
    if freq_unit == "rad/s": 
        freq = 2.0 * np.pi * freq
    elif freq_unit == "rpm": 
        freq = freq * 60.0

    return freq, amplitude

# =============================================================================
# BLOCO 1: MOTOR MATEMÁTICO E UTILITÁRIOS NUMBA
# Funções puras em C/C++ (Não podem ficar dentro da classe)
# =============================================================================

import sys
import time
import shutil

def print_ode15s_progress(current_t, t_start, t_end, start_time, method_name, dt):
    """
    Exibe o progresso baseado no tempo de simulação contínuo (adaptado para solvers implícitos).
    """
    t_range = t_end - t_start
    progress_t = current_t - t_start
    
    if t_range <= 0: return
    
    percent = (progress_t / t_range) * 100.0
    if percent > 100.0: percent = 100.0
    
    elapsed = time.time() - start_time
    if elapsed <= 0.0: elapsed = 1e-6
    
    # Cálculo de velocidade: Segundos de Simulação por Segundo Real
    if elapsed > 0.0 and progress_t > 0:
        speed_sim = progress_t / elapsed
        if speed_sim > 0:
            eta = (t_range - progress_t) / speed_sim
        else:
            eta = 0.0
    else:
        speed_sim = 0.0
        eta = 0.0

    bar_length = 20
    filled_len = int(bar_length * percent / 100.0)
    bar = '█' * filled_len + '-' * (bar_length - filled_len)

    # Formatação de Tempo e ETA
    e_h = int(elapsed // 3600)
    e_m = int((elapsed % 3600) // 60)
    e_s = int(elapsed % 60)
    str_elapsed = "%dh%02dm%02ds" % (e_h, e_m, e_s) if e_h > 0 else "%02dm%02ds" % (e_m, e_s)
    
    eta_h = int(eta // 3600)
    eta_m = int((eta % 3600) // 60)
    eta_s = int(eta % 60)
    str_eta = "%dh%02dm%02ds" % (eta_h, eta_m, eta_s) if eta_h > 0 else "%02dm%02ds" % (eta_m, eta_s)

    str_dt = "%.2e" % dt
    str_percent = "%.1f" % percent
    str_speed = "%.4f" % speed_sim
    
    # Montagem da String (Substituindo 'passos' pelo tempo da simulação 't')
    raw_msg = f"{method_name} |{bar}| {str_percent}% | t: {current_t:.4f}/{t_end:.4f}s | dt: {str_dt} | Tempo: {str_elapsed} | ETA: {str_eta} | Vel: {str_speed} sim_s/s"
    
    term_width = shutil.get_terminal_size((80, 20)).columns
    if len(raw_msg) > term_width - 1:
        raw_msg = raw_msg[:term_width - 4] + "..."
        
    final_msg = "\r\033[2K" + raw_msg
    sys.stdout.write(final_msg)
    sys.stdout.flush()

@njit
def print_integration_progress(step, total_steps, start_time, print_interval, method_name, dt):
    """Exibe o progresso atualizando a mesma linha no terminal com relógio formatado e dt."""
    
    # ⚠️ AVISO: Removi o "or True" que estava aqui! 
    # Deixar "or True" faz o código abrir o portal do objmode e printar a cada 
    # ÚNICO passo da simulação (100.000 vezes). Isso destrói a performance 
    # e faz o terminal piscar loucamente. Deixe apenas o módulo do print_interval!
    if (step > 0 and step % print_interval == 0):
        with objmode():
            
            elapsed = time.time() - start_time
            percent = (step / (total_steps - 1)) * 100.0

            if elapsed <= 0.0:
                elapsed = 1e-6  
            
            # --- CORREÇÃO DEFINITIVA AQUI ---
            # Só fazemos a divisão se o step for maior que zero!
            if elapsed > 0.0 and step > 0:
                speed_it = step / elapsed
                sec_per_it = elapsed / step
                eta = ((total_steps - 1) - step) * sec_per_it
            else:
                speed_it = 0.0
                sec_per_it = 0.0
                eta = 0.0
            # --------------------------------
                
            bar_length = 20 
            filled_len = int(bar_length * percent / 100.0)
            bar = '█' * filled_len + '-' * (bar_length - filled_len)
            
            e_h = int(elapsed // 3600)
            e_m = int((elapsed % 3600) // 60)
            e_s = int(elapsed % 60)
            if e_h > 0:
                str_elapsed = "%dh%02dm%02ds" % (e_h, e_m, e_s)
            else:
                str_elapsed = "%02dm%02ds" % (e_m, e_s)
                
            eta_h = int(eta // 3600)
            eta_m = int((eta % 3600) // 60)
            eta_s = int(eta % 60)
            if eta_h > 0:
                str_eta = "%dh%02dm%02ds" % (eta_h, eta_m, eta_s)
            else:
                str_eta = "%02dm%02ds" % (eta_m, eta_s)

            str_dt = "%.2e" % dt
            str_percent = "%.1f" % percent
            str_speed = "%.1f" % speed_it
            str_sec_per_it = "%.5f" % sec_per_it
            
            # 1. Montamos a string PURA, sem os códigos de terminal ainda
            raw_msg = str(method_name) + " |" + bar + "| " + str_percent + "% | Passos: " + str(step) + "/" + str(total_steps-1) + " | dt: " + str_dt + " | Tempo: " + str_elapsed + " | ETA: " + str_eta + " | Vel: " + str_speed + " it/s (" + str_sec_per_it + " s/it)"
            
            # 2. Descobrimos a largura atual do terminal (se falhar, assume 80 colunas)
            term_width = shutil.get_terminal_size((80, 20)).columns
            
            # 3. Se a mensagem for maior que a tela, nós a truncamos para não quebrar a linha!
            # Subtraímos 1 para deixar uma margem de segurança no final da tela
            if len(raw_msg) > term_width - 1:
                raw_msg = raw_msg[:term_width - 4] + "..."
            
            # 4. Agora sim, colamos o \r e o \033[2K (limpar linha)
            final_msg = "\r\033[2K" + raw_msg
            
            sys.stdout.write(final_msg)
            sys.stdout.flush()


@njit
def inv(angle):
    """Função involuta do círculo base: inv(a) = tan(a) - a"""
    return np.tan(angle) - angle

@njit
def bilinear_interp(theta, cr, theta_arr, cr_arr, K_table):
    """Interpolação bilinear ultrarrápida 2D para encontrar a Rigidez Instantânea (k_m)."""
    
    # A SACADA: Em vez de fazer módulo com 2*pi, fazemos com o passo do dente!
    theta = theta % theta_arr[-1] 
    
    i = np.searchsorted(theta_arr, theta) - 1
    j = np.searchsorted(cr_arr, cr) - 1
    
    # Prevenção de "Index Out of Bounds"
    if i < 0: i = 0
    if i >= len(theta_arr) - 1: i = len(theta_arr) - 2
    if j < 0: j = 0
    if j >= len(cr_arr) - 1: j = len(cr_arr) - 2
    
    t1, t2 = theta_arr[i], theta_arr[i+1]
    c1, c2 = cr_arr[j], cr_arr[j+1]
    
    wt = (theta - t1) / (t2 - t1) if t2 != t1 else 0.0
    wc = (cr - c1) / (c2 - c1) if c2 != c1 else 0.0
    
    k00, k10 = K_table[i, j], K_table[i+1, j]
    k01, k11 = K_table[i, j+1], K_table[i+1, j+1]
    
    k0 = k00 * (1 - wt) + k10 * wt
    k1 = k01 * (1 - wt) + k11 * wt
    
    return k0 * (1 - wc) + k1 * wc

# =============================================================================
# BLOCO 2: FÍSICA DO ENGRENAMENTO (YI ET AL., 2019)
# =============================================================================

# import numpy as np
# from numba import njit

# @njit
# def calculate_dynamic_backlash_force(
#     disp_resp, velc_resp, gear_nodes, number_of_dof, ndof_total,
#     d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
#     error_step, angular_pos, compute_cr_flag, nominal_cr,
#     Ra1, Ra2, module, sigma, smooth_operator, 
#     theta_arr, cr_arr, K_table,
#     M_eq, damping_ratio, error_dot_step
# ):
#     idx1, idx2 = number_of_dof * gear_nodes[0], number_of_dof * gear_nodes[1]
    
#     # =========================================================================
#     # EXTRAÇÃO DE GRAUS DE LIBERDADE (O artigo usa modelo estritamente 2D)
#     # Variáveis fora do plano (z, rx, ry) serão ignoradas na formulação.
#     # =========================================================================
#     x1, y1 = disp_resp[idx1], disp_resp[idx1+1]
#     t1 = disp_resp[idx1+5] # theta_1
    
#     x2, y2 = disp_resp[idx2], disp_resp[idx2+1]
#     t2 = disp_resp[idx2+5] # theta_2 (Anti-horário positivo)

#     vx1, vy1 = velc_resp[idx1], velc_resp[idx1+1]
#     vt1 = velc_resp[idx1+5] # dot_theta_1
    
#     vx2, vy2 = velc_resp[idx2], velc_resp[idx2+1]
#     vt2 = velc_resp[idx2+5] # dot_theta_2

#     # =========================================================================
#     # 1. CINEMÁTICA BÁSICA (Eqs. 3, 4 e 5 do paper)
#     # =========================================================================
#     # Distância relativa
#     dx = x2 - x1 + d0
#     dy = y2 - y1
    
#     d_inst2 = dx**2 + dy**2
#     d_inst = np.sqrt(d_inst2)
#     if d_inst < 1e-12: d_inst = 1e-12 
    
#     beta = np.arctan2(dy, dx)
    
#     cos_alfa_val = (R1 + R2) / d_inst
#     if cos_alfa_val > 1.0: cos_alfa_val = 1.0
#     elif cos_alfa_val < -1.0: cos_alfa_val = -1.0
#     alfa = np.arccos(cos_alfa_val)

#     # =========================================================================
#     # 2. ERRO DE TRANSMISSÃO DINÂMICO - DTE
#     # Atualizado para convenção matemática universal (+ R1*t1 + R2*t2)
#     # =========================================================================
#     delta = (x1 - x2) * np.sin(alfa - beta) + (y1 - y2) * np.cos(alfa - beta) + R1 * t1 + R2 * t2 - error_step

#     # =========================================================================
#     # 3. FOLGA DINÂMICA TOTAL (Eq. 12 do paper)
#     # =========================================================================
#     inv_alfa = np.tan(alfa) - alfa
#     inv_alfa0 = np.tan(alfa0) - alfa0
#     bt = b0 + (R1 + R2) * (inv_alfa - inv_alfa0)

#     # =========================================================================
#     # 4. DERIVADAS ANALÍTICAS DE ALFA E BETA (Eqs. 31, 32 e 33 do paper)
#     # =========================================================================
#     term_in_sqrt = d_inst2 - (R1 + R2)**2
#     if term_in_sqrt < 1e-12: term_in_sqrt = 1e-12
#     term_sqrt = np.sqrt(term_in_sqrt)

#     alfa_x1 = -((R1 + R2) * dx) / (d_inst2 * term_sqrt)
#     alfa_y1 = -((R1 + R2) * dy) / (d_inst2 * term_sqrt)
#     alfa_x2 = -alfa_x1
#     alfa_y2 = -alfa_y1

#     beta_x1 = dy / d_inst2
#     beta_y1 = -dx / d_inst2
#     beta_x2 = -beta_x1
#     beta_y2 = -beta_y1

#     # =========================================================================
#     # 5. DERIVADAS PARCIAIS DO DTE (Eqs. 22 a 26 do paper)
#     # =========================================================================
#     sin_diff = np.sin(alfa - beta)
#     cos_diff = np.cos(alfa - beta)
#     geom_term = (x1 - x2) * cos_diff - (y1 - y2) * sin_diff

#     delta_x1 = sin_diff + geom_term * (alfa_x1 - beta_x1)
#     delta_y1 = cos_diff + geom_term * (alfa_y1 - beta_y1)
    
#     delta_x2 = -sin_diff + geom_term * (alfa_x2 - beta_x2)
#     delta_y2 = -cos_diff + geom_term * (alfa_y2 - beta_y2)
    
#     delta_t1 = R1
#     delta_t2 = R2  # <-- Alterado de -R2 para +R2

#     # =========================================================================
#     # 6. DERIVADAS PARCIAIS DA FOLGA (Eqs. 27 e 28 do paper)
#     # =========================================================================
#     tan2_alfa = np.tan(alfa)**2
#     bt_x1 = (R1 + R2) * tan2_alfa * alfa_x1
#     bt_y1 = (R1 + R2) * tan2_alfa * alfa_y1
#     bt_x2 = (R1 + R2) * tan2_alfa * alfa_x2
#     bt_y2 = (R1 + R2) * tan2_alfa * alfa_y2

#     # =========================================================================
#     # 7. VELOCIDADES EFETIVAS (Eqs. 20 e 21 do paper)
#     # =========================================================================
#     delta_dot = (vx1 * delta_x1 + vy1 * delta_y1 + vt1 * delta_t1 +
#                  vx2 * delta_x2 + vy2 * delta_y2 + vt2 * delta_t2 - error_dot_step)

#     bt_dot = (R1 + R2) * tan2_alfa * (vx1 * alfa_x1 + vy1 * alfa_y1 + vx2 * alfa_x2 + vy2 * alfa_y2)

#     # =========================================================================
#     # 8. FUNÇÕES NÃO-LINEARES E DERIVADAS DIRECIONAIS (Eqs. 9, 10 e 27-30 do paper)
#     # =========================================================================
#     if delta > bt: 
#         f_val = delta - bt
#         f1_val = delta_dot - bt_dot
#         f_x1 = delta_x1 - bt_x1
#         f_y1 = delta_y1 - bt_y1
#         f_x2 = delta_x2 - bt_x2
#         f_y2 = delta_y2 - bt_y2
#         f_t1 = delta_t1
#         f_t2 = delta_t2
#     elif delta < -bt: 
#         f_val = delta + bt
#         f1_val = delta_dot + bt_dot
#         f_x1 = delta_x1 + bt_x1
#         f_y1 = delta_y1 + bt_y1
#         f_x2 = delta_x2 + bt_x2
#         f_y2 = delta_y2 + bt_y2
#         f_t1 = delta_t1
#         f_t2 = delta_t2
#     else:
#         f_val = 0.0
#         f1_val = 0.0
#         f_x1 = f_y1 = f_x2 = f_y2 = f_t1 = f_t2 = 0.0

#     # =========================================================================
#     # 9. CÁLCULO DAS RIGIDEZES, AMORTECIMENTO E FORÇA MESHING (Eqs. 6 e 8)
#     # =========================================================================
#     contact_ratio = nominal_cr
#     if compute_cr_flag:
#         pb = np.pi * module * np.cos(alfa0) 
#         contact_ratio = (np.sqrt(Ra1**2 - R1**2) + np.sqrt(Ra2**2 - R2**2) - d_inst * np.sin(alfa)) / pb

#     K_time_step = bilinear_interp(angular_pos, contact_ratio, theta_arr, cr_arr, K_table)
#     c_m = 2.0 * damping_ratio * np.sqrt(K_time_step * M_eq)

#     # Força Dinâmica de Engrenamento (Eq. 8)
#     Fm = K_time_step * f_val + c_m * f1_val

#     # =========================================================================
#     # 10. COMPOSIÇÃO DO VETOR DE FORÇAS GENERALIZADAS
#     # Graus de liberdade fora do plano (z, rx, ry) recebem contribuição nula.
#     # =========================================================================
#     backlash_force = np.zeros(ndof_total)

#     backlash_force[idx1]   = -Fm * f_x1 
#     backlash_force[idx1+1] = -Fm * f_y1 
#     backlash_force[idx1+2] = 0.0 # Eixo Z 
#     backlash_force[idx1+3] = 0.0 # RX
#     backlash_force[idx1+4] = 0.0 # RY
#     backlash_force[idx1+5] = -Fm * f_t1 
    
#     backlash_force[idx2]   = -Fm * f_x2 
#     backlash_force[idx2+1] = -Fm * f_y2 
#     backlash_force[idx2+2] = 0.0 # Eixo Z
#     backlash_force[idx2+3] = 0.0 # RX
#     backlash_force[idx2+4] = 0.0 # RY 
#     backlash_force[idx2+5] = -Fm * f_t2  # Isso aplicará o torque (-Fm * R2), opondo-se ao movimento

#     # Log de monitoramento
#     logs = np.array([
#         x1, y1, x2, y2, t1, t2, 
#         d_inst, beta, alfa, contact_ratio, delta, bt, f_val, K_time_step, Fm
#     ], dtype=np.float64)

#     return backlash_force, logs


@njit
def calculate_dynamic_backlash_force(
    disp_resp, velc_resp, gear_nodes, number_of_dof, ndof_total,
    d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
    error_step, angular_pos, compute_cr_flag, nominal_cr,
    Ra1, Ra2, module, sigma, smooth_operator, # <-- Parâmetro Booleano Adicionado
    theta_arr, cr_arr, K_table,
    M_eq, damping_ratio, error_dot_step
):
    idx1, idx2 = number_of_dof * gear_nodes[0], number_of_dof * gear_nodes[1]
    
    # Extração COMPLETA dos 6 Graus de Liberdade (GL)
    x1, y1, z1 = disp_resp[idx1], disp_resp[idx1+1], disp_resp[idx1+2]
    rx1, ry1, t1 = disp_resp[idx1+3], disp_resp[idx1+4], disp_resp[idx1+5]
    
    x2, y2, z2 = disp_resp[idx2], disp_resp[idx2+1], disp_resp[idx2+2]
    rx2, ry2, t2 = disp_resp[idx2+3], disp_resp[idx2+4], disp_resp[idx2+5]

    # Extração das Velocidades
    vx1, vy1, vz1 = velc_resp[idx1], velc_resp[idx1+1], velc_resp[idx1+2]
    vrx1, vry1, vt1 = velc_resp[idx1+3], velc_resp[idx1+4], velc_resp[idx1+5]
    vx2, vy2, vz2 = velc_resp[idx2], velc_resp[idx2+1], velc_resp[idx2+2]
    vrx2, vry2, vt2 = velc_resp[idx2+3], velc_resp[idx2+4], velc_resp[idx2+5]

    # Cálculo da cinemática não-linear no plano transversal
    cos_ori, sin_ori = np.cos(orientation_angle), np.sin(orientation_angle)
    x2_abs, y2_abs = x2 + d0 * cos_ori, y2 + d0 * sin_ori
    dx, dy = x2_abs - x1, y2_abs - y1
    d_inst = np.sqrt(dx**2 + dy**2)
    if d_inst < 1e-12: d_inst = 1e-12 
    beta = np.arctan2(dy, dx)
    
    cos_alfa_val = (R1 + R2) / d_inst
    if cos_alfa_val > 1.0: cos_alfa_val = 1.0
    elif cos_alfa_val < -1.0: cos_alfa_val = -1.0
    alfa = np.arccos(cos_alfa_val)

    psi = alfa - beta
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    sin_beta_h, cos_beta_h = np.sin(helix_angle), np.cos(helix_angle)

    # --- DERIVADAS GEOMÉTRICAS BÁSICAS ---
    d_inst2 = d_inst**2
    term_in_sqrt = d_inst2 - (R1 + R2)**2
    if term_in_sqrt < 1e-12: term_in_sqrt = 1e-12
    term_sqrt = np.sqrt(term_in_sqrt)

    alfa_x1 = -((R1 + R2) * dx) / (d_inst2 * term_sqrt)
    alfa_y1 = -((R1 + R2) * dy) / (d_inst2 * term_sqrt)
    beta_x1 = dy / d_inst2
    beta_y1 = -dx / d_inst2

    alfa_x2, alfa_y2 = -alfa_x1, -alfa_y1
    beta_x2, beta_y2 = -beta_x1, -beta_y1

    # --- EQUAÇÃO DO DTE 3D Completo ---
    delta = (
        ((x1 - x2) * sin_psi + (y1 - y2) * cos_psi + R1 * t1 + R2 * t2) * cos_beta_h +
        ((-z1 + z2) + (R1 * rx1 + R2 * rx2) * sin_psi + (R1 * ry1 + R2 * ry2) * cos_psi) * sin_beta_h
        - error_step
    )

    # delta_dot = (
    #    ((vx1 - vx2) * sin_psi + (vy1 - vy2) * cos_psi + R1 * vt1 + R2 * vt2) * cos_beta_h +
    #    ((-vz1 + vz2) + (R1 * vrx1 + R2 * vrx2) * sin_psi + (R1 * vry1 + R2 * vry2) * cos_psi) * sin_beta_h
    #    - error_dot_step
    # )

    

    # --- CÁLCULO DA FOLGA DINÂMICA (bt) ---
    inv_alfa = np.tan(alfa) - alfa
    inv_alfa0 = np.tan(alfa0) - alfa0
    delta_b = (R1 + R2) * (inv_alfa - inv_alfa0)
    bt = b0 + delta_b * cos_beta_h

    # --- DERIVADAS ANALÍTICAS DO DELTA PELOS DOFS ---
    geom_trans = (x1 - x2) * cos_psi - (y1 - y2) * sin_psi
    geom_rot   = (R1 * rx1 + R2 * rx2) * cos_psi - (R1 * ry1 + R2 * ry2) * sin_psi
    
    psi_x1 = alfa_x1 - beta_x1
    psi_y1 = alfa_y1 - beta_y1
    psi_x2 = alfa_x2 - beta_x2
    psi_y2 = alfa_y2 - beta_y2

    d_delta_dx1 = (sin_psi + geom_trans * psi_x1) * cos_beta_h + (geom_rot * psi_x1) * sin_beta_h
    d_delta_dy1 = (cos_psi + geom_trans * psi_y1) * cos_beta_h + (geom_rot * psi_y1) * sin_beta_h
    d_delta_dx2 = (-sin_psi + geom_trans * psi_x2) * cos_beta_h + (geom_rot * psi_x2) * sin_beta_h
    d_delta_dy2 = (-cos_psi + geom_trans * psi_y2) * cos_beta_h + (geom_rot * psi_y2) * sin_beta_h

    d_delta_dz1, d_delta_dz2 = -sin_beta_h, sin_beta_h

    d_delta_drx1 = R1 * sin_psi * sin_beta_h
    d_delta_dry1 = R1 * cos_psi * sin_beta_h
    d_delta_drx2 = R2 * sin_psi * sin_beta_h
    d_delta_dry2 = R2 * cos_psi * sin_beta_h

    d_delta_dt1, d_delta_dt2 = R1 * cos_beta_h, R2 * cos_beta_h

    # --- DERIVADAS DA FOLGA ---
    tan2_alfa = np.tan(alfa)**2
    bt_x1 = (R1 + R2) * tan2_alfa * alfa_x1 * cos_beta_h
    bt_y1 = (R1 + R2) * tan2_alfa * alfa_y1 * cos_beta_h
    bt_x2 = (R1 + R2) * tan2_alfa * alfa_x2 * cos_beta_h
    bt_y2 = (R1 + R2) * tan2_alfa * alfa_y2 * cos_beta_h


    delta_dot = (
        d_delta_dx1 * vx1 + d_delta_dy1 * vy1 + d_delta_dz1 * vz1 +
        d_delta_drx1 * vrx1 + d_delta_dry1 * vry1 + d_delta_dt1 * vt1 +
        d_delta_dx2 * vx2 + d_delta_dy2 * vy2 + d_delta_dz2 * vz2 +
        d_delta_drx2 * vrx2 + d_delta_dry2 * vry2 + d_delta_dt2 * vt2
        - error_dot_step
    )

    # =========================================================================
    # CORREÇÃO: CÁLCULO DA DERIVADA TEMPORAL DA FOLGA (bt_dot) - Eq. (21)
    # =========================================================================
    # Apenas as translações afetam alfa, logo alfa_dot depende de vx e vy
    alfa_dot = alfa_x1 * vx1 + alfa_y1 * vy1 + alfa_x2 * vx2 + alfa_y2 * vy2
    bt_dot = (R1 + R2) * tan2_alfa * alfa_dot * cos_beta_h

    # =========================================================================
    # LÓGICA DE CÁLCULO DAS PENALIDADES (f_val, f1_val e f_qi)
    # =========================================================================
    
    if smooth_operator:
        # 1. Abordagem de Suavização Global (Eq. 6 de Walha et al.)
        x1_val = delta - bt
        x2_val = delta + bt
        
        tanh_x1 = np.tanh(sigma * x1_val)
        tanh_x2 = np.tanh(sigma * x2_val)
        
        # Funções g1 e g2 (Eq. 7 do artigo)
        g1 = x1_val * tanh_x1
        g2 = x2_val * tanh_x2
        
        # Cálculo da penalidade elástica global
        f_val = delta + 0.5 * (g1 - g2)
        
        # Derivadas analíticas de g1 e g2 em relação aos seus argumentos
        gp1 = tanh_x1 + sigma * x1_val * (1.0 - tanh_x1**2)
        gp2 = tanh_x2 + sigma * x2_val * (1.0 - tanh_x2**2)
        
        # Derivadas parciais da força global em relação à delta e a folga bt
        df_ddelta = 1.0 + 0.5 * (gp1 - gp2)
        df_dbt    = 0.5 * (-gp1 - gp2)
        
        # O f1_val atua como a velocidade efetiva de penetração para o amortecimento
        # f1_val = df_ddelta * delta_dot
        f1_val = df_ddelta * delta_dot + df_dbt * bt_dot
        
        # Composição analítica de f_qi = (df/d_delta)*delta_qi + (df/d_bt)*bt_qi
        f_x1 = d_delta_dx1 * df_ddelta + bt_x1 * df_dbt
        f_y1 = d_delta_dy1 * df_ddelta + bt_y1 * df_dbt
        f_z1 = d_delta_dz1 * df_ddelta
        f_rx1 = d_delta_drx1 * df_ddelta
        f_ry1 = d_delta_dry1 * df_ddelta
        f_t1  = d_delta_dt1 * df_ddelta
        
        f_x2 = d_delta_dx2 * df_ddelta + bt_x2 * df_dbt
        f_y2 = d_delta_dy2 * df_ddelta + bt_y2 * df_dbt
        f_z2 = d_delta_dz2 * df_ddelta
        f_rx2 = d_delta_drx2 * df_ddelta
        f_ry2 = d_delta_dry2 * df_ddelta
        f_t2  = d_delta_dt2 * df_ddelta

    else:
        # 2. Abordagem Rígida Original (Condicional Discreto)
        # se bt=0 
        if delta > bt: 
            f_val = delta - bt
            f1_val = delta_dot - bt_dot  # <-- Adicionado -bt_dot (Eq. 10)
            sgn = 1.0
        elif delta < -bt: 
            f_val = delta + bt
            f1_val = delta_dot + bt_dot  # <-- Adicionado +bt_dot (Eq. 10)
            sgn = -1.0
        else:
            f_val = 0.0
            f1_val = 0.0
            sgn = 0.0
            
        # Composição condicional de f_qi
        if sgn != 0.0:
            f_x1 = d_delta_dx1 - sgn * bt_x1
            f_y1 = d_delta_dy1 - sgn * bt_y1
            f_z1 = d_delta_dz1
            f_rx1 = d_delta_drx1
            f_ry1 = d_delta_dry1
            f_t1  = d_delta_dt1
            
            f_x2 = d_delta_dx2 - sgn * bt_x2
            f_y2 = d_delta_dy2 - sgn * bt_y2
            f_z2 = d_delta_dz2
            f_rx2 = d_delta_drx2
            f_ry2 = d_delta_dry2
            f_t2  = d_delta_dt2
        else:
            f_x1 = f_y1 = f_z1 = f_rx1 = f_ry1 = f_t1 = 0.0
            f_x2 = f_y2 = f_z2 = f_rx2 = f_ry2 = f_t2 = 0.0

    # =========================================================================
    # CÁLCULO DAS RIGIDEZES, AMORTECIMENTO E FORÇA TOTAL
    # =========================================================================

    contact_ratio = nominal_cr
    if compute_cr_flag:
        pb = np.pi * module * np.cos(alfa0) 
        contact_ratio = (np.sqrt(Ra1**2 - R1**2) + np.sqrt(Ra2**2 - R2**2) - d_inst * np.sin(alfa)) / pb

    K_time_step = bilinear_interp(angular_pos, contact_ratio, theta_arr, cr_arr, K_table)
    c_m = 2.0 * damping_ratio * np.sqrt(K_time_step * M_eq)

    # Força Normal Total na Linha de Ação 
    Fm = K_time_step * f_val + c_m * f1_val

    # =========================================================================
    # DECOMPOSIÇÃO DAS FORÇAS (Estritamente Q_i = -Fm * f_{,q_i})
    # =========================================================================
    
    backlash_force = np.zeros(ndof_total)

    backlash_force[idx1]   = -Fm * f_x1 
    backlash_force[idx1+1] = -Fm * f_y1 
    backlash_force[idx1+2] = -Fm * f_z1 
    backlash_force[idx1+3] = -Fm * f_rx1 
    backlash_force[idx1+4] = -Fm * f_ry1 
    backlash_force[idx1+5] = -Fm * f_t1 
    
    backlash_force[idx2]   = -Fm * f_x2 
    backlash_force[idx2+1] = -Fm * f_y2 
    backlash_force[idx2+2] = -Fm * f_z2 
    backlash_force[idx2+3] = -Fm * f_rx2 
    backlash_force[idx2+4] = -Fm * f_ry2 
    backlash_force[idx2+5] = -Fm * f_t2 

    # Log de monitoramento
    logs = np.array([
        x1, y1, x2_abs, y2_abs, t1, t2, 
        d_inst, beta, alfa, contact_ratio, delta, bt, f_val, K_time_step, Fm
    ], dtype=np.float64)

    return backlash_force, logs


# =============================================================================
# BLOCO 3: INTEGRADOR RUNGE-KUTTA (RK45) ACELERADO POR NUMBA
# Resolve o sistema no Espaço de Estados de tempo contínuo
# =============================================================================

@njit
def get_rk45_deriv_and_logs(y_i, t_i, A, B, f_unb_i,
                           gear_nodes, number_of_dof, ndof_total,
                           d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
                           error_amp, wm, speed_driving, compute_cr_flag, nominal_cr,
                           Ra1, Ra2, module, sigma, smooth_operator,
                           theta_arr, cr_arr, K_table,
                           M_eq, damping_ratio):
    
    disp_resp, velc_resp = y_i[:ndof_total], y_i[ndof_total:]
    error_step = error_amp * np.sin(wm * t_i)
    error_dot_step = error_amp * wm * np.cos(wm * t_i) 
    angular_pos = speed_driving * t_i

    backlash_force, logs = calculate_dynamic_backlash_force(
        disp_resp, velc_resp, gear_nodes, number_of_dof, ndof_total,
        d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
        error_step, angular_pos, compute_cr_flag, nominal_cr,
        Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
        M_eq, damping_ratio, error_dot_step 
    )

    total_force = f_unb_i + backlash_force
    deriv = A @ y_i + B @ total_force 

    return deriv, backlash_force, logs


@njit
def rk45_solver_full(t_array, dt_initial, yout, logs_matrix, force_matrix,
                    A, B, F_unb, gear_nodes, number_of_dof, ndof_total,
                    d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
                    error_amp, wm, speed_driving, compute_cr_flag, nominal_cr,
                    Ra1, Ra2, module, sigma, smooth_operator,
                    theta_arr, cr_arr, K_table,
                    M_eq, damping_ratio, start_time):
    
    n_steps = len(t_array)
    print_interval = max(1, n_steps // 100)
    
    # Parâmetros de tolerância RK45
    rtol = 1e-3
    atol = 1e-6
    dt = dt_initial
    
    # --- Ponto Inicial (t=0) ---
    y_current = np.copy(yout[0])
    dy0, fb0, logs_0 = get_rk45_deriv_and_logs(
        y_current, t_array[0], A, B, F_unb[0], gear_nodes, number_of_dof, ndof_total,
        d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving,
        compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
        M_eq, damping_ratio
    )
    for col in range(15): logs_matrix[0, col] = logs_0[col]
    for dof in range(ndof_total): force_matrix[0, dof] = fb0[dof]

    # --- LOOP DE INTEGRAÇÃO ---
    for i in range(n_steps - 1):
        t_start = t_array[i]
        t_end = t_array[i+1]
        t_now = t_start
        
        # O RK45 pode precisar de vários sub-passos para chegar de t_array[i] até t_array[i+1]
        while t_now < t_end:
            if t_now + dt > t_end:
                dt = t_end - t_now
            
            # Para simplificar o código, f_unb é interpolado ou pego do início do step
            f_unb_i = F_unb[i] 

            # Estágios Dormand-Prince (RK45)
            k1, _, _ = get_rk45_deriv_and_logs(y_current, t_now, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)
            
            k2, _, _ = get_rk45_deriv_and_logs(y_current + dt*(1/5)*k1, t_now + (1/5)*dt, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)
            
            k3, _, _ = get_rk45_deriv_and_logs(y_current + dt*(3/40*k1 + 9/40*k2), t_now + (3/10)*dt, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)
            
            k4, _, _ = get_rk45_deriv_and_logs(y_current + dt*(44/45*k1 - 56/15*k2 + 32/9*k3), t_now + (4/5)*dt, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)
            
            k5, _, _ = get_rk45_deriv_and_logs(y_current + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4), t_now + (8/9)*dt, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)
            
            k6, _, _ = get_rk45_deriv_and_logs(y_current + dt*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5), t_now + dt, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)

            # Estimativa de 5ª ordem
            y_next = y_current + dt * (35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
            
            # Necessário k7 para estimativa de erro
            k7, _, _ = get_rk45_deriv_and_logs(y_next, t_now + dt, A, B, f_unb_i, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, M_eq, damping_ratio)

            # Cálculo do Erro
            error_vec = dt * (71/57600*k1 - 71/16695*k3 + 71/1920*k4 - 17253/339200*k5 + 22/525*k6 - 1/40*k7)
            scale = atol + rtol * np.maximum(np.abs(y_current), np.abs(y_next))
            err_ratio = np.sqrt(np.mean((error_vec / scale) ** 2))

            if err_ratio <= 1.0:
                t_now += dt
                y_current = y_next
                # Ajusta dt para cima
                dt = dt * min(5.0, 0.9 * (err_ratio + 1e-15)**-0.2)
            else:
                # Rejeita o passo e reduz dt
                dt = dt * max(0.2, 0.9 * err_ratio**-0.25)
        
        # Após atingir t_end (o checkpoint atual do t_array)
        yout[i+1] = y_current
        
        # Log final do step para armazenamento (calculado no t_array[i+1])
        _, fb_next, logs_next = get_rk45_deriv_and_logs(
            yout[i+1], t_array[i+1], A, B, F_unb[i+1], gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving,
            compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
            M_eq, damping_ratio
        )
        for col in range(15): logs_matrix[i+1, col] = logs_next[col]
        for dof in range(ndof_total): force_matrix[i+1, dof] = fb_next[dof]

        print_integration_progress(i, n_steps, start_time, print_interval, "RK45", dt)

# =============================================================================
# BLOCO 4: INTEGRADOR NEWMARK + NEWTON-RAPHSON ACELERADO POR NUMBA
# Permite passo de tempo adaptativo com Jacobiano numérico focado.
# =============================================================================

@njit
def newmark_predict(ny, y0, ydot0, y2dot0, dt, gamma, beta):
    y2dot = np.zeros(ny)
    ydot = ydot0 + y2dot0 * (1.0 - gamma) * dt
    y = y0 + ydot0 * dt + y2dot0 * (0.5 - beta) * (dt**2)
    return y, ydot, y2dot

@njit
def newmark_calc_rotor_res(y, ydot, y2dot, t_eval, F_unb_eval, M, C, K, 
                           gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, 
                           R1, R2, alfa0, helix_angle, b0, compute_cr_flag, nominal_cr,
                           Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
                           M_eq, damping_ratio, 
                           theta_eval, error_eval, error_dot_eval):

    # error_step = error_amp * np.sin(wm * t_eval)
    # error_dot_step = error_amp * wm * np.cos(wm * t_eval)
    # angular_pos = speed_driving * t_eval

    F_backlash, logs = calculate_dynamic_backlash_force(
        y, ydot, gear_nodes, number_of_dof, ndof_total,
        d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
        error_eval, theta_eval, compute_cr_flag, nominal_cr, # <-- PASSANDO OS AVALIADOS
        Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
        M_eq, damping_ratio, error_dot_eval                  # <-- PASSANDO O AVALIADO
    )
    F_total = F_unb_eval + F_backlash
    res = F_total - (M @ y2dot + C @ ydot + K @ y)
    return res, F_backlash, logs

@njit
def newmark_build_jacobian(y, ydot, y2dot, dt, gamma, beta, t_eval, F_unb_eval, res_base, 
                           M, C, K, active_dofs, epsilon, gear_nodes, number_of_dof, ndof_total, 
                           d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
                           compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
                           M_eq, damping_ratio, theta_eval, error_eval, error_dot_eval): # <-- LIMPO
    
    J = M + C * (gamma * dt) + K * (beta * (dt**2))
    
    F_nl_base, _ = calculate_dynamic_backlash_force(
        y, ydot, gear_nodes, number_of_dof, ndof_total,
        d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
        error_eval, theta_eval, compute_cr_flag, nominal_cr, 
        Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
        M_eq, damping_ratio, error_dot_eval                  
    )
    
    for i in active_dofs:
        y_orig, ydot_orig = y[i], ydot[i]

        y[i] = y_orig + epsilon * beta * (dt**2)
        ydot[i] = ydot_orig + epsilon * gamma * dt

        F_nl_pert, _ = calculate_dynamic_backlash_force(
            y, ydot, gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
            error_eval, theta_eval, compute_cr_flag, nominal_cr, 
            Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
            M_eq, damping_ratio, error_dot_eval                  
        )

        dF_nl_daccel = (F_nl_pert - F_nl_base) / epsilon

        for j in active_dofs:
            J[j, i] -= dF_nl_daccel[j]

        y[i], ydot[i] = y_orig, ydot_orig

    return J

# @njit
# def newmark_build_jacobian(y, ydot, y2dot, dt, gamma, beta, t_eval, F_unb_eval, res_base, 
#                            M, C, K, active_dofs, epsilon, gear_nodes, number_of_dof, ndof_total, 
#                            d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, 
#                            compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
#                            M_eq, damping_ratio):
#     J = M + C * (gamma * dt) + K * (beta * (dt**2))
    
#     for i in active_dofs:
#         y2dot_orig, ydot_orig, y_orig = y2dot[i], ydot[i], y[i]

#         y2dot[i] = y2dot_orig + epsilon
#         ydot[i] = ydot_orig + epsilon * gamma * dt
#         y[i] = y_orig + epsilon * beta * (dt**2)

#         res_pert, _, _ = newmark_calc_rotor_res(
#             y, ydot, y2dot, t_eval, F_unb_eval, M, C, K, gear_nodes, number_of_dof, 
#             ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, error_amp, wm, speed_driving, 
#             compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, # <-- Repasse
#             M_eq, damping_ratio 
#         )

#         J[:, i] = (res_base - res_pert) / epsilon
#         y2dot[i], ydot[i], y[i] = y2dot_orig, ydot_orig, y_orig

#     return J

@njit
def newmark_converge_nr(y0, ydot0, y2dot0, dt_sub, gamma, beta, tol, epsilon, t_eval, F_unb_eval, 
                        M, C, K, active_dofs, gear_nodes, number_of_dof, ndof_total, d0, 
                        orientation_angle, R1, R2, alfa0, helix_angle, b0, 
                        compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
                        M_eq, damping_ratio, theta_eval, error_eval, error_dot_eval): # <-- LIMPO
    ny = len(y0)
    y, ydot, y2dot = newmark_predict(ny, y0, ydot0, y2dot0, dt_sub, gamma, beta)
    
    convergiu, need_rebuild = False, True
    norm_res = 0.0
    J = np.zeros((ny, ny))
    F_b_out = np.zeros(ny)
    logs_out = np.zeros(15)
    
    for nr_iter in range(1, 16):
        res_base, F_b, logs = newmark_calc_rotor_res(
            y, ydot, y2dot, t_eval, F_unb_eval, M, C, K, gear_nodes, number_of_dof, 
            ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
            compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, 
            M_eq, damping_ratio, theta_eval, error_eval, error_dot_eval # <-- CHAMADA LIMPA
        )
        norm_res = np.linalg.norm(res_base)

        if norm_res < tol:
            convergiu = True
            F_b_out, logs_out = F_b, logs
            break

        need_rebuild = True

        if need_rebuild or nr_iter % 5 == 0:
            J = newmark_build_jacobian(
                y, ydot, y2dot, dt_sub, gamma, beta, t_eval, F_unb_eval, res_base, 
                M, C, K, active_dofs, epsilon, gear_nodes, number_of_dof, ndof_total, d0, 
                orientation_angle, R1, R2, alfa0, helix_angle, b0, 
                compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, 
                M_eq, damping_ratio, theta_eval, error_eval, error_dot_eval # <-- CHAMADA LIMPA
            )
            need_rebuild = False

        dy2dot = np.linalg.solve(J, res_base)
        y2dot += dy2dot
        ydot += dy2dot * gamma * dt_sub
        y += dy2dot * beta * (dt_sub**2)

    return y, ydot, y2dot, nr_iter, convergiu, norm_res, F_b_out, logs_out

@njit
def newmark_solver_full(t_array, yout, logs_matrix, force_matrix, F_unb,
                        M, C, K, active_dofs, gamma, beta, tol, epsilon,
                        gear_nodes, number_of_dof, ndof_total, d0, 
                        orientation_angle, R1, R2, alfa0, helix_angle, b0, 
                        compute_cr_flag, nominal_cr, Ra1, Ra2, 
                        module, sigma, smooth_operator, theta_arr, cr_arr, K_table,
                        M_eq, damping_ratio, start_time, y_init, ydot_init, y2dot_init,
                        theta_array, error_array, error_dot_array):  # <-- LIMPO
    
    n_steps = len(t_array)
    ny = ndof_total
    dt_macro = t_array[1] - t_array[0]
    dt_min = dt_macro * 1e-5 
    time_tol = dt_macro * 1e-6 

    y0 = np.copy(y_init)
    ydot0 = np.copy(ydot_init)
    y2dot0 = np.copy(y2dot_init)
    
    # --- Ponto Inicial (t=0) --- CORRIGIDO COM OS VETORES NA POSIÇÃO ZERO
    _, fb0, logs_0 = newmark_calc_rotor_res(
            y0, ydot0, y2dot0, t_array[0], F_unb[0], M, C, K, gear_nodes, number_of_dof, 
            ndof_total, d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
            compute_cr_flag, nominal_cr, Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, 
            M_eq, damping_ratio, theta_array[0], error_array[0], error_dot_array[0] 
    )
    yout[0, :] = y0
    for col in range(15): logs_matrix[0, col] = logs_0[col]
    for dof in range(ndof_total): force_matrix[0, dof] = fb0[dof]

    print_interval = max(1, n_steps // 20)
    dt_sub = dt_macro 

    for step in range(1, n_steps):
        t_target, t_current = t_array[step], t_array[step - 1]
        F_unb_prev, F_unb_next = F_unb[step - 1], F_unb[step]

        while (t_target - t_current) > time_tol:
            if (t_current + dt_sub) > (t_target - time_tol):
                dt_current_step = t_target - t_current
                is_last_substep = True
            else:
                dt_current_step = dt_sub
                is_last_substep = False

            t_eval = t_current + dt_current_step
            ratio = (t_eval - t_array[step - 1]) / dt_macro
            F_unb_eval = F_unb_prev + ratio * (F_unb_next - F_unb_prev)

            theta_eval = theta_array[step - 1] + ratio * (theta_array[step] - theta_array[step - 1])
            error_eval = error_array[step - 1] + ratio * (error_array[step] - error_array[step - 1])
            error_dot_eval = error_dot_array[step - 1] + ratio * (error_dot_array[step] - error_dot_array[step - 1])

            y_new, ydot_new, y2dot_new, nr_iter, convergiu, norm_res, F_b_out, logs_out = newmark_converge_nr(
                y0, ydot0, y2dot0, dt_current_step, gamma, beta, tol, epsilon, t_eval, F_unb_eval, 
                M, C, K, active_dofs, gear_nodes, number_of_dof, ndof_total, d0, orientation_angle, 
                R1, R2, alfa0, helix_angle, b0, compute_cr_flag, nominal_cr, 
                Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, 
                M_eq, damping_ratio, theta_eval=theta_eval, error_eval=error_eval, error_dot_eval=error_dot_eval 
            )

            if convergiu:
                t_current += dt_current_step
                y0, ydot0, y2dot0 = y_new, ydot_new, y2dot_new
                
                if nr_iter <= 5: 
                    dt_sub = min(dt_sub * 2.0, dt_macro)
                elif nr_iter >= 10: 
                    dt_sub = max(dt_sub * 0.5, dt_min)
                
                if is_last_substep:
                    yout[step, :] = y0
                    for col in range(15): logs_matrix[step, col] = logs_out[col]
                    for dof in range(ndof_total): force_matrix[step, dof] = F_b_out[dof]
            else:
                dt_sub *= 0.25 
                if dt_sub < dt_min:
                    raise RuntimeError("Newmark divergiu. Limite dt_min atingido.")

            print_integration_progress(step, n_steps, start_time, print_interval, "Newmark Adapt.", dt_sub)

    return ydot0, y2dot0

# =============================================================================
# BLOCO: BDF INTEGRATION
# =============================================================================

@njit
def bdf_rhs_wrapper(t, y_i, t_array, F_unb_matrix, A, B, gear_nodes, number_of_dof, ndof_total,
                    d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
                    error_amp, wm, speed_driving, compute_cr_flag, nominal_cr,
                    Ra1, Ra2, module, sigma, smooth_operator,
                    theta_arr, cr_arr, K_table, M_eq, damping_ratio):
    
    # 1. Interpolação Linear Rápida de F_unb para qualquer instante t que o SciPy escolher
    idx = np.searchsorted(t_array, t) - 1
    if idx < 0: idx = 0
    if idx >= len(t_array) - 1: idx = len(t_array) - 2
    
    t0, t1 = t_array[idx], t_array[idx+1]
    f0, f1 = F_unb_matrix[idx], F_unb_matrix[idx+1]
    
    weight = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    f_unb_i = f0 + weight * (f1 - f0)
    
    # Garante que a memória seja contígua antes de multiplicar as matrizes gigantes
    y_i_contig = np.ascontiguousarray(y_i)
    
    # 2. Chama a sua rotina exata (já em Numba)
    deriv, _, _ = get_rk45_deriv_and_logs(
        y_i_contig, t, A, B, f_unb_i,
        gear_nodes, number_of_dof, ndof_total,
        d0, orientation_angle, R1, R2, alfa0, helix_angle, b0,
        error_amp, wm, speed_driving, compute_cr_flag, nominal_cr,
        Ra1, Ra2, module, sigma, smooth_operator,
        theta_arr, cr_arr, K_table, M_eq, damping_ratio
    )
    
    return deriv


# =============================================================================
# BLOCO 5: EXTRATOR DE BASELINE (LINEAR -> NÃO-LINEAR)
# =============================================================================


@njit
def extract_backlash_logs_from_trajectory(yout, ydot, t_array, gear_nodes, number_of_dof, ndof_total,
                                          d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
                                          compute_cr_flag, nominal_cr,
                                          Ra1, Ra2, module, sigma, smooth_operator, theta_arr, cr_arr, K_table, 
                                          M_eq, damping_ratio, start_time,
                                          theta_array, error_array, error_dot_array): # <-- Recebe os Vetores Cinemáticos
    n_steps = len(t_array)
    logs_matrix = np.zeros((n_steps, 15))
    force_matrix = np.zeros((n_steps, ndof_total))
    
    print_interval = max(1, n_steps // 100) 
    dt_macro = t_array[1] - t_array[0] if n_steps > 1 else 0.0
    
    for i in range(n_steps):
        disp_resp = yout[i, :]
        velc_resp = ydot[i, :]
        
        # --- PUXA DIRETO DOS ARRAYS PRÉ-CALCULADOS ---
        angular_pos = theta_array[i] % (2 * np.pi)
        error_step = error_array[i]
        error_dot_step = error_dot_array[i]
        
        f_b, logs = calculate_dynamic_backlash_force(
            disp_resp, velc_resp, gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, b0, 
            error_step, angular_pos, compute_cr_flag, nominal_cr,
            Ra1, Ra2, module, sigma, smooth_operator, 
            theta_arr, cr_arr, K_table,
            M_eq, damping_ratio, error_dot_step
        )
        
        for col in range(15): logs_matrix[i, col] = logs[col]
        for dof in range(ndof_total): force_matrix[i, dof] = f_b[dof]
            
        print_integration_progress(i, n_steps, start_time, print_interval, "Extraindo Logs e Forças", dt_macro)
            
    return logs_matrix, force_matrix


# =============================================================================
# BLOCO 6: CLASSE PRINCIPAL (INTERFACE PYTHON ORIENTADA A OBJETOS)
# =============================================================================

class Backlash:
    def __init__(self, 
                 multirotor,
                 speed_driving_gear,
                 b0=0.0,
                 error_amp=0.0,
                 gear_mesh_stiffness=None,
                 num_points_cicle=1000, 
                 n_cicles=2, 
                 cut_cicles=1,
                 use_multirotor_coupling_stiffness=False,
                 compute_contact_ratio=True,
                 mesh_damping_ratio=0.07,  # csi (\xi) - Padrão 0.07 segundo Yi et al. (2019)

                 ):            
        
        self.multirotor = copy(multirotor)
        self.speed_driving_gear = speed_driving_gear
        self.b0 = b0
        self.error_amp = error_amp
        self.gear_mesh_stiffness = gear_mesh_stiffness
        self.n_cicles = n_cicles
        self.cut_cicles = cut_cicles
        self.compute_contact_ratio = compute_contact_ratio
        
        # Parâmetros Numéricos e de Amortecimento
        self.mesh_damping_ratio = mesh_damping_ratio 

        self.gears = np.array([e for e in self.multirotor.disk_elements if isinstance(e, rs.GearElement)])
        if len(self.gears) != 2:
            raise ValueError("O multirotor deve conter exatamente duas engrenagens acopladas.")

        # --- CÁLCULO DA MASSA EQUIVALENTE DO ENGRENAMENTO (M_eq) ---
        J1, J2 = self.gears[0].Ip, self.gears[1].Ip
        R1, R2 = self.gears[0].base_radius, self.gears[1].base_radius
        self.M_eq = (J1 * J2) / (J2 * (R1**2) + J1 * (R2**2))

        # Configuração de Vetores de Tempo
        n_cicles_sim = self.n_cicles + self.cut_cicles
        max_time = n_cicles_sim * (2 * np.pi * self.gears[0].n_teeth) / (self.speed_driving_gear * self.gears[0].n_teeth)
        
        self.num_points_total = num_points_cicle * n_cicles_sim
        self.time = np.linspace(0, max_time, self.num_points_total)
        self.n_cut = num_points_cicle * self.cut_cicles

        # Erro de transmissão estático pré-calculado
        wm = self.speed_driving_gear * self.gears[0].n_teeth
        self.error = self.error_amp * np.sin(wm * self.time)

        self.init_backlash_results()

        # Desliga a rigidez de malha fixa nativa do ROSS
        if not use_multirotor_coupling_stiffness:
            self.multirotor.gear_mesh_stiffness = 0
            self.multirotor.update_mesh_stiffness = False

    def init_backlash_results(self):
        """Inicializa o dicionário global que armazena os dados instantâneos."""
        def zeros_arr(): return np.zeros(self.num_points_total)
        self.backlash_total_force = np.zeros((self.num_points_total, self.multirotor.ndof))

        self.backlash_results = {
            "x1": zeros_arr(), "y1": zeros_arr(), "x2": zeros_arr(), "y2": zeros_arr(),
            "t1": zeros_arr(), "t2": zeros_arr(), "d": zeros_arr(), "beta": zeros_arr(),
            "alfa": zeros_arr(), "contact_ratio": zeros_arr(), "delta": zeros_arr(),
            "bt": zeros_arr(), "f": zeros_arr(), "K_time": zeros_arr(), "Fm": zeros_arr()
        }

    def cut_backlash_results(self):
        """Remove o período transiente inicial da simulação."""
        for key in self.backlash_results:
            self.backlash_results[key] = self.backlash_results[key][self.n_cut:]
        if hasattr(self, 'backlash_total_force'):
            self.backlash_total_force = self.backlash_total_force[self.n_cut:, :]
        if hasattr(self, 'unb_force'):
            self.unbalance_force = self.unb_force.T[self.n_cut:, :]
    
    # def generate_speed_ramp(self, ramp_fraction=0.0):
    #     """Gera uma rampa linear de velocidade angular."""
    #     t = np.asarray(self.time)
    #     omega_max = self.speed_driving_gear
    #     if ramp_fraction <= 0.0: return omega_max

    #     T_ramp = ramp_fraction * t[-1]
    #     speed_ramp = np.zeros_like(t)
    #     ramp_mask = t <= T_ramp
    #     speed_ramp[ramp_mask] = omega_max * (t[ramp_mask] / T_ramp)
    #     speed_ramp[~ramp_mask] = omega_max
    #     return speed_ramp

    def generate_speed_ramp(self, ramp_fraction=0.0):
        """Gera uma rampa linear de velocidade angular."""
        t = np.asarray(self.time)
        omega_max = self.speed_driving_gear
        if ramp_fraction <= 0.0: return np.full_like(t, omega_max)  

        T_ramp = ramp_fraction * t[-1]
        speed_ramp = np.zeros_like(t)
        ramp_mask = t <= T_ramp
        speed_ramp[ramp_mask] = omega_max * (t[ramp_mask] / T_ramp)
        speed_ramp[~ramp_mask] = omega_max
        return speed_ramp


    def _get_or_create_stiffness_table(self, force_recalculate=False, square_varying_stiffness= False, kd=0, ks=0, n_poits = 200):
        """Carrega a tabela de rigidez 2D da memória ou gera uma nova usando o ROSS."""
        
        # 1 e 2. Tenta descobrir o arquivo que chamou a execução e o seu diretório
        try:
            # Pega o caminho completo do script principal em execução (.py)
            main_file = sys.modules['__main__'].__file__
            
            # Pega o DIRETÓRIO onde esse script principal está salvo
            diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
            
            # Pega o NOME do arquivo sem a extensão
            nome_arquivo = os.path.splitext(os.path.basename(main_file))[0]
            
        except AttributeError:
            # Em ambientes interativos (como Jupyter Notebook .ipynb), o kernel não 
            # expõe o nome do notebook diretamente.
            # Aqui sim, usamos o getcwd() porque o Jupyter costuma rodar no próprio diretório.
            diretorio_execucao = os.getcwd() 
            nome_arquivo = "notebook_simulacao" 
            
        # 3. Monta o novo nome do arquivo e o caminho completo dinâmico
        filename = f"k_table_{nome_arquivo}.npz"
        caminho_completo = os.path.join(diretorio_execucao, filename)

        if os.path.exists(caminho_completo) and not force_recalculate:
            print(f"Carregando Lookup Table de rigidez: '{caminho_completo}'...")
            with np.load(caminho_completo) as data:
                return data['theta_arr'], data['cr_arr'], data['K_table']
            
        print(f"Gerando nova grade de rigidez 2D em: {caminho_completo}...")
        
        # O período da rigidez se repete a cada dente (Pitch Angle)
        pitch_angle = 2 * np.pi / self.gears[0].n_teeth
        
        # Mapeamos APENAS 1 dente com n_poits pontos. É o mesmo que usar milhares de pontos!
        theta_arr = np.linspace(0.0, pitch_angle, n_poits) 
        cr_arr = np.linspace(0.8, 2.5, n_poits) 
        K_table = np.zeros((len(theta_arr), len(cr_arr)))

        cr_original = self.multirotor.mesh.contact_ratio


        if square_varying_stiffness:
            # Definimos os parâmetros baseados no problema de engrenagens
            Tm = pitch_angle
            
            # Criamos uma matriz de limites (um limite para cada valor de cr_val)
            # cr_arr tem tamanho (len_cr,), então limites terá o mesmo tamanho.
            limites = (cr_arr - 1) * Tm
            
            # Em vez de dois loops for, podemos processar cada th (cada linha i) 
            # para todos os cr_val (colunas j) de uma vez usando vetorização do NumPy.
            for i, th in enumerate(tqdm(theta_arr, desc="Gerando K_table (Square Mode)")):
                # th % Tm garante a periodicidade dentro do passo de dente
                fase = th % Tm
                
                # np.where aninhado:
                # 1. Se for menor que o limite -> kd
                # 2. Se for IGUAL ao limite (com tolerância) -> (kd + ks) / 2
                # 3. Se for maior -> ks
                K_table[i, :] = np.where(
                    fase < limites, 
                    kd, 
                    np.where(np.isclose(fase, limites), (kd + ks) / 2, ks)
                )

        else:
            # Mantém sua lógica original para o caso False
            for i, th in enumerate(tqdm(theta_arr, desc="Gerando K_table")):
                for j, cr_val in enumerate(cr_arr):
                    # theta_range, stiffness_range = self.multirotor.mesh.get_stiffness_for_mesh_period()
                    # self.multirotor.mesh.theta_range = theta_range
                    # self.multirotor.mesh.stiffness_range = stiffness_range
                    # K_table[i, j] = self.multirotor.mesh.interpolate_stiffness(angular_position=th)
                    self.multirotor.mesh.contact_ratio = cr_val
                    K_table[i, j] = self.multirotor.mesh.get_variable_stiffness(angular_position=th)


        self.multirotor.mesh.contact_ratio = cr_original

        np.savez(caminho_completo, theta_arr=theta_arr, cr_arr=cr_arr, K_table=K_table)
        return theta_arr, cr_arr, K_table
    
    # =========================================================================
    # BLOCO 7: ROTEADORES DOS INTEGRADORES NUMÉRICOS
    # =========================================================================

    def compute_backlash_force(self, step, time_step, disp_resp, velc_resp, accl_resp, **kwargs):
        """Callback compatível com a API nativa do método de integração do ROSS."""
        
        # =========================================================================
        # CORREÇÃO: Garante que a tabela de rigidez (Lookup Table) está carregada
        # antes do solver do ROSS calcular a primeira força.
        # =========================================================================
        if not hasattr(self, 'theta_arr'):
            self.theta_arr, self.cr_arr, self.K_table = self._get_or_create_stiffness_table()
        # =========================================================================

        gear_nodes = np.array([e.n for e in self.gears], dtype=np.int64)
        number_of_dof = self.multirotor.number_dof
        ndof_total = self.multirotor.ndof
        
        R1, R2 = self.gears[0].base_radius, self.gears[1].base_radius
        Ra1, Ra2 = self.gears[0].radii_dict["addendum"], self.gears[1].radii_dict["addendum"]
        module, alfa0 = self.gears[0].module, self.gears[0].pr_angle
        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2
        orientation_angle = self.multirotor.orientation_angle
        nominal_cr = self.multirotor.mesh.contact_ratio
        
        # --- NOVO: Extração do Ângulo de Hélice ---
        helix_angle = self.multirotor.mesh.helix_angle
        
        wm = self.speed_driving_gear * self.gears[0].n_teeth
        error_step = self.error[step]
        error_dot_step = self.error_amp * wm * np.cos(wm * self.time[step]) 
        angular_pos = self.speed_driving_gear * self.time[step]

        backlash_force, logs = calculate_dynamic_backlash_force(
            disp_resp, velc_resp, gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, self.b0,
            error_step, angular_pos, self.compute_contact_ratio, nominal_cr,
            Ra1, Ra2, module, self.sigma, self.smooth_operator, # <-- ADICIONADO AQUI
            self.theta_arr, self.cr_arr, self.K_table,
            self.M_eq, self.mesh_damping_ratio, error_dot_step 
        )

        keys = ["x1", "y1", "x2", "y2", "t1", "t2", "d", "beta", "alfa", 
                "contact_ratio", "delta", "bt", "f", "K_time", "Fm"]
        for i, key in enumerate(keys):
            self.backlash_results[key][step] = logs[i]

        self.backlash_total_force[step, :] = backlash_force
        self.multirotor.contact_ratio = logs[9] 
        return backlash_force

    def runge_kutta(self, F_unb):
        """Integra o sistema global convertendo o modelo do ROSS para Espaço de Estados."""
        sysc_ross = self.multirotor._lti(speed=self.speed_driving_gear)
        A, B = np.ascontiguousarray(sysc_ross.A), np.ascontiguousarray(sysc_ross.B)
        F_unb = np.ascontiguousarray(F_unb)

        dt = self.time[1] - self.time[0]
        n_steps = len(self.time)
        wm = self.speed_driving_gear * self.gears[0].n_teeth

        gear_nodes = np.array([e.n for e in self.gears], dtype=np.int64)
        number_of_dof, ndof_total = self.multirotor.number_dof, self.multirotor.ndof
        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2
        R1, R2 = self.gears[0].base_radius, self.gears[1].base_radius
        alfa0 = self.gears[0].pr_angle
        orientation_angle = self.multirotor.orientation_angle
        nominal_cr = self.multirotor.mesh.contact_ratio
        Ra1, Ra2 = self.gears[0].radii_dict["addendum"], self.gears[1].radii_dict["addendum"]
        module = self.gears[0].module
        
        # --- NOVO: Extração do Ângulo de Hélice ---
        helix_angle = self.multirotor.mesh.helix_angle

        if not hasattr(self, 'theta_arr'):
            self.theta_arr, self.cr_arr, self.K_table = self._get_or_create_stiffness_table()

        yout = np.zeros((n_steps, A.shape[0]))
        logs_matrix = np.zeros((n_steps, 15))
        force_matrix = np.zeros((n_steps, ndof_total))

        print(f"\nIniciando RK45 de alta performance para {n_steps} passos com dt = {dt:.2e} s...")

        # import time
        start_time_rk45 = time.time()

        rk45_solver_full(
            self.time, dt, yout, logs_matrix, force_matrix,
            A, B, F_unb, gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, self.b0, 
            self.error_amp, wm, self.speed_driving_gear, self.compute_contact_ratio, nominal_cr,
            Ra1, Ra2, module, self.sigma, self.smooth_operator, # <-- ADICIONADO AQUI
            self.theta_arr, self.cr_arr, self.K_table,
            self.M_eq, self.mesh_damping_ratio, start_time_rk45
        )

        keys = ["x1", "y1", "x2", "y2", "t1", "t2", "d", "beta", "alfa",
                "contact_ratio", "delta", "bt", "f", "K_time", "Fm"]
        for idx, key in enumerate(keys): self.backlash_results[key] = logs_matrix[:, idx]
        self.backlash_total_force = force_matrix

        return yout

    def internal_newmark(self, F_unb, gamma=0.5, beta=0.25, tol=1e-6, epsilon=1e-8,
                         y_init=None, ydot_init=None, y2dot_init=None, # <-- SWEEP
                         theta_array=None, error_array=None, error_dot_array=None): # <-- CINEMÁTICA
        """Integra o sistema global usando Newmark Adaptativo C/C++ customizado."""
        M = np.ascontiguousarray(self.multirotor.M())
        K_sys = np.ascontiguousarray(self.multirotor.K(self.speed_driving_gear))
        F_unb = np.ascontiguousarray(F_unb)
        
        C_base = np.asarray(self.multirotor.C(self.speed_driving_gear))
        G_mat = np.asarray(self.multirotor.G())
        C_sys = np.ascontiguousarray(C_base + G_mat * self.speed_driving_gear)

        

        n_steps = len(self.time)
        wm = self.speed_driving_gear * self.gears[0].n_teeth

        gear_nodes = np.array([e.n for e in self.gears], dtype=np.int64)
        number_of_dof, ndof_total = self.multirotor.number_dof, self.multirotor.ndof

        # A VARIÁVEL É CRIADA AQUI:
        gear_nodes = np.array([e.n for e in self.gears], dtype=np.int64)
        number_of_dof, ndof_total = self.multirotor.number_dof, self.multirotor.ndof

        # =====================================================================
        # CORREÇÃO FÍSICA: AMORTECIMENTO TORSIONAL DOS MANCAIS/CARGA
        # Agora sim, colamos o bloco aqui, pois o gear_nodes e o number_of_dof já existem!
        # =====================================================================
        # idx1, idx2 = gear_nodes[0] * number_of_dof, gear_nodes[1] * number_of_dof
        # c_torsional = 15.0  # Arraste torsional para segurar o caos
        
        # C_sys[idx1+5, idx1+5] += c_torsional
        # C_sys[idx2+5, idx2+5] += c_torsional
        # =====================================================================

        # PROTEÇÃO PARA A PARTIDA A FRIO (Primeira RPM da varredura)
        if y_init is None: y_init = np.zeros(ndof_total)
        if ydot_init is None: ydot_init = np.zeros(ndof_total)
        if y2dot_init is None: y2dot_init = np.zeros(ndof_total)

        # Detecta automaticamente todos os GLs envolvidos, independente se estão batendo ou na folga:
        active_dofs = np.concatenate([
            np.arange(node * self.multirotor.number_dof, (node + 1) * self.multirotor.number_dof)
            for node in gear_nodes
        ]).astype(np.int64)
        
        # # Modificando active_dofs para pegar os 6 GLs dos nós das engrenagens 
        # # (translação x,y,z e rotação rx,ry,rz)
        # idx1, idx2 = number_of_dof * gear_nodes[0], number_of_dof * gear_nodes[1]
        # active_dofs = np.array([
        #     idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5, 
        #     idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5
        # ], dtype=np.int64)

        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2
        R1, R2 = self.gears[0].base_radius, self.gears[1].base_radius
        alfa0 = self.gears[0].pr_angle
        orientation_angle = self.multirotor.orientation_angle
        nominal_cr = self.multirotor.mesh.contact_ratio
        Ra1, Ra2 = self.gears[0].radii_dict["addendum"], self.gears[1].radii_dict["addendum"]
        module = self.gears[0].module
        
        # --- NOVO: Extração do Ângulo de Hélice ---
        helix_angle = self.multirotor.mesh.helix_angle

        if not hasattr(self, 'theta_arr'):
            self.theta_arr, self.cr_arr, self.K_table = self._get_or_create_stiffness_table()

        yout = np.zeros((n_steps, ndof_total))
        logs_matrix = np.zeros((n_steps, 15))
        force_matrix = np.zeros((n_steps, ndof_total))

        print(f"\nIniciando Newmark Adaptativo Interno para {n_steps} passos...")

        # import time
        start_time_newmark = time.time()

        # PASSA TUDO PARA O NUMBA E CAPTURA O RETORNO
        final_ydot, final_y2dot = newmark_solver_full(
            self.time, yout, logs_matrix, force_matrix, F_unb,
            M, C_sys, K_sys, active_dofs, gamma, beta, tol, epsilon,
            gear_nodes, number_of_dof, ndof_total, d0, 
            orientation_angle, R1, R2, alfa0, helix_angle, self.b0, 
            self.compute_contact_ratio, nominal_cr, Ra1, Ra2, 
            module, self.sigma, self.smooth_operator, self.theta_arr, self.cr_arr, self.K_table,
            self.M_eq, self.mesh_damping_ratio, start_time_newmark,
            y_init, ydot_init, y2dot_init,
            theta_array, error_array, error_dot_array
        )

        # SALVA NA CLASSE PARA O PRÓXIMO PASSO
        self.final_states = (yout[-1, :], final_ydot, final_y2dot)

        keys = ["x1", "y1", "x2", "y2", "t1", "t2", "d", "beta", "alfa",
                "contact_ratio", "delta", "bt", "f", "K_time", "Fm"]
        for idx, key in enumerate(keys): self.backlash_results[key] = logs_matrix[:, idx]
        self.backlash_total_force = force_matrix

        return yout
    
    def run_scipy_bdf(self, F_unb):
        """Integra o sistema global usando o solver BDF do SciPy (Equivalente ao ode15s do MATLAB)."""
        from scipy.integrate import solve_ivp
        import time

        # sysc_ross = self.multirotor._lti(speed=self.speed_driving_gear)
        # A = np.ascontiguousarray(sysc_ross.A)
        # B = np.ascontiguousarray(sysc_ross.B)

        # NOVA MONTAGEM (Segura e Regularizada):
        M_mat = self.multirotor.M()
        C_mat = self.multirotor.C(self.speed_driving_gear)
        K_mat = self.multirotor.K(self.speed_driving_gear)
        
        # Adiciona uma massa "fantasma" de 1e-10 para evitar divisões por zero
        M_safe = M_mat + np.eye(M_mat.shape[0]) * 1e-10
        M_inv = np.linalg.inv(M_safe)
        
        ndof_total = M_mat.shape[0]
        
        # Montagem analítica do Espaço de Estados
        A = np.zeros((2 * ndof_total, 2 * ndof_total))
        A[:ndof_total, ndof_total:] = np.eye(ndof_total)
        A[ndof_total:, :ndof_total] = -M_inv @ K_mat
        A[ndof_total:, ndof_total:] = -M_inv @ C_mat
        
        B = np.zeros((2 * ndof_total, ndof_total))
        B[ndof_total:, :] = M_inv
        
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)

        F_unb = np.ascontiguousarray(F_unb)

        gear_nodes = np.array([e.n for e in self.gears], dtype=np.int64)
        number_of_dof, ndof_total = self.multirotor.number_dof, self.multirotor.ndof
        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2
        R1, R2 = self.gears[0].base_radius, self.gears[1].base_radius
        alfa0 = self.gears[0].pr_angle
        orientation_angle = self.multirotor.orientation_angle
        nominal_cr = self.multirotor.mesh.contact_ratio
        Ra1, Ra2 = self.gears[0].radii_dict["addendum"], self.gears[1].radii_dict["addendum"]
        module = self.gears[0].module
        helix_angle = self.multirotor.mesh.helix_angle
        wm = self.speed_driving_gear * self.gears[0].n_teeth

        if not hasattr(self, 'theta_arr'):
            self.theta_arr, self.cr_arr, self.K_table = self._get_or_create_stiffness_table()

        y0 = np.zeros(A.shape[0])
        t_span = (self.time[0], self.time[-1])

        print(f"\nIniciando integração implícita (ODE15s / BDF) via SciPy...")
        start_time_bdf = time.time()

        # Empacota os argumentos físicos para a função compilada do Numba
        args = (self.time, F_unb, A, B, gear_nodes, number_of_dof, ndof_total,
                d0, orientation_angle, R1, R2, alfa0, helix_angle, self.b0,
                self.error_amp, wm, self.speed_driving_gear, self.compute_contact_ratio, nominal_cr,
                Ra1, Ra2, module, self.sigma, self.smooth_operator,
                self.theta_arr, self.cr_arr, self.K_table, self.M_eq, self.mesh_damping_ratio)

        # --- LÓGICA DA BARRA DE PROGRESSO CUSTOMIZADA ---
        state = {
            'max_t': t_span[0],
            'last_print_time': time.time(),
            'start_time': time.time(),
            'last_t_eval': t_span[0]
        }  
        
        # Intervalo de atualização em segundos REAIS (ex: atualiza a tela a cada 0.1s)
        # Isso substitui o "print_interval" dos passos, garantindo que o terminal não pisque loucamente
        PRINT_UPDATE_RATE = 0.1 

        def rhs_with_progress(t, y, *args):
            current_time = time.time()
            
            # Atualiza a barra apenas se o solver avançou no tempo E se passou o tempo do update da tela
            if t > state['max_t']:
                dt = t - state['max_t']
                state['max_t'] = t
                
                if (current_time - state['last_print_time']) >= PRINT_UPDATE_RATE:
                    print_ode15s_progress(
                        current_t=t, 
                        t_start=t_span[0], 
                        t_end=t_span[1], 
                        start_time=state['start_time'], 
                        method_name="ODE15s (BDF)", 
                        dt=dt
                    )
                    state['last_print_time'] = current_time
            
            # Chama a sua rotina Numba ultrarrápida
            return bdf_rhs_wrapper(t, y, *args)

        # Chama o solver implícito BDF passando a função com a barra de progresso
        sol = solve_ivp(
            fun=rhs_with_progress,
            t_span=t_span,
            y0=y0,
            t_eval=self.time,
            method='BDF',
            args=args,
            rtol=1e-3, 
            atol=1e-6
        )
        
        # Print final para travar a barra em 100%
        print_ode15s_progress(t_span[1], t_span[0], t_span[1], state['start_time'], "ODE15s (BDF)", 0.0)
        print() # Quebra a linha final
        # -----------------------------------

        if not sol.success:
            print("AVISO: A integração BDF falhou. Mensagem do solver:", sol.message)

        yout_full = sol.y.T # Transpõe para o formato (n_steps, 2*ndof_total)
        
        print(f"Integração BDF concluída em {time.time() - start_time_bdf:.2f} s. Extraindo logs da órbita...")

        # Separa posições e velocidades para extrair as forças da malha
        yout_disp = np.ascontiguousarray(yout_full[:, :ndof_total])
        yout_vel = np.ascontiguousarray(yout_full[:, ndof_total:])

        # Reutiliza a sua rotina inteligente de extração de logs do Numba!
        logs_matrix, force_matrix = extract_backlash_logs_from_trajectory(
            yout_disp, yout_vel, self.time, gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, self.b0, 
            self.error_amp, wm, self.speed_driving_gear, self.compute_contact_ratio, nominal_cr,
            Ra1, Ra2, module, self.sigma, self.smooth_operator, self.theta_arr, self.cr_arr, self.K_table,
            self.M_eq, self.mesh_damping_ratio, time.time()
        )

        keys = ["x1", "y1", "x2", "y2", "t1", "t2", "d", "beta", "alfa",
                "contact_ratio", "delta", "bt", "f", "K_time", "Fm"]
        for idx, key in enumerate(keys): 
            self.backlash_results[key] = logs_matrix[:, idx]
        self.backlash_total_force = force_matrix

        return yout_full
    
    def run_dynamic_backlash(self, 
                             unb_node,
                             unb_magnitude,
                             unb_phase,
                             integration_method="ross_newmark",
                             add_force=None,
                             sigma=1e4,             
                             smooth_operator=True,    
                             y_init=None, ydot_init=None, y2dot_init=None, # <-- SWEEP
                             **kwargs):
        """Função Principal Roteadora - Prepara Forças e Chama o Integrador Selecionado."""
        # Atualiza as propriedades na classe para os solvers puxarem

        start_time_total = time.perf_counter() # <-- INICIA O CRONÔMETRO AQUI

        self.sigma = sigma
        self.smooth_operator = smooth_operator
        
        ramp_fraction = kwargs.get('ramp_fraction', 0.0)
        speed_array = self.generate_speed_ramp(ramp_fraction=ramp_fraction)

        theta_array = cumulative_trapezoid(speed_array, self.time, initial=0.0)
        
        z1 = self.gears[0].n_teeth
        error_array = self.error_amp * np.sin(z1 * theta_array)
        error_dot_array = self.error_amp * (z1 * speed_array) * np.cos(z1 * theta_array)

        # Força de desbalanceamento
        self.unb_force, _, _, _ = self.multirotor.unbalance_force_over_time(
            unb_node, unb_magnitude, unb_phase, speed_array, self.time, return_all=True)
        
        F = self.unb_force.T
        if add_force is not None: F += add_force

        print(f"==================================================")
        print(f"Iniciando simulação com o integrador: '{integration_method.upper()}'")
        print(f"==================================================")

        if integration_method.lower() == "rk45":
            yout = self.runge_kutta(F_unb=F)
            yout_disp = yout[:, :self.multirotor.ndof]
            results = TimeResponseResults(rotor=self.multirotor, t=self.time, yout=yout_disp, xout=[])

        elif integration_method.lower() == "ross_newmark":
            results = self.multirotor.run_time_response(
                speed=self.speed_driving_gear, F=F, t=self.time, method="newmark", 
                add_to_RHS=self.compute_backlash_force, **kwargs
            )

        elif integration_method.lower() == "internal_newmark":
            gamma = kwargs.get('gamma', 0.5)
            beta = kwargs.get('beta', 0.25)
            tol = kwargs.get('tol', 1e-6)
            yout_disp = self.internal_newmark(
                F_unb=F, gamma=gamma, beta=beta, tol=tol,
                y_init=y_init, ydot_init=ydot_init, y2dot_init=y2dot_init, # <-- SWEEP
                theta_array=theta_array, error_array=error_array, error_dot_array=error_dot_array # <-- CINEMÁTICA
            )
            results = TimeResponseResults(rotor=self.multirotor, t=self.time, yout=yout_disp, xout=[])

        elif integration_method.lower() in ["ode15s", "bdf"]:
            yout = self.run_scipy_bdf(F_unb=F)
            yout_disp = yout[:, :self.multirotor.ndof]
            results = TimeResponseResults(rotor=self.multirotor, t=self.time, yout=yout_disp, xout=[])
            
        else:
            raise ValueError(f"Método '{integration_method}' inválido! Escolha 'rk45', 'ross_newmark' ou 'internal_newmark'.")

        self.cut_backlash_results()
        
        results.yout = results.yout[self.n_cut:, :]
        results.t = results.t[self.n_cut:]
        # results.t -= results.t[0]
        self.time = self.time[self.n_cut:]
        self.time_response = results

        self.exec_time_dynamic = time.perf_counter() - start_time_total
                                                       
        return results

    def run_linear_baseline(self, unb_node, unb_magnitude, unb_phase, add_force=None, 
                            sigma=1e4, smooth_operator=True, **kwargs): 
        
        """
        Executa a simulação linear em DOIS PASSOS (Pseudo-Acoplamento Iterativo):
        1. Roda com desbalanceamento puro para encontrar a órbita linear.
        2. Extrai a força exata do engrenamento para essa órbita via Numba.
        3. Roda novamente o ROSS aplicando o desbalanceamento + força de engrenamento.
        """
        import time
        import numpy as np
        from scipy.integrate import cumulative_trapezoid # <-- IMPORTAÇÃO NECESSÁRIA

        start_time_total = time.perf_counter() # <-- INICIA O CRONÔMETRO AQUI

        self.sigma = sigma
        self.smooth_operator = smooth_operator

        ramp_fraction = kwargs.get('ramp_fraction', 0.0)
        speed_array = self.generate_speed_ramp(ramp_fraction=ramp_fraction)

        # =====================================================================
        # CORREÇÃO: CÁLCULO FÍSICO DA POSIÇÃO ANGULAR PARA VELOCIDADE VARIÁVEL
        # Como mudamos a arquitetura para suportar o soft-start/sweep, precisamos
        # gerar os vetores cinemáticos aqui também antes de chamar o Numba!
        # =====================================================================
        theta_array = cumulative_trapezoid(speed_array, self.time, initial=0.0)
        
        z1 = self.gears[0].n_teeth
        error_array = self.error_amp * np.sin(z1 * theta_array)
        error_dot_array = self.error_amp * (z1 * speed_array) * np.cos(z1 * theta_array)
        # =====================================================================

        # ---------------------------------------------------------------------
        # PASSO 1: CALCULAR FORÇA DE DESBALANCEAMENTO
        # ---------------------------------------------------------------------
        unb_force, _, _, _ = self.multirotor.unbalance_force_over_time(
            unb_node, unb_magnitude, unb_phase, speed_array, self.time, return_all=True)
        
        F_unb = unb_force.T
        if add_force is not None: 
            F_unb += add_force

        print(f"==================================================")
        print(f"Iniciando simulação BASELINE LINEAR (2 Passos Iterativos)")
        print(f"==================================================")

        # ---------------------------------------------------------------------
        # PASSO 2: PRIMEIRA SIMULAÇÃO (Apenas Desbalanceamento)
        # ---------------------------------------------------------------------
        print("-> Passo 1/2: Simulação ROSS (Apenas Desbalanceamento)...")
        t1 = time.time()
        results_step1 = self.multirotor.run_time_response(
            speed=speed_array, F=F_unb, t=self.time, method="default", **kwargs
        )
        print(f"   Tempo decorrido Passo 1: {time.time()-t1:.2e} s")

        ndof_total = self.multirotor.ndof
        yout_raw_1 = np.ascontiguousarray(results_step1.yout)
        dt = self.time[1] - self.time[0]

        if yout_raw_1.shape[1] == 2 * ndof_total:
            yout_1 = np.ascontiguousarray(yout_raw_1[:, :ndof_total])
            ydot_1 = np.ascontiguousarray(yout_raw_1[:, ndof_total:])
        else:
            yout_1 = yout_raw_1
            ydot_1 = np.ascontiguousarray(np.gradient(yout_raw_1, dt, axis=0))

        # ---------------------------------------------------------------------
        # PASSO 3: EXTRAÇÃO DAS FORÇAS EXATAS DA MALHA (Via Órbita 1)
        # ---------------------------------------------------------------------
        gear_nodes = np.array([e.n for e in self.gears], dtype=np.int64)
        number_of_dof = self.multirotor.number_dof
        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2
        R1, R2 = self.gears[0].base_radius, self.gears[1].base_radius
        alfa0 = self.gears[0].pr_angle
        orientation_angle = self.multirotor.orientation_angle
        nominal_cr = self.multirotor.mesh.contact_ratio
        Ra1, Ra2 = self.gears[0].radii_dict["addendum"], self.gears[1].radii_dict["addendum"]
        module = self.gears[0].module
        helix_angle = self.multirotor.mesh.helix_angle

        if not hasattr(self, 'theta_arr'):
            self.theta_arr, self.cr_arr, self.K_table = self._get_or_create_stiffness_table()

        print("-> Calculando Força de Engrenamento Teórica via Numba...")
        start_time_extract = time.time() 

        # Extrai os logs e a Matriz de Força usando os arrays cinemáticos gerados
        logs_matrix_1, F_mesh_global = extract_backlash_logs_from_trajectory(
            yout_1, ydot_1, self.time, gear_nodes, number_of_dof, ndof_total,
            d0, orientation_angle, R1, R2, alfa0, helix_angle, self.b0, 
            self.compute_contact_ratio, nominal_cr, 
            Ra1, Ra2, module, self.sigma, self.smooth_operator, self.theta_arr, self.cr_arr, self.K_table,
            self.M_eq, self.mesh_damping_ratio, start_time_extract,
            theta_array, error_array, error_dot_array # <-- OS VETORES AGORA EXISTEM E SÃO PASSADOS
        )

        # ---------------------------------------------------------------------
        # PASSO 4: A NOVA FORÇA TOTAL (Desbalanceamento + Matriz Numba)
        # ---------------------------------------------------------------------
        F_total = F_unb + F_mesh_global

        # ---------------------------------------------------------------------
        # PASSO 5: SEGUNDA SIMULAÇÃO (Órbita Acoplada)
        # ---------------------------------------------------------------------
        print("-> Passo 2/2: Simulação ROSS (Desbalanceamento + Malha)...")
        t2 = time.time()
        results_final = self.multirotor.run_time_response(
            speed=speed_array, F=F_total, t=self.time, method="default", **kwargs
        )
        print(f"   Tempo decorrido Passo 2: {time.time()-t2:.2e} s")

        # ---------------------------------------------------------------------
        # PASSO 6: SALVAR RESULTADOS FINAIS
        # ---------------------------------------------------------------------
        self.linear_backlash_results = {}
        keys = ["x1", "y1", "x2", "y2", "t1", "t2", "d", "beta", "alfa",
                "contact_ratio", "delta", "bt", "f", "K_time", "Fm"]
        
        for idx, key in enumerate(keys):
            self.linear_backlash_results[key] = logs_matrix_1[self.n_cut:, idx]

        results_final.yout = results_final.yout[self.n_cut:, :ndof_total] 
        results_final.t = results_final.t[self.n_cut:]
        
        self.linear_time_response = results_final
        self.exec_time_linear = time.perf_counter() - start_time_total
        
        print("Baseline linear recalculado e acoplado com sucesso!")
        return results_final

    # =========================================================================
    # BLOCO 8: PÓS-PROCESSAMENTO E EXPORTAÇÃO (DASHBOARDS E ARQUIVOS)
    # =========================================================================

    def save_results(self, unb_node, unb_magnitude, unb_phase, integration_method, output_dir="resultados_backlash", compress_csv=False):
        """Salva metadados detalhados, séries temporais e o objeto binário no diretório de execução."""

        # =====================================================================
        # CAPTURA O DIRETÓRIO E NOME DO ARQUIVO DE TESTE EM EXECUÇÃO
        # =====================================================================
        try:
            main_file = sys.modules['__main__'].__file__
            diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
            nome_arquivo_teste = os.path.splitext(os.path.basename(main_file))[0]
        except AttributeError:
            # Fallback para Jupyter Notebook
            diretorio_execucao = os.getcwd()
            nome_arquivo_teste = "jupyter_notebook"

        # Diretório base agora é sempre relativo ao local do script em execução
        if os.path.isabs(output_dir):
            base_dir = output_dir
        else:
            base_dir = os.path.join(diretorio_execucao, output_dir)
            
        if not os.path.exists(base_dir): 
            os.makedirs(base_dir)

        # Formato: simulacao_YYYY-MM-DD_15h-30m-45s
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
        run_folder = os.path.join(base_dir, f"simulacao_{timestamp}")
        os.makedirs(run_folder)

        # Função auxiliar para formatar tempo em HH:MM:SS
        def formatar_tempo(segundos_totais):
            horas, resto = divmod(segundos_totais, 3600)
            minutos, segundos = divmod(resto, 60)
            # Retorna no formato 00:00:00.000
            return f"{int(horas):02d}:{int(minutos):02d}:{segundos:06.3f}"

        # 1. SALVA O RELATÓRIO METADADOS (EXPANDIDO)
        report_path = os.path.join(run_folder, "relatorio_simulacao.txt")
        dt = self.time[1] - self.time[0]
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write("       RELATÓRIO DE SIMULAÇÃO - ROTODINÂMICA        \n")
            f.write("====================================================\n")
            f.write(f"Data e Hora           : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Diretório de Origem   : {base_dir}\n") 
            f.write(f"Script de Execução    : {nome_arquivo_teste}.py\n")
            f.write("----------------------------------------------------\n")
            
            f.write("TEMPOS DE PROCESSAMENTO (HH:MM:SS)\n")
            # Verifica e plota o tempo do Baseline Linear (se existir)
            if hasattr(self, 'exec_time_linear'):
                f.write(f"Baseline Linear       : {formatar_tempo(self.exec_time_linear)}\n")
            
            # Verifica e plota o tempo da Simulação Não-Linear (se existir)
            if hasattr(self, 'exec_time_dynamic'):
                f.write(f"Dinâmica Não-Linear   : {formatar_tempo(self.exec_time_dynamic)}\n")
            f.write("----------------------------------------------------\n")

            f.write("PARÂMETROS DE INTEGRAÇÃO E SOLVER\n")
            f.write(f"Método de Integração  : {integration_method.upper()}\n")
            f.write(f"Passo de Tempo (dt)   : {dt:.2e} s\n")
            f.write(f"Frequência de Amostr. : {1/dt:.2f} Hz\n")
            f.write(f"Ciclos Simulados      : {getattr(self, 'n_cicles', 'N/A')} (Totais) | {getattr(self, 'cut_cicles', 'N/A')} (Descartados)\n")
            f.write("----------------------------------------------------\n")
            
            f.write("CONFIGURAÇÕES DE NÃO-LINEARIDADE (CONTATO E FOLGA)\n")
            f.write(f"Smooth Operator       : {getattr(self, 'smooth_operator', 'Não Definido')}\n")
            f.write(f"Sigma (Suavização)    : {getattr(self, 'sigma', 'Não Definido')}\n")
            f.write(f"Cálculo de CR Dinâmico: {getattr(self, 'compute_contact_ratio', 'Não Definido')}\n")
            f.write(f"Usa Rigidez do ROSS   : {'Sim' if self.multirotor.update_mesh_stiffness else 'Não (Desativada)'}\n")
            f.write("----------------------------------------------------\n")
            
            f.write("GEOMETRIA DO ENGRENAMENTO E FÍSICA\n")
            f.write(f"Velocidade (Pinhão)   : {self.speed_driving_gear:.2f} rad/s\n")
            f.write(f"Frequência de Malha   : {(self.speed_driving_gear * self.gears[0].n_teeth)/(2*np.pi):.2f} Hz\n")
            f.write(f"Dentes (Z1 / Z2)      : {self.gears[0].n_teeth} / {self.gears[1].n_teeth}\n")
            f.write(f"Módulo                : {self.gears[0].module} mm\n")
            f.write(f"Ângulo de Pressão (a0): {np.degrees(self.gears[0].pr_angle):.2f} °\n")
            f.write(f"Ângulo de Hélice      : {np.degrees(self.multirotor.mesh.helix_angle):.2f} °\n")
            f.write(f"Backlash Inicial (b0) : {self.b0:.4e} m\n")
            f.write(f"Erro Estático (Amp)   : {self.error_amp:.4e} m\n")
            f.write(f"Razão de Amortecimento: {self.mesh_damping_ratio}\n")
            f.write(f"Massa Equivalente (M) : {self.M_eq:.4f} kg\n")
            f.write("----------------------------------------------------\n")
            
            f.write("PARÂMETROS DE DESBALANCEAMENTO\n")
            f.write(f"Nós Excitados         : {unb_node}\n")
            f.write(f"Magnitude (kg.m)      : {unb_magnitude}\n")
            f.write(f"Fase (rad)            : {unb_phase}\n")
            f.write("----------------------------------------------------\n")
            
            f.write(f"Possui Baseline Linear? : {'Sim' if hasattr(self, 'linear_backlash_results') else 'Não'}\n")
            f.write("====================================================\n")

        # 2. SALVA OS DADOS DA SIMULAÇÃO NÃO-LINEAR (PANDAS)
        if hasattr(self, 'backlash_results') and len(self.backlash_results["x1"]) > 0:
            csv_nl_path = os.path.join(run_folder, "historico_temporal_nao_linear.csv")
            if compress_csv: csv_nl_path += ".gz"
            
            df_nl = pd.DataFrame({"Tempo_s": self.time})
            for key, array_data in self.backlash_results.items():
                df_nl[key] = array_data
                
            df_nl.to_csv(csv_nl_path, index=False, compression='gzip' if compress_csv else None)

        # 3. SALVA OS DADOS DA SIMULAÇÃO LINEAR BASELINE (PANDAS)
        if hasattr(self, 'linear_backlash_results'):
            csv_lin_path = os.path.join(run_folder, "historico_temporal_linear.csv")
            if compress_csv: csv_lin_path += ".gz"
            
            df_lin = pd.DataFrame({"Tempo_s": self.linear_time_response.t})
            for key, array_data in self.linear_backlash_results.items():
                df_lin[key] = array_data
                
            df_lin.to_csv(csv_lin_path, index=False, compression='gzip' if compress_csv else None)

        # =========================================================================
        # 4. SALVA O OBJETO DA CLASSE INTEIRO (PICKLE) COM NOME DINÂMICO
        # =========================================================================
        nome_arquivo_pickle = f"modelo_completo_{nome_arquivo_teste}.pkl"
        pickle_path = os.path.join(run_folder, nome_arquivo_pickle)
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Objeto binário '{nome_arquivo_pickle}' salvo com sucesso!")
        except Exception as e:
            print(f"AVISO: Ocorreu um erro ao salvar o objeto binário: {e}")

        print(f"\nResultados salvos com sucesso na pasta:\n'{run_folder}'")
        return run_folder
    
    @staticmethod
    def load_model(filepath):
        """
        Carrega um modelo de simulação salvo anteriormente a partir de um arquivo .pkl.

        from seu_arquivo_onde_esta_a_classe import Backlash

        # Você NÃO cria os rotores nem as engrenagens de novo.
        # Simplesmente aponta para o arquivo que foi salvo na pasta ontem:
        modelo_recuperado = Backlash.load_model("resultados_backlash/simulacao_2026-03-17_15h-30m-45s/modelo_completo.pkl")

        # O objeto voltou à vida! Você pode plotar tudo direto:
        import plotly.express as px
        fig = px.line(x=modelo_recuperado.time, y=modelo_recuperado.backlash_results["Fm"])
        fig.show()

        # E até mesmo os métodos originais do ROSS continuam funcionando!
        modelo_recuperado.multirotor.plot_rotor()
        
        Exemplo de uso:
        >>> from seu_script import Backlash
        >>> meu_modelo = Backlash.load_model("caminho/para/modelo_completo.pkl")
        """        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
            
        print(f"Carregando modelo do arquivo '{filepath}'...")
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
            
        print("Modelo carregado com sucesso!")
        return loaded_model

    def plot_dashboard(self, freq_unit="Hz", decimation=1, save_path=None, is_linear=False):
        """
        Gera um painel interativo consolidado.
        Se is_linear=True, gera o gráfico para os dados do Baseline Linear.
        O HTML será salvo na mesma pasta do backlash.py se um caminho absoluto não for fornecido.
        """
        # Garante que salva na mesma pasta do script que CHAMOU a execução (validation)
        if save_path:
            try:
                main_file = sys.modules['__main__'].__file__
                diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
            except AttributeError:
                diretorio_execucao = os.getcwd()
                
            if not os.path.isabs(save_path):
                save_path = os.path.join(diretorio_execucao, save_path)

        # Seleciona qual dicionário de dados usar (Linear ou Não-Linear)
        if is_linear:
            if not hasattr(self, 'linear_backlash_results'):
                print("ERRO: Resultados lineares não encontrados. Rode 'run_linear_baseline()' primeiro.")
                return
            dados = self.linear_backlash_results
            tempo = self.linear_time_response.t
            titulo = "Dashboard de Resultados LINEARES (Baseline) - Engrenamento"
        else:
            dados = self.backlash_results
            tempo = self.time
            titulo = "Dashboard de Resultados NÃO-LINEARES - Engrenamento"

        # Extração das grandezas nominais e geométricas
        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2
        Ra1, Ra2 = self.gears[0].radii_dict["addendum"], self.gears[1].radii_dict["addendum"]
        Rb1, Rb2 = self.gears[0].base_radius, self.gears[1].base_radius
        Rf1, Rf2 = self.gears[0].radii_dict["root"], self.gears[1].radii_dict["root"]
        
        alfa0_deg = np.degrees(self.gears[0].pr_angle)
        beta0_deg = np.degrees(self.multirotor.orientation_angle)
        nominal_cr = self.multirotor.mesh.contact_ratio # <-- Extrai o CR Nominal

        # Aplica a decimação para não travar o navegador
        t_plot = tempo[::decimation]
        res = {k: v[::decimation] for k, v in dados.items()}
        
        # Criação dos vetores constantes para plotagem
        d0_vec = np.full_like(t_plot, d0)
        Ra_vec = np.full_like(t_plot, Ra1 + Ra2)
        Rb_vec = np.full_like(t_plot, Rb1 + Rb2)
        Rf_vec = np.full_like(t_plot, Rf1 + Ra2)
        alfa0_vec = np.full_like(t_plot, alfa0_deg)
        beta0_vec = np.full_like(t_plot, beta0_deg)
        cr0_vec = np.full_like(t_plot, nominal_cr) # <-- Cria o vetor constante do CR

        # Nova grelha: 5 linhas x 2 colunas
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                "Erro Dinâmico de Transmissão (DTE) vs Backlash (+bt)", "Força Dinâmica de Engrenamento (DMF)",
                "Distância entre Centros Instantânea", "Rigidez de Engrenamento (k_m)",
                "Ângulo de Pressão Dinâmico (α)", "Ângulo de Posição (β)",
                "Razão de Contato Instantânea", "Órbitas dos Rotores (R1 e R2)",
                f"Espectro FFT (Fm) [{freq_unit}]", f"Espectro FFT (k_m) [{freq_unit}]"
            ),
            vertical_spacing=0.06, horizontal_spacing=0.08
        )

        # --- Linha 1 (DTE sem -bt) ---
        fig.add_trace(go.Scattergl(x=t_plot, y=res["delta"], name="δ(t)", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=res["bt"], name="+bt", line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=res["Fm"], name="Fm", line=dict(color='purple')), row=1, col=2)

        # --- Linha 2 ---
        fig.add_trace(go.Scattergl(x=t_plot, y=res["d"], name="d(t)", line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=d0_vec, name="d0 (Nominal)", line=dict(color='black', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=Ra_vec, name="Ra1+Ra2", line=dict(color='orange', dash='dashdot')), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=Rf_vec, name="Rf1+Ra2", line=dict(color='brown', dash='dashdot')), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=Rb_vec, name="Rb1+Rb2", line=dict(color='magenta', dash='dashdot')), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=res["K_time"], name="K_m(t)", line=dict(color='teal')), row=2, col=2)

        # --- Linha 3 (Ângulos Dinâmicos vs Nominais) ---
        fig.add_trace(go.Scattergl(x=t_plot, y=np.degrees(res["alfa"]), name="α(t) [deg]", line=dict(color='darkorange')), row=3, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=alfa0_vec, name="α0 (Nominal)", line=dict(color='black', dash='dot')), row=3, col=1)
        
        fig.add_trace(go.Scattergl(x=t_plot, y=np.degrees(res["beta"]), name="β(t) [deg]", line=dict(color='darkcyan')), row=3, col=2)
        fig.add_trace(go.Scattergl(x=t_plot, y=beta0_vec, name="β0 (Nominal)", line=dict(color='black', dash='dot')), row=3, col=2)

        # --- Linha 4 ---
        fig.add_trace(go.Scattergl(x=t_plot, y=res["contact_ratio"], name="CR(t)", line=dict(color='olive')), row=4, col=1)
        fig.add_trace(go.Scattergl(x=t_plot, y=cr0_vec, name="CR0 (Nominal)", line=dict(color='black', dash='dot')), row=4, col=1) # <-- Novo traço do CR Nominal
        fig.add_trace(go.Scattergl(x=res["x1"], y=res["y1"], name="Órbita Pinhão", line=dict(color='blue')), row=4, col=2)
        fig.add_trace(go.Scattergl(x=res["x2"], y=res["y2"], name="Órbita Coroa", line=dict(color='red')), row=4, col=2)

        # --- Linha 5 (Chamando a função externa compute_dfft que agora usa numpy puro) ---
        freq_fm, amp_fm = compute_dfft(dados["Fm"], tempo, freq_unit=freq_unit)
        freq_km, amp_km = compute_dfft(dados["K_time"], tempo, freq_unit=freq_unit)
        
        fig.add_trace(go.Scattergl(x=freq_fm, y=amp_fm, name="FFT(Fm)", line=dict(color='indigo')), row=5, col=1)
        fig.add_trace(go.Scattergl(x=freq_km, y=amp_km, name="FFT(K_m)", line=dict(color='darkgreen')), row=5, col=2)

        # --- Configurações de Layout ---
        fig.update_layout(title_text=titulo, height=1400, width=1400, template="plotly_white", hovermode="x unified")
        
        for r in range(1, 6):
            fig.update_xaxes(title_text="Tempo (s)", row=r, col=1)
            if r != 4:
                fig.update_xaxes(title_text="Tempo (s)", row=r, col=2)
                
        fig.update_xaxes(title_text="x (m)", scaleanchor="y8", scaleratio=1, row=4, col=2) 
        fig.update_xaxes(title_text=f"Frequência ({freq_unit})", row=5, col=1)
        fig.update_xaxes(title_text=f"Frequência ({freq_unit})", row=5, col=2)
        
        fig.update_yaxes(title_text="Deslocamento (m)", row=1, col=1); fig.update_yaxes(title_text="Força (N)", row=1, col=2)
        fig.update_yaxes(title_text="Distância (m)", row=2, col=1);   fig.update_yaxes(title_text="Rigidez (N/m)", row=2, col=2)
        fig.update_yaxes(title_text="Ângulo (graus)", row=3, col=1);  fig.update_yaxes(title_text="Ângulo (graus)", row=3, col=2)
        fig.update_yaxes(title_text="Razão de Contato", row=4, col=1); fig.update_yaxes(title_text="y (m)", row=4, col=2)
        fig.update_yaxes(title_text="Amplitude (N)", row=5, col=1);   fig.update_yaxes(title_text="Amplitude (N/m)", row=5, col=2)

        if save_path:
            fig.write_html(save_path, include_plotlyjs="cdn")
            print(f"Dashboard leve HTML salvo em: {save_path}")
        else:
            fig.show()

    def save_and_plot_linear_baseline(self, csv_filename="baseline_linear_dados.csv", plot_filename="baseline_linear_plot.html"):
        """
        Guarda os dados do baseline linear num ficheiro CSV e gera um painel 
        interativo (Plotly HTML) com os gráficos das principais variáveis.
        """

        if not hasattr(self, 'linear_backlash_results') or not self.linear_backlash_results:
            raise ValueError("Nenhum resultado linear encontrado. Execute run_linear_baseline() primeiro.")

        print(f"\nA guardar os dados em '{csv_filename}'...")

        # 1. EXPORTAÇÃO DOS DADOS PARA CSV
        time_arr = self.linear_time_response.t
        data = {"Time": time_arr}
        for key, value_array in self.linear_backlash_results.items():
            data[key] = value_array

        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        print("Dados CSV guardados com sucesso!")

        # 2. GERAÇÃO E EXPORTAÇÃO DOS GRÁFICOS (PLOT)
        print(f"A gerar os gráficos em '{plot_filename}'...")
        
        # Criação de um painel com 3 subgráficos empilhados
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Força Dinâmica de Engrenamento (Fm)", 
                "Erro de Transmissão vs. Folga (Delta e limites bt)", 
                "Rigidez Variável no Tempo (K_time)"
            )
        )

        res = self.linear_backlash_results

        # Subgráfico 1: Força Fm
        fig.add_trace(go.Scatter(x=time_arr, y=res["Fm"], name="Fm", line=dict(color='blue')), row=1, col=1)

        # Subgráfico 2: Delta e Limites da Folga
        fig.add_trace(go.Scatter(x=time_arr, y=res["delta"], name="Delta", line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_arr, y=res["bt"], name="+bt (Limite Folga)", line=dict(color='red', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_arr, y=-res["bt"], name="-bt (Limite Folga)", line=dict(color='red', dash='dash')), row=2, col=1)

        # Subgráfico 3: Rigidez K_time
        fig.add_trace(go.Scatter(x=time_arr, y=res["K_time"], name="Rigidez (K)", line=dict(color='green')), row=3, col=1)

        # Configurações de layout
        fig.update_layout(
            title_text="Dashboard - Resultados do Baseline Linear",
            height=900,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Tempo (s)", row=3, col=1)
        fig.update_yaxes(title_text="Força (N)", row=1, col=1)
        fig.update_yaxes(title_text="Deslocamento (m)", row=2, col=1)
        fig.update_yaxes(title_text="Rigidez (N/m)", row=3, col=1)

        # Guarda o ficheiro HTML
        fig.write_html(plot_filename)
        print("Gráficos guardados com sucesso!")
        
        # Opcional: abre o gráfico automaticamente no navegador
        # fig.show()

    def plot_poincare_map(self, is_linear=False, save_dir=None, plot_filename="mapa_poincare.html", csv_filename="dados_poincare.csv", discard_periods=None, phase_offset_ratio=0.25, use_spline=True):
        """
        Gera o Mapa de Poincaré sobreposto ao Espaço de Fase Contínuo,
        dividindo em dois arquivos HTML (Pinhão+DTE e Coroa), e exportando para CSV. 
        Cores otimizadas para daltonismo (Paleta Okabe-Ito).
        
        Novo: Inclui phase_offset_ratio para sincronizar a fase com papers de referência
        e cálculo de derivadas analíticas precisas via CubicSpline.
        """

        if is_linear:
            if not hasattr(self, 'linear_backlash_results'):
                print("ERRO: Resultados lineares não encontrados. Rode 'run_linear_baseline()' primeiro.")
                return
            dados = self.linear_backlash_results
            yout_disp = self.linear_time_response.yout
            tempo = self.linear_time_response.t
            prefixo = "linear_"
        else:
            if not hasattr(self, 'backlash_results'):
                print("ERRO: Resultados não-lineares não encontrados. Rode a simulação primeiro.")
                return
            dados = self.backlash_results
            yout_disp = self.time_response.yout
            tempo = self.time
            prefixo = "nao_linear_"

        # Garante que salva na mesma pasta do script que CHAMOU a execução
        try:
            main_file = sys.modules['__main__'].__file__
            diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
        except AttributeError:
            diretorio_execucao = os.getcwd()

        if save_dir is None:
            save_dir = diretorio_execucao
        elif not os.path.isabs(save_dir):
            save_dir = os.path.join(diretorio_execucao, save_dir)
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Passo de Tempo e Derivadas Numéricas (mantidas para a linha de fundo do espaço de fase)
        dt = tempo[1] - tempo[0]
        delta = dados["delta"]
        delta_dot = np.gradient(delta, dt)
        yout_vel = np.gradient(yout_disp, dt, axis=0)

        # =========================================================================
        # --- Formulação Matemática do Período ---
        wm = self.speed_driving_gear * self.gears[0].n_teeth
        Tm = (2.0 * np.pi) / wm  
        # =========================================================================
        
        t_max = tempo[-1]
        n_periods = int(t_max / Tm)
        
        # --- Corte de Transiente Agressivo ---
        if discard_periods is None:
            discard_periods = int(n_periods * 0.8) 
            print(f"Descarte automático: ignorando os primeiros {discard_periods} de {n_periods} ciclos.")
            
        if discard_periods >= n_periods:
            print(f"AVISO: 'discard_periods' ({discard_periods}) é maior que o total de ciclos ({n_periods}).")
            discard_periods = int(n_periods * 0.5)
            print(f"Ajustando descarte para {discard_periods} ciclos.")
            
        # Determina o instante de tempo onde o regime permanente (steady state) começa
        t_discard_start = discard_periods * Tm
        idx_steady = np.searchsorted(tempo, t_discard_start)
        
        # =========================================================================
        # --- CORREÇÃO 1: Offset no tempo de amostragem (Ajuste de Fase) ---
        t_offset = phase_offset_ratio * Tm
        t_poincare = np.arange(discard_periods, n_periods) * Tm + t_offset
        # Garante que os pontos não ultrapassem o vetor de tempo simulado
        t_poincare = t_poincare[t_poincare < tempo[-1]]
        # =========================================================================

        num_dof = self.multirotor.number_dof
        idx1, idx2 = num_dof * self.gears[0].n, num_dof * self.gears[1].n

        # =========================================================================
        # --- CORREÇÃO 2: Interpolação e Derivada Analítica com Spline ---
        if use_spline:
            # Para o Erro de Transmissão Dinâmico (DTE)
            cs_delta = CubicSpline(tempo, delta)
            p_delta = cs_delta(t_poincare)
            p_delta_dot = cs_delta(t_poincare, nu=1) # nu=1 tira a derivada analítica do spline
            
            # Listas para armazenar os graus de liberdade das engrenagens
            p_disp_g1, p_vel_g1 = [], []
            p_disp_g2, p_vel_g2 = [], []
            
            for i in range(6):
                # Splines para o Pinhão
                cs_g1 = CubicSpline(tempo, yout_disp[:, idx1 + i])
                p_disp_g1.append(cs_g1(t_poincare))
                p_vel_g1.append(cs_g1(t_poincare, nu=1))
                
                # Splines para a Coroa
                cs_g2 = CubicSpline(tempo, yout_disp[:, idx2 + i])
                p_disp_g2.append(cs_g2(t_poincare))
                p_vel_g2.append(cs_g2(t_poincare, nu=1))
        else:
            # Fallback (Método antigo - não recomendado)
            p_delta = np.interp(t_poincare, tempo, delta)
            p_delta_dot = np.interp(t_poincare, tempo, delta_dot)
            
            p_disp_g1 = [np.interp(t_poincare, tempo, yout_disp[:, idx1 + i]) for i in range(6)]
            p_vel_g1  = [np.interp(t_poincare, tempo, yout_vel[:, idx1 + i]) for i in range(6)]
            p_disp_g2 = [np.interp(t_poincare, tempo, yout_disp[:, idx2 + i]) for i in range(6)]
            p_vel_g2  = [np.interp(t_poincare, tempo, yout_vel[:, idx2 + i]) for i in range(6)]
        # =========================================================================

        # Dados Contínuos para o Espaço de Fase (Apenas a partir do regime permanente)
        c_delta = delta[idx_steady:]
        c_delta_dot = delta_dot[idx_steady:]
        c_disp_g1 = [yout_disp[idx_steady:, idx1 + i] for i in range(6)]
        c_vel_g1  = [yout_vel[idx_steady:, idx1 + i] for i in range(6)]
        c_disp_g2 = [yout_disp[idx_steady:, idx2 + i] for i in range(6)]
        c_vel_g2  = [yout_vel[idx_steady:, idx2 + i] for i in range(6)]

        # --- 1. EXPORTAÇÃO CSV ---
        caminho_csv = os.path.join(save_dir, prefixo + csv_filename)
        print(f"Salvando CSV do Mapa de Poincaré em '{caminho_csv}'...")
        dados_exportacao = {
            "Tempo_s": t_poincare,
            "Delta_um": p_delta * 1e6, "dDelta_dt_mm_s": p_delta_dot * 1000,
            "Pinao_x_um": p_disp_g1[0] * 1e6, "Pinao_dx_dt_mm_s": p_vel_g1[0] * 1000,
            "Pinao_y_um": p_disp_g1[1] * 1e6, "Pinao_dy_dt_mm_s": p_vel_g1[1] * 1000,
            "Pinao_z_um": p_disp_g1[2] * 1e6, "Pinao_dz_dt_mm_s": p_vel_g1[2] * 1000,
            "Pinao_rx_mrad": p_disp_g1[3] * 1e3, "Pinao_drx_dt_mrad_s": p_vel_g1[3] * 1000,
            "Pinao_ry_mrad": p_disp_g1[4] * 1e3, "Pinao_dry_dt_mrad_s": p_vel_g1[4] * 1000,
            "Pinao_tz_mrad": p_disp_g1[5] * 1e3, "Pinao_dtz_dt_mrad_s": p_vel_g1[5] * 1000,
            "Coroa_x_um": p_disp_g2[0] * 1e6, "Coroa_dx_dt_mm_s": p_vel_g2[0] * 1000,
            "Coroa_y_um": p_disp_g2[1] * 1e6, "Coroa_dy_dt_mm_s": p_vel_g2[1] * 1000,
            "Coroa_z_um": p_disp_g2[2] * 1e6, "Coroa_dz_dt_mm_s": p_vel_g2[2] * 1000,
            "Coroa_rx_mrad": p_disp_g2[3] * 1e3, "Coroa_drx_dt_mrad_s": p_vel_g2[3] * 1000,
            "Coroa_ry_mrad": p_disp_g2[4] * 1e3, "Coroa_dry_dt_mrad_s": p_vel_g2[4] * 1000,
            "Coroa_tz_mrad": p_disp_g2[5] * 1e3, "Coroa_dtz_dt_mrad_s": p_vel_g2[5] * 1000,
        }
        pd.DataFrame(dados_exportacao).to_csv(caminho_csv, index=False, encoding='utf-8')

        # --- CORES E ESTILOS (Daltonismo Friendly - Okabe-Ito Palette) ---
        cor_espaco_fase = '#56B4E9' # Azul Claro Celeste
        cor_poincare = '#D55E00'    # Vermelhão/Laranja
        
        line_style = dict(color=cor_espaco_fase, width=1.5)
        marker_style = dict(size=6, color=cor_poincare, symbol='circle', line=dict(color='white', width=0.5), opacity=0.9)

        # --- 2. DASHBOARD 1: PINHÃO + DTE ---
        caminho_plot_pinhao = os.path.join(save_dir, prefixo + "pinhao_" + plot_filename)
        fig1 = make_subplots(
            rows=3, cols=3,
            specs=[[None, {"type": "scatter"}, None], [{"type": "scatter"}]*3, [{"type": "scatter"}]*3],
            subplot_titles=["DTE: δ vs dδ/dt", "Pinhão: x", "Pinhão: y", "Pinhão: z", "Pinhão: rx", "Pinhão: ry", "Pinhão: θz"],
            vertical_spacing=0.12
        )
        
        # PLOT DTE (Linha 1, Coluna 2)
        fig1.add_trace(go.Scatter(x=c_delta * 1e6, y=c_delta_dot * 1000, mode='lines', line=line_style, opacity=0.5, name="Espaço Fase"), row=1, col=2)
        fig1.add_trace(go.Scatter(x=dados_exportacao["Delta_um"], y=dados_exportacao["dDelta_dt_mm_s"], mode='markers', marker=marker_style, name="Poincaré"), row=1, col=2)
        
        ch_p_d = ["Pinao_x_um", "Pinao_y_um", "Pinao_z_um", "Pinao_rx_mrad", "Pinao_ry_mrad", "Pinao_tz_mrad"]
        ch_p_v = ["Pinao_dx_dt_mm_s", "Pinao_dy_dt_mm_s", "Pinao_dz_dt_mm_s", "Pinao_drx_dt_mrad_s", "Pinao_dry_dt_mrad_s", "Pinao_dtz_dt_mrad_s"]
        
        for i in range(3): 
            fig1.add_trace(go.Scatter(x=c_disp_g1[i] * 1e6, y=c_vel_g1[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=2, col=i+1)
            fig1.add_trace(go.Scatter(x=dados_exportacao[ch_p_d[i]], y=dados_exportacao[ch_p_v[i]], mode='markers', marker=marker_style, showlegend=False), row=2, col=i+1)
            
        for i in range(3, 6): 
            fig1.add_trace(go.Scatter(x=c_disp_g1[i] * 1e3, y=c_vel_g1[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=3, col=i-2)
            fig1.add_trace(go.Scatter(x=dados_exportacao[ch_p_d[i]], y=dados_exportacao[ch_p_v[i]], mode='markers', marker=marker_style, showlegend=False), row=3, col=i-2)

        fig1.update_layout(title_text=f"Poincaré + Espaço de Fase - Pinhão e DTE ({prefixo[:-1].upper()})", height=900, width=1200, template="plotly_white")
        
        fig1.update_xaxes(title_text="Deslocamento (μm)", row=1, col=2); fig1.update_yaxes(title_text="Velocidade (mm/s)", row=1, col=2)
        for c in range(1, 4):
            fig1.update_xaxes(title_text="Deslocamento (μm)", row=2, col=c); fig1.update_yaxes(title_text="Velocidade (mm/s)", row=2, col=c)
            fig1.update_xaxes(title_text="Ângulo (mrad)", row=3, col=c); fig1.update_yaxes(title_text="Vel. Angular (mrad/s)", row=3, col=c)

        fig1.write_html(caminho_plot_pinhao, include_plotlyjs="cdn")

        # --- 3. DASHBOARD 2: COROA ---
        caminho_plot_coroa = os.path.join(save_dir, prefixo + "coroa_" + plot_filename)
        fig2 = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "scatter"}]*3, [{"type": "scatter"}]*3],
            subplot_titles=["Coroa: x", "Coroa: y", "Coroa: z", "Coroa: rx", "Coroa: ry", "Coroa: θz"],
            vertical_spacing=0.15
        )
        
        ch_c_d = ["Coroa_x_um", "Coroa_y_um", "Coroa_z_um", "Coroa_rx_mrad", "Coroa_ry_mrad", "Coroa_tz_mrad"]
        ch_c_v = ["Coroa_dx_dt_mm_s", "Coroa_dy_dt_mm_s", "Coroa_dz_dt_mm_s", "Coroa_drx_dt_mrad_s", "Coroa_dry_dt_mrad_s", "Coroa_dtz_dt_mrad_s"]
        
        for i in range(3): 
            fig2.add_trace(go.Scatter(x=c_disp_g2[i] * 1e6, y=c_vel_g2[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=1, col=i+1)
            fig2.add_trace(go.Scatter(x=dados_exportacao[ch_c_d[i]], y=dados_exportacao[ch_c_v[i]], mode='markers', marker=marker_style, showlegend=False), row=1, col=i+1)
            
        for i in range(3, 6): 
            fig2.add_trace(go.Scatter(x=c_disp_g2[i] * 1e3, y=c_vel_g2[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=2, col=i-2)
            fig2.add_trace(go.Scatter(x=dados_exportacao[ch_c_d[i]], y=dados_exportacao[ch_c_v[i]], mode='markers', marker=marker_style, showlegend=False), row=2, col=i-2)

        fig2.update_layout(title_text=f"Poincaré + Espaço de Fase - Coroa ({prefixo[:-1].upper()})", height=700, width=1200, template="plotly_white")
        
        for c in range(1, 4):
            fig2.update_xaxes(title_text="Deslocamento (μm)", row=1, col=c); fig2.update_yaxes(title_text="Velocidade (mm/s)", row=1, col=c)
            fig2.update_xaxes(title_text="Ângulo (mrad)", row=2, col=c); fig2.update_yaxes(title_text="Vel. Angular (mrad/s)", row=2, col=c)

        fig2.write_html(caminho_plot_coroa, include_plotlyjs="cdn")

        print("Mapa de Poincaré e Espaço de Fase gerados com sucesso!")

    # def plot_poincare_map(self, is_linear=False, save_dir=None, plot_filename="mapa_poincare.html", csv_filename="dados_poincare.csv", discard_periods=None, use_spline=True):
    #     """
    #     Gera o Mapa de Poincaré sobreposto ao Espaço de Fase Contínuo,
    #     dividindo em dois arquivos HTML (Pinhão+DTE e Coroa), e exportando para CSV. 
    #     Cores otimizadas para daltonismo (Paleta Okabe-Ito).
    #     """
    #     from scipy.interpolate import CubicSpline

    #     if is_linear:
    #         if not hasattr(self, 'linear_backlash_results'):
    #             print("ERRO: Resultados lineares não encontrados. Rode 'run_linear_baseline()' primeiro.")
    #             return
    #         dados = self.linear_backlash_results
    #         yout_disp = self.linear_time_response.yout
    #         tempo = self.linear_time_response.t
    #         prefixo = "linear_"
    #     else:
    #         if not hasattr(self, 'backlash_results'):
    #             print("ERRO: Resultados não-lineares não encontrados. Rode a simulação primeiro.")
    #             return
    #         dados = self.backlash_results
    #         yout_disp = self.time_response.yout
    #         tempo = self.time
    #         prefixo = "nao_linear_"

    #     # Garante que salva na mesma pasta do script que CHAMOU a execução
    #     try:
    #         main_file = sys.modules['__main__'].__file__
    #         diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
    #     except AttributeError:
    #         diretorio_execucao = os.getcwd()

    #     if save_dir is None:
    #         save_dir = diretorio_execucao
    #     elif not os.path.isabs(save_dir):
    #         save_dir = os.path.join(diretorio_execucao, save_dir)
            
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     # Passo de Tempo e Derivadas Numéricas
    #     dt = tempo[1] - tempo[0]
    #     delta = dados["delta"]
    #     delta_dot = np.gradient(delta, dt)
    #     yout_vel = np.gradient(yout_disp, dt, axis=0)

    #     # =========================================================================
    #     # --- CORREÇÃO 1: Formulação Matemática do Período ---
    #     wm = self.speed_driving_gear * self.gears[0].n_teeth
    #     Tm = (2.0 * np.pi) / wm  
    #     # =========================================================================
        
    #     t_max = tempo[-1]
    #     n_periods = int(t_max / Tm)
        
    #     # --- CORREÇÃO 2: Corte de Transiente Agressivo ---
    #     if discard_periods is None:
    #         discard_periods = int(n_periods * 0.8) 
    #         print(f"Descarte automático: ignorando os primeiros {discard_periods} de {n_periods} ciclos.")
            
    #     if discard_periods >= n_periods:
    #         print(f"AVISO: 'discard_periods' ({discard_periods}) é maior que o total de ciclos ({n_periods}).")
    #         discard_periods = int(n_periods * 0.5)
    #         print(f"Ajustando descarte para {discard_periods} ciclos.")
            
    #     # Determina o instante de tempo onde o regime permanente (steady state) começa
    #     t_discard_start = discard_periods * Tm
    #     idx_steady = np.searchsorted(tempo, t_discard_start)
        
    #     # Vetor de tempo exato para o Poincaré
    #     t_poincare = np.arange(discard_periods, n_periods) * Tm
    #     t_poincare = t_poincare[t_poincare < tempo[-1]]

    #     # --- CORREÇÃO 3: Interpolação (Spline vs Linear) para o Poincaré ---
    #     def amostrar(vetor_completo):
    #         if use_spline:
    #             cs = CubicSpline(tempo, vetor_completo, extrapolate=False)
    #             return cs(t_poincare)
    #         else:
    #             return np.interp(t_poincare, tempo, vetor_completo)

    #     # Dados do Poincaré (Pontos)
    #     p_delta = amostrar(delta)
    #     p_delta_dot = amostrar(delta_dot)

    #     num_dof = self.multirotor.number_dof
    #     idx1, idx2 = num_dof * self.gears[0].n, num_dof * self.gears[1].n

    #     p_disp_g1 = [amostrar(yout_disp[:, idx1 + i]) for i in range(6)]
    #     p_vel_g1  = [amostrar(yout_vel[:, idx1 + i]) for i in range(6)]
    #     p_disp_g2 = [amostrar(yout_disp[:, idx2 + i]) for i in range(6)]
    #     p_vel_g2  = [amostrar(yout_vel[:, idx2 + i]) for i in range(6)]

    #     # Dados Contínuos para o Espaço de Fase (Apenas a partir do regime permanente)
    #     c_delta = delta[idx_steady:]
    #     c_delta_dot = delta_dot[idx_steady:]
    #     c_disp_g1 = [yout_disp[idx_steady:, idx1 + i] for i in range(6)]
    #     c_vel_g1  = [yout_vel[idx_steady:, idx1 + i] for i in range(6)]
    #     c_disp_g2 = [yout_disp[idx_steady:, idx2 + i] for i in range(6)]
    #     c_vel_g2  = [yout_vel[idx_steady:, idx2 + i] for i in range(6)]

    #     # --- 1. EXPORTAÇÃO CSV ---
    #     caminho_csv = os.path.join(save_dir, prefixo + csv_filename)
    #     print(f"Salvando CSV do Mapa de Poincaré em '{caminho_csv}'...")
    #     dados_exportacao = {
    #         "Tempo_s": t_poincare,
    #         "Delta_um": p_delta * 1e6, "dDelta_dt_mm_s": p_delta_dot * 1000,
    #         "Pinao_x_um": p_disp_g1[0] * 1e6, "Pinao_dx_dt_mm_s": p_vel_g1[0] * 1000,
    #         "Pinao_y_um": p_disp_g1[1] * 1e6, "Pinao_dy_dt_mm_s": p_vel_g1[1] * 1000,
    #         "Pinao_z_um": p_disp_g1[2] * 1e6, "Pinao_dz_dt_mm_s": p_vel_g1[2] * 1000,
    #         "Pinao_rx_mrad": p_disp_g1[3] * 1e3, "Pinao_drx_dt_mrad_s": p_vel_g1[3] * 1000,
    #         "Pinao_ry_mrad": p_disp_g1[4] * 1e3, "Pinao_dry_dt_mrad_s": p_vel_g1[4] * 1000,
    #         "Pinao_tz_mrad": p_disp_g1[5] * 1e3, "Pinao_dtz_dt_mrad_s": p_vel_g1[5] * 1000,
    #         "Coroa_x_um": p_disp_g2[0] * 1e6, "Coroa_dx_dt_mm_s": p_vel_g2[0] * 1000,
    #         "Coroa_y_um": p_disp_g2[1] * 1e6, "Coroa_dy_dt_mm_s": p_vel_g2[1] * 1000,
    #         "Coroa_z_um": p_disp_g2[2] * 1e6, "Coroa_dz_dt_mm_s": p_vel_g2[2] * 1000,
    #         "Coroa_rx_mrad": p_disp_g2[3] * 1e3, "Coroa_drx_dt_mrad_s": p_vel_g2[3] * 1000,
    #         "Coroa_ry_mrad": p_disp_g2[4] * 1e3, "Coroa_dry_dt_mrad_s": p_vel_g2[4] * 1000,
    #         "Coroa_tz_mrad": p_disp_g2[5] * 1e3, "Coroa_dtz_dt_mrad_s": p_vel_g2[5] * 1000,
    #     }
    #     pd.DataFrame(dados_exportacao).to_csv(caminho_csv, index=False, encoding='utf-8')

    #     # --- CORES E ESTILOS (Daltonismo Friendly - Okabe-Ito Palette) ---
    #     cor_espaco_fase = '#56B4E9' # Azul Claro Celeste
    #     cor_poincare = '#D55E00'    # Vermelhão/Laranja
        
    #     line_style = dict(color=cor_espaco_fase, width=1.5)
    #     marker_style = dict(size=6, color=cor_poincare, symbol='circle', line=dict(color='white', width=0.5), opacity=0.9)

    #     # --- 2. DASHBOARD 1: PINHÃO + DTE ---
    #     caminho_plot_pinhao = os.path.join(save_dir, prefixo + "pinhao_" + plot_filename)
    #     fig1 = make_subplots(
    #         rows=3, cols=3,
    #         specs=[[None, {"type": "scatter"}, None], [{"type": "scatter"}]*3, [{"type": "scatter"}]*3],
    #         subplot_titles=["DTE: δ vs dδ/dt", "Pinhão: x", "Pinhão: y", "Pinhão: z", "Pinhão: rx", "Pinhão: ry", "Pinhão: θz"],
    #         vertical_spacing=0.12
    #     )
        
    #     # PLOT DTE (Linha 1, Coluna 2)
    #     # 1. Espaço de Fase Contínuo
    #     fig1.add_trace(go.Scatter(x=c_delta * 1e6, y=c_delta_dot * 1000, mode='lines', line=line_style, opacity=0.5, name="Espaço Fase"), row=1, col=2)
    #     # 2. Mapa de Poincaré
    #     fig1.add_trace(go.Scatter(x=dados_exportacao["Delta_um"], y=dados_exportacao["dDelta_dt_mm_s"], mode='markers', marker=marker_style, name="Poincaré"), row=1, col=2)
        
    #     ch_p_d = ["Pinao_x_um", "Pinao_y_um", "Pinao_z_um", "Pinao_rx_mrad", "Pinao_ry_mrad", "Pinao_tz_mrad"]
    #     ch_p_v = ["Pinao_dx_dt_mm_s", "Pinao_dy_dt_mm_s", "Pinao_dz_dt_mm_s", "Pinao_drx_dt_mrad_s", "Pinao_dry_dt_mrad_s", "Pinao_dtz_dt_mrad_s"]
        
    #     # PLOT PINHÃO (Laços para preencher a grade)
    #     for i in range(3): 
    #         # Espaço de Fase
    #         fig1.add_trace(go.Scatter(x=c_disp_g1[i] * 1e6, y=c_vel_g1[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=2, col=i+1)
    #         # Poincaré
    #         fig1.add_trace(go.Scatter(x=dados_exportacao[ch_p_d[i]], y=dados_exportacao[ch_p_v[i]], mode='markers', marker=marker_style, showlegend=False), row=2, col=i+1)
            
    #     for i in range(3, 6): 
    #         # Espaço de Fase
    #         fig1.add_trace(go.Scatter(x=c_disp_g1[i] * 1e3, y=c_vel_g1[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=3, col=i-2)
    #         # Poincaré
    #         fig1.add_trace(go.Scatter(x=dados_exportacao[ch_p_d[i]], y=dados_exportacao[ch_p_v[i]], mode='markers', marker=marker_style, showlegend=False), row=3, col=i-2)

    #     fig1.update_layout(title_text=f"Poincaré + Espaço de Fase - Pinhão e DTE ({prefixo[:-1].upper()})", height=900, width=1200, template="plotly_white")
        
    #     fig1.update_xaxes(title_text="Deslocamento (μm)", row=1, col=2); fig1.update_yaxes(title_text="Velocidade (mm/s)", row=1, col=2)
    #     for c in range(1, 4):
    #         fig1.update_xaxes(title_text="Deslocamento (μm)", row=2, col=c); fig1.update_yaxes(title_text="Velocidade (mm/s)", row=2, col=c)
    #         fig1.update_xaxes(title_text="Ângulo (mrad)", row=3, col=c); fig1.update_yaxes(title_text="Vel. Angular (mrad/s)", row=3, col=c)

    #     fig1.write_html(caminho_plot_pinhao, include_plotlyjs="cdn")

    #     # --- 3. DASHBOARD 2: COROA ---
    #     caminho_plot_coroa = os.path.join(save_dir, prefixo + "coroa_" + plot_filename)
    #     fig2 = make_subplots(
    #         rows=2, cols=3,
    #         specs=[[{"type": "scatter"}]*3, [{"type": "scatter"}]*3],
    #         subplot_titles=["Coroa: x", "Coroa: y", "Coroa: z", "Coroa: rx", "Coroa: ry", "Coroa: θz"],
    #         vertical_spacing=0.15
    #     )
        
    #     ch_c_d = ["Coroa_x_um", "Coroa_y_um", "Coroa_z_um", "Coroa_rx_mrad", "Coroa_ry_mrad", "Coroa_tz_mrad"]
    #     ch_c_v = ["Coroa_dx_dt_mm_s", "Coroa_dy_dt_mm_s", "Coroa_dz_dt_mm_s", "Coroa_drx_dt_mrad_s", "Coroa_dry_dt_mrad_s", "Coroa_dtz_dt_mrad_s"]
        
    #     for i in range(3): 
    #         # Espaço de Fase
    #         fig2.add_trace(go.Scatter(x=c_disp_g2[i] * 1e6, y=c_vel_g2[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=1, col=i+1)
    #         # Poincaré
    #         fig2.add_trace(go.Scatter(x=dados_exportacao[ch_c_d[i]], y=dados_exportacao[ch_c_v[i]], mode='markers', marker=marker_style, showlegend=False), row=1, col=i+1)
            
    #     for i in range(3, 6): 
    #         # Espaço de Fase
    #         fig2.add_trace(go.Scatter(x=c_disp_g2[i] * 1e3, y=c_vel_g2[i] * 1000, mode='lines', line=line_style, opacity=0.5, showlegend=False), row=2, col=i-2)
    #         # Poincaré
    #         fig2.add_trace(go.Scatter(x=dados_exportacao[ch_c_d[i]], y=dados_exportacao[ch_c_v[i]], mode='markers', marker=marker_style, showlegend=False), row=2, col=i-2)

    #     fig2.update_layout(title_text=f"Poincaré + Espaço de Fase - Coroa ({prefixo[:-1].upper()})", height=700, width=1200, template="plotly_white")
        
    #     for c in range(1, 4):
    #         fig2.update_xaxes(title_text="Deslocamento (μm)", row=1, col=c); fig2.update_yaxes(title_text="Velocidade (mm/s)", row=1, col=c)
    #         fig2.update_xaxes(title_text="Ângulo (mrad)", row=2, col=c); fig2.update_yaxes(title_text="Vel. Angular (mrad/s)", row=2, col=c)

    #     fig2.write_html(caminho_plot_coroa, include_plotlyjs="cdn")

    #     print("Mapa de Poincaré e Espaço de Fase gerados com sucesso!")