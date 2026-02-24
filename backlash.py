import numpy as np
from re import search
from copy import deepcopy as copy
from collections.abc import Iterable
from scipy.integrate import cumulative_trapezoid as integrate

import sys
# sys.path.append('C:\\Users\\Murillo\\OneDrive - Universidade Federal de Uberlândia\\Área de Trabalho\\Mestrado\\ENGRENAMENTO\\Implementacao\\ross')
# sys.path.append('C:\\Users\\Murillo\\OneDrive - Universidade Federal de Uberlândia\\Área de Trabalho\\Mestrado\\ENGRENAMENTO\\Implementacao\\ross_dev_backlash\\ross')

#home
sys.path.append('C:\\Users\\M\\Documents\\Mestrado\\ENGRENAMENTO\\ross')

import ross as rs
from ross.gear_element import Mesh
from ross.rotor_assembly import Rotor
from ross.units import Q_, check_units
from scipy.interpolate import interp1d

__all__ = ["Backlash"]

class Backlash:
    def __init__(self, 
                 multirotor,
                 speed_driving_gear,
                 b0=0,
                 error_amp = 0,
                 gear_mesh_stiffness = None,
                 num_points_cicle=1000, 
                 n_cicles=2, 
                 cut_cicles=1,
                 use_multirotor_coupling_stiffness = False,
                 RHS = True,
                 ):
        
        self.multirotor = copy(multirotor)
        self.speed_driving_gear = speed_driving_gear
        self.b0 = b0,
        self.error_amp = error_amp,
        self.n_cicles = n_cicles
        self.cut_cicles = cut_cicles
        self.gear_mesh_stiffness = gear_mesh_stiffness
        self.RHS = RHS

        self.gears = np.array([e for e in self.multirotor.disk_elements if isinstance(e, rs.GearElement)])


        mmc_teeth = np.lcm(self.gears[0].n_teeth, self.gears[1].n_teeth)

        # Periodo completo do engreanamento
        n_cicles_sim = n_cicles + cut_cicles

        max_time = n_cicles_sim * 2*np.pi*self.gears[0].n_teeth / (speed_driving_gear * self.gears[0].n_teeth)

        self.num_points_total = num_points_cicle*n_cicles_sim
        
        self.time = np.linspace(0,max_time, self.num_points_total)

        T_cycle = max_time/n_cicles_sim
        dt = self.time[1] - self.time[0]
        n_cut = num_points_cicle*cut_cicles
        # n_cut = int(T_cycle / dt)*cut_cicles

        self.n_cut = n_cut

        
        
        self.init_backlash_results()

        wm = speed_driving_gear*self.gears[0].n_teeth
        self.error = self.error_amp*np.sin(wm*self.time)


        if use_multirotor_coupling_stiffness is False:
            self.multirotor.gear_mesh_stiffness = 0
            self.multirotor.update_mesh_stiffness = False



    def generate_speed_ramp(self, ramp_fraction=0.0):
        """
        Gera rampa de velocidade angular.

        ramp_fraction : fração do tempo total usada para rampa

        Retorna:
            escalar se ramp_fraction == 0
            array caso contrário
        """

        t = np.asarray(self.time)
        omega_max = self.speed_driving_gear

        # 🔹 Se não houver rampa, retorna escalar
        if ramp_fraction == 0:
            return omega_max

        T_total = t[-1]
        T_ramp = ramp_fraction * T_total

        speed_ramp = np.zeros_like(t)

        ramp_mask = t <= T_ramp
        speed_ramp[ramp_mask] = omega_max * (t[ramp_mask] / T_ramp)
        speed_ramp[~ramp_mask] = omega_max

        return speed_ramp

    
    
    def run_dynamic_backlash(self, unb_node,
                             unb_magnitude,
                             unb_phase,
                             add_force=None,
                             **kwargs
                             ):
        
        speed_ramp = self.generate_speed_ramp(ramp_fraction=kwargs.get('ramp_fraction', 0.0))


        #  Calculo das forças de desbalanceamento residual
        self.unb_force, _, _, _ = self.multirotor.unbalance_force_over_time(
            unb_node, unb_magnitude, unb_phase, speed_ramp, self.time, return_all=True)
        
        F = self.unb_force.T

        if add_force is not None:
            F += self.unb_force.T + add_force

        
        if self.RHS is True:

            print("RHS == True")

            results = self.multirotor.run_time_response(speed=speed_ramp,
                                                        F = F, 
                                                        t = self.time, 
                                                        method="newmark",
                                                        add_to_RHS = self.compute_backlash_force,
                                                        **kwargs
                                                        )
        
        

        if self.RHS is False:

            print("RHS == False")

            orbit, _ = self._determine_individual_orbits(speed_ramp, F, self.time, integration_method=kwargs.get('integration_method', "default"))

            self.calc_backlash_via_orbit(self.b0, orbit)
            
            F_total = F + self.backlash_total_force

            results = self.multirotor.run_time_response(speed=self.speed_driving_gear,
                                                        F = F_total, 
                                                        t = self.time, 
                                                        # method="newmark",
                                                        # add_to_RHS = self.compute_backlash_force,
                                                        )




        self.cut_backlash_results()
        
        results.yout = results.yout[self.n_cut:, :]
        results.t = results.t[self.n_cut:]
        results.t -= results.t[0]
        self.time = self.time[self.n_cut:]

        self.time_response = results
                                                       

        return results

    def compute_backlash_force(self, step, time_step, disp_resp, velc_resp, accl_resp):

        number_of_dof = self.multirotor.number_dof
        
        gear_nodes = np.array([e.n for e in self.gears])

        x1_dof = number_of_dof*gear_nodes[0] + 0
        y1_dof = number_of_dof*gear_nodes[0] + 1
        tz1_dof = number_of_dof*gear_nodes[0] + 5

        x2_dof = number_of_dof*gear_nodes[1] + 0
        y2_dof = number_of_dof*gear_nodes[1] + 1
        tz2_dof = number_of_dof*gear_nodes[1] + 5

         # Órbita 1
        x1 = disp_resp[x1_dof]
        y1 = disp_resp[y1_dof]
        theta1 = disp_resp[tz1_dof]

        # x1_c1 = x1 - np.mean(x1)
        # y1_c1 = y1 - np.mean(y1)

        # theta1 = np.arctan2(y1_c1, x1_c1)
        # theta1 = np.unwrap(theta1)


        # Órbita 2
        x2 = disp_resp[x2_dof]
        y2 = disp_resp[y2_dof]
        theta2 = disp_resp[tz2_dof]

        # x2_c1 = x2 - np.mean(x2)
        # y2_c1 = y2 - np.mean(y2)

        # theta2 = np.arctan2(y2_c1, x2_c1)
        # theta2 = np.unwrap(theta2)


        # =========================
        # CENTROS (referencial local)
        # =========================

        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2

        x1_c = 0
        y1_c = 0

        x2_c = x1_c + d0*np.cos(self.multirotor.orientation_angle)
        y2_c = y1_c + d0*np.sin(self.multirotor.orientation_angle)

        x2 += d0*np.cos(self.multirotor.orientation_angle)
        y2 += d0*np.sin(self.multirotor.orientation_angle)

        # Deslocamentos relativos ao centro
        x1r = x1 - x1_c
        y1r = y1 - y1_c

        x2r = x2 - x2_c
        y2r = y2 - y2_c

        self.backlash_results["x1"][step] = x1
        self.backlash_results["y1"][step] = y1
        self.backlash_results["x2"][step] = x2
        self.backlash_results["y2"][step] = y2
        self.backlash_results["t1"][step] = theta1
        self.backlash_results["t2"][step] = theta2


        # =========================
        # PARÂMETROS GEOMÉTRICOS
        # =========================

        R1 = self.gears[0].base_radius
        R2 = self.gears[1].base_radius


        # =========================
        # GEOMETRIA INSTANTÂNEA
        # =========================

        # Distância entre centros
        d = np.sqrt((x2r - x1r + d0)**2 + (y2r - y1r)**2)

        self.backlash_results["d"][step] = d

        # Ângulo de posição (use arctan2!)
        beta = self.multirotor.orientation_angle + np.arctan2((y2r - y1r), (x2r - x1r + d0))

        self.backlash_results["beta"][step] = beta
        
        # Ângulo de pressão instantâneo
        alfa = np.arccos((R1 + R2) / d)

        self.backlash_results["alfa"][step] = alfa


        # =========================
        # DEFORMAÇÃO DE ENGRENAMENTO
        # =========================

        delta = ((x1r - x2r)*np.sin(alfa - beta) + (y1r - y2r)*np.cos(alfa - beta) + R1*theta1 - R2*theta2 - self.error[step] )

        self.backlash_results["delta"][step] = delta
        
        def inv(angle):
            return np.tan(angle)-angle

        # dynamic backlash
        alfa0 = self.gears[0].pr_angle #pressure angle original
        b = (R1+R2)*(inv(alfa)-inv(alfa0))
        bt = self.b0 + b

        self.backlash_results["bt"][step] = bt

        f = 0

        if delta > bt:
            f = delta - bt
        elif abs(delta) <= bt:
            f = 0.0
        elif delta < -bt:
            f = delta + bt

        self.backlash_results["f"][step] = f
        
        Ra1 = self.gears[0].radii_dict["addendum"]
        Ra2 = self.gears[1].radii_dict["addendum"]
        
        contact_ratio = (np.sqrt(Ra1**2-R1**2)+np.sqrt(Ra2**2-R2**2)-d*np.sin(alfa))/(np.pi*self.gears[0].module)

        self.backlash_results["contact_ratio"][step] = contact_ratio
 
        K_time_step = self._get_K_time_step(step, contact_ratio)

        self.backlash_results["K_time"][step] = K_time_step

        Fm = K_time_step * f

        self.backlash_results["Fm"][step] = Fm

        backlash_force = np.zeros(self.multirotor.ndof)

        backlash_force[x1_dof] = -Fm*np.sin(alfa0)
        backlash_force[y1_dof] = -Fm*np.cos(alfa0)
        backlash_force[tz1_dof] = -Fm

        backlash_force[x2_dof] = Fm*np.sin(alfa0)
        backlash_force[y2_dof] = Fm*np.cos(alfa0)
        backlash_force[tz2_dof] = Fm   

        self.backlash_total_force[step, :] = backlash_force
        
        if self.RHS is False:
            backlash_force = np.zeros(self.multirotor.ndof) #issaqui pra nao incluir o RHS
        
        return backlash_force
    
    def _get_K_time_step(self, step, contact_ratio):
        
        if self.gear_mesh_stiffness is None:

            # theta_range, stiffness_range = self.multirotor.mesh.get_stiffness_for_mesh_period(contact_ratio = contact_ratio)

            # K_time = self.K_theta_to_time(
            #     theta_k=theta_range,
            #     K_theta=stiffness_range,
            #     t=self.time,
            #     speed=self.speed_driving_gear
            # )

            angular_position = self.speed_driving_gear * self.time[step]
            K_time_step = self.multirotor.mesh.get_variable_stiffness(angular_position=angular_position, contact_ratio=contact_ratio)

        else:
            # K_time = self.gear_mesh_stiffness*np.ones_like(self.time)

            K_time_step = self.gear_mesh_stiffness
        
        return K_time_step
    
    def K_theta_to_time(self, theta_k, K_theta, t, speed):

        theta_k = np.asarray(theta_k)
        K_theta = np.asarray(K_theta)
        t = np.asarray(t)

        # Garantir ordenação
        sort_idx = np.argsort(theta_k)
        theta_k = theta_k[sort_idx]
        K_theta = K_theta[sort_idx]

        # Período físico baseado no domínio angular
        theta_period = theta_k[-1] - theta_k[0]

        # Fechar ciclo explicitamente
        theta_extended = np.append(theta_k, theta_k[0] + theta_period)
        K_extended = np.append(K_theta, K_theta[0])

        interp_func = interp1d(
            theta_extended,
            K_extended,
            kind='linear',
            bounds_error=True
        )

        # Transformação física
        theta_t = speed * t

        # Redução modular (ESSENCIAL)
        theta_t_mod = np.mod(theta_t - theta_k[0], theta_period) + theta_k[0]

        K_time = interp_func(theta_t_mod)

        return K_time
    
    def _determine_individual_orbits(self, speed, force, time, integration_method="default"):

        
        # speed = np.asarray(speed)

        # if speed.ndim > 0 and np.allclose(speed, speed[0]):
        #     speed = float(speed[0])

        # print(f"type speed {type(speed)}")
        
        # Time Response
        time_response = self.multirotor.run_time_response(speed=speed, F = force, t = time, method = integration_method)

        gear_nodes = np.array([e.n for e in self.gears])
        
        # Obtenção dos veotres de orbitas individuais
        ndof = self.multirotor.number_dof

        n_nodes = len(gear_nodes)
        n_time = time_response.yout.shape[0]

        orbit = np.zeros((n_nodes, n_time, 3))

        for i, node in enumerate(gear_nodes):

            dofx = ndof * node + 0
            dofy = ndof * node + 1
            dofrz = ndof * node + 5


            x = time_response.yout[:, dofx]
            y = time_response.yout[:, dofy]
            rz = time_response.yout[:, dofrz]

            orbit[i, :, 0] = x   # x(t) do nó i
            orbit[i, :, 1] = y   # y(t) do nó i
            orbit[i, :, 2] = rz   # theta(t) do nó i


        # correção do centro da orbita do rotor 2
        center_distance = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter)/2

        orbit[1,:,0] += center_distance*np.cos(self.multirotor.orientation_angle)
        orbit[1,:,1] += center_distance*np.sin(self.multirotor.orientation_angle)

        # orbit[0,:,0] = Q_(orbit[0,:,0], "m").to(displacement_units).m
        # orbit[0,:,1] = Q_(orbit[0,:,1], "m").to(displacement_units).m
        # orbit[0,:,2] = Q_(orbit[0,:,2], "rad").m
        # orbit[1,:,0] = Q_(orbit[1,:,0], "m").to(displacement_units).m
        # orbit[1,:,1] = Q_(orbit[1,:,1], "m").to(displacement_units).m
        # orbit[1,:,2] = Q_(orbit[1,:,2], "rad").m


        return orbit, time_response
    
    def calc_backlash_via_orbit(self, b0, orbit):

        number_of_dof = self.multirotor.number_dof
        
        gear_nodes = np.array([e.n for e in self.gears])

        x1_dof = number_of_dof*gear_nodes[0] + 0
        y1_dof = number_of_dof*gear_nodes[0] + 1
        tz1_dof = number_of_dof*gear_nodes[0] + 5

        x2_dof = number_of_dof*gear_nodes[1] + 0
        y2_dof = number_of_dof*gear_nodes[1] + 1
        tz2_dof = number_of_dof*gear_nodes[1] + 5

        # Órbita 1
        x1 = orbit[0, :, 0]
        y1 = orbit[0, :, 1]
        theta1 = orbit[0, :, 2]

        # x1_c1 = x1 - np.mean(x1)
        # y1_c1 = y1 - np.mean(y1)

        # theta1 = np.arctan2(y1_c1, x1_c1)
        # theta1 = np.unwrap(theta1)


        # Órbita 2
        x2 = orbit[1, :, 0]
        y2 = orbit[1, :, 1]
        theta2 = orbit[1, :, 2]

        # x2_c1 = x2 - np.mean(x2)
        # y2_c1 = y2 - np.mean(y2)

        # theta2 = np.arctan2(y2_c1, x2_c1)
        # theta2 = np.unwrap(theta2)


        # =========================
        # CENTROS (referencial local)
        # =========================

        x1_c = np.mean(x1)
        y1_c = np.mean(y1)

        x2_c = np.mean(x2)
        y2_c = np.mean(y2)


        # Deslocamentos relativos ao centro
        x1r = x1 - x1_c
        y1r = y1 - y1_c

        x2r = x2 - x2_c
        y2r = y2 - y2_c

        self.backlash_results["x1"] = x1
        self.backlash_results["y1"] = y1
        self.backlash_results["x2"] = x2
        self.backlash_results["y2"] = y2
        self.backlash_results["t1"] = theta1
        self.backlash_results["t2"] = theta2



        # =========================
        # PARÂMETROS GEOMÉTRICOS
        # =========================

        R1 = self.gears[0].base_radius
        R2 = self.gears[1].base_radius

        d0 = (self.gears[0].pitch_diameter + self.gears[1].pitch_diameter) / 2


        # =========================
        # GEOMETRIA INSTANTÂNEA
        # =========================

        # Distância entre centros
        d = np.sqrt((x2r - x1r + d0)**2 + (y2r - y1r)**2)

        self.backlash_results["d"] = d

        # Ângulo de posição (use arctan2!)
        beta = self.multirotor.orientation_angle + np.arctan2((y2r - y1r), (x2r - x1r + d0))

        self.backlash_results["beta"] = beta

        # Ângulo de pressão instantâneo
        alfa = np.arccos((R1 + R2) / d)

        self.backlash_results["alfa"] = alfa


        # =========================
        # DEFORMAÇÃO DE ENGRENAMENTO
        # =========================

        delta = ((x1r - x2r)*np.sin(alfa - beta) + (y1r - y2r)*np.cos(alfa - beta) + R1*theta1 - R2*theta2 - self.error)

        self.backlash_results["delta"] = delta

        def inv(angle):
            return np.tan(angle)-angle

        # dynamic backlash
        alfa0 = self.gears[0].pr_angle #pressure angle original
        b = (R1+R2)*(inv(alfa)-inv(alfa0))
        bt = b0 + b

        self.backlash_results["bt"] = bt

        f = []  # lista
        for i in range(len(delta)):
            if delta[i] > bt[i]:
                f.append(delta[i] - bt[i])
            elif abs(delta[i]) <= bt[i]:
                f.append(0.0)
            elif delta[i] < -bt[i]:
                f.append(delta[i] + bt[i])

        f = np.array(f)


        self.backlash_results["f"] = f
        
        Ra1 = self.gears[0].radii_dict["addendum"]
        Ra2 = self.gears[1].radii_dict["addendum"]
        
        contact_ratio = (np.sqrt(Ra1**2-R1**2)+np.sqrt(Ra2**2-R2**2)-d*np.sin(alfa))/(np.pi*self.gears[0].module)

        self.backlash_results["contact_ratio"] = contact_ratio

        K_time_step = np.zeros_like(self.time)

        Fm = np.zeros_like(self.time)

        for step in range(len(self.time)):

            k = self._get_K_time_step(step, contact_ratio[step])
 
            K_time_step[step] = k

            self.backlash_results["K_time"][step] = k

            Fm[step] = K_time_step[step] * f[step]

            self.backlash_results["Fm"][step] = Fm[step]

            self.backlash_total_force[step, x1_dof] = -Fm[step]*np.sin(alfa0)
            self.backlash_total_force[step, y1_dof] = -Fm[step]*np.cos(alfa0)
            self.backlash_total_force[step, tz1_dof] = -Fm[step]

            self.backlash_total_force[step, x2_dof] = Fm[step]*np.sin(alfa0)
            self.backlash_total_force[step, y2_dof] = Fm[step]*np.cos(alfa0)
            self.backlash_total_force[step, tz2_dof] = Fm[step]   



    
    def init_backlash_results(self):

        self.backlash_total_force = np.zeros((len(self.time), self.multirotor.ndof))

        d = np.zeros(self.num_points_total)
        beta = np.zeros(self.num_points_total)
        alfa = np.zeros(self.num_points_total)
        delta = np.zeros(self.num_points_total)
        bt = np.zeros(self.num_points_total)
        f = np.zeros(self.num_points_total)
        Fm = np.zeros(self.num_points_total)
        contact_ratio = np.zeros(self.num_points_total)
        K_time = np.zeros(self.num_points_total)

        x1 = np.zeros(self.num_points_total)
        y1 = np.zeros(self.num_points_total)
        x2 = np.zeros(self.num_points_total)
        y2 = np.zeros(self.num_points_total)
        t1 = np.zeros(self.num_points_total)
        t2 = np.zeros(self.num_points_total)

        self.backlash_results = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "t1": t1,
            "t2": t2,
            "d": d,        # distância entre centros
            "beta": beta,                # ângulo de posição
            "alfa": alfa,               # ângulo de pressão instantâneo
            "delta": delta,              # deformação de engrenamento
            "bt": bt,        # backlash dinâmico total
            "f": f,           # deslocamento efetivo após backlash
            "Fm": Fm,          # Força
            "contact_ratio": contact_ratio,          # razao de contato
            "K_time": K_time,
        }
    
    def cut_backlash_results(self):
        self.backlash_results["d"] = self.backlash_results["d"][self.n_cut:]
        self.backlash_results["beta"] = self.backlash_results["beta"][self.n_cut:]
        self.backlash_results["alfa"] = self.backlash_results["alfa"][self.n_cut:]
        self.backlash_results["delta"] = self.backlash_results["delta"][self.n_cut:]
        self.backlash_results["bt"] = self.backlash_results["bt"][self.n_cut:]
        self.backlash_results["f"] = self.backlash_results["f"][self.n_cut:]
        self.backlash_results["Fm"] = self.backlash_results["Fm"][self.n_cut:]
        self.backlash_results["contact_ratio"] = self.backlash_results["contact_ratio"][self.n_cut:]
        self.backlash_results["K_time"] = self.backlash_results["K_time"][self.n_cut:]
        self.backlash_results["x1"] = self.backlash_results["x1"][self.n_cut:]
        self.backlash_results["y1"] = self.backlash_results["y1"][self.n_cut:]
        self.backlash_results["x2"] = self.backlash_results["x2"][self.n_cut:]
        self.backlash_results["y2"] = self.backlash_results["y2"][self.n_cut:]
        self.backlash_results["t1"] = self.backlash_results["t1"][self.n_cut:]
        self.backlash_results["t2"] = self.backlash_results["t2"][self.n_cut:]

        self.backlash_total_force = self.backlash_total_force[self.n_cut:, :]

        unbalance_force = self.unb_force.T
        self.unbalance_force = unbalance_force[self.n_cut:, :]
    
