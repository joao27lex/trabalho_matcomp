import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Corpo_Celeste import Corpo_Celeste
from RK4Solver import RK4Solver


class Universo:
    def __init__(self, duracao_padrao=4e6, ganho_duracao=100, intervalo_animacao=100):
        self.G = 6.67430e-11
        self.duracao = duracao_padrao * ganho_duracao
        self.intervalo_animacao = intervalo_animacao
        # se deixar no criar_plot como equal, datalim ele só ignora o limite menor. mas se usar o razao, box, ele vai considerar o limite menor tambem (porém o raio fica errado)
        # self.limite_x = 2e11
        # self.limite_y = 1e10
        self.limite_x = 1e12
        self.limite_y = 1e12
        self.criar_corpos_celestes()
        self.y0 = self.get_y0()

    def criar_corpos_celestes(self):
        # === Inicialização das listas ===
        self.corpos_celestes = []
        self.corpos_names = []
        self.corpos_massas = []

        # === Sol (corpo fixo no centro) ===
        self.Sol = Corpo_Celeste(
            massa=2e30,
            raio=6.957e8,
            color="yellow",
            name="Sol",
            pos_x=0,
            pos_y=0.0,
            abg=False,
        )
        self.fixed_body_name = "Sol"
        self.fixed_body_index = 0
        self.corpos_celestes.append(self.Sol)
        self.corpos_names.append(self.Sol.name)
        self.corpos_massas.append(self.Sol.massa)

        # === Marte ===
        raio_orbita_marte = 2.279e11
        angulo_rad = np.deg2rad(45)
        x_marte = raio_orbita_marte * np.cos(angulo_rad)
        y_marte = raio_orbita_marte * np.sin(angulo_rad)
        v_marte_modulo = np.sqrt(self.G * self.Sol.massa / raio_orbita_marte)
        vx_marte = -v_marte_modulo * np.sin(angulo_rad)
        vy_marte = v_marte_modulo * np.cos(angulo_rad)
        self.Marte = Corpo_Celeste(
            massa=6.4e23,
            raio=3.389e6,
            color="red",
            name="Marte",
            pos_x=x_marte,
            pos_y=y_marte,
            vel_x=vx_marte,
            vel_y=vy_marte,
        )
        self.corpos_celestes.append(self.Marte)
        self.corpos_names.append(self.Marte.name)
        self.corpos_massas.append(self.Marte.massa)

        # === Terra ===
        self.Terra = Corpo_Celeste(
            massa=5.972e24,
            raio=6.371e6,
            color="blue",
            name="Terra",
            pos_x=1.496e11,
            pos_y=0.0,
        )
        self.Terra.vel_y = np.sqrt(self.G * self.Sol.massa / self.Terra.pos_x)
        self.corpos_celestes.append(self.Terra)
        self.corpos_names.append(self.Terra.name)
        self.corpos_massas.append(self.Terra.massa)

        # === Lua (em órbita da Terra) ===
        self.Lua = Corpo_Celeste(
            massa=7.346e22,
            raio=1.737e6,
            color="white",
            name="Lua",
            pos_x=self.Terra.pos_x,
            pos_y=-3.84e8,
        )
        distancia_lua_terra = abs(self.Lua.pos_y - self.Terra.pos_y)
        v_rel_lua = np.sqrt(self.G * self.Terra.massa / distancia_lua_terra)
        v_terra = np.array([self.Terra.vel_x, self.Terra.vel_y])
        vetor_terra_lua = np.array(
            [self.Lua.pos_x - self.Terra.pos_x, self.Lua.pos_y - self.Terra.pos_y]
        )
        norma = np.linalg.norm(vetor_terra_lua)
        direcao_tangente = np.array([-vetor_terra_lua[1], vetor_terra_lua[0]]) / norma
        v_lua_rel = direcao_tangente * v_rel_lua
        v_lua_total = v_terra + v_lua_rel
        self.Lua.vel_x = v_lua_total[0]
        self.Lua.vel_y = v_lua_total[1]
        self.corpos_celestes.append(self.Lua)
        self.corpos_names.append(self.Lua.name)
        self.corpos_massas.append(self.Lua.massa)

        # === Asteroide (rumo à Terra) ===
        terra_pos = np.array([1.496e11, 0])
        asteroide_pos = np.array([5e11, 5e11])
        direcao = terra_pos - asteroide_pos
        direcao_unitaria = direcao / np.linalg.norm(direcao)
        vel_impacto = 2e5
        velocidade_asteroide = direcao_unitaria * vel_impacto
        self.Asteroide = Corpo_Celeste(
            massa=9e10,
            raio=1e5,
            color="black",
            name="Asteroide",
            pos_x=asteroide_pos[0],
            pos_y=asteroide_pos[1],
            vel_x=velocidade_asteroide[0],
            vel_y=velocidade_asteroide[1],
        )
        self.corpos_celestes.append(self.Asteroide)
        self.corpos_names.append(self.Asteroide.name)
        self.corpos_massas.append(self.Asteroide.massa)

        # === Projétil (lançado da Terra, velocidade inicial nula por padrão) ===
        self.Projetil = Corpo_Celeste(
            massa=1e3,  # massa pequena
            raio=1e5,
            color="green",
            name="Projetil",
            pos_x=self.Terra.pos_x,
            pos_y=self.Terra.pos_y,
            vel_x=0.0,
            vel_y=0.0,
            abg=True,  # afetado pela gravidade
            awg=False,  # não afeta os outros
        )
        self.corpos_celestes.append(self.Projetil)
        self.corpos_names.append(self.Projetil.name)
        self.corpos_massas.append(self.Projetil.massa)

    # y0 = vetor inicial
    def get_y0(self):
        y0 = []
        for index, cc in enumerate(self.corpos_celestes):
            if index != self.fixed_body_index:
                estado = cc.return_estado()
                y0.extend(estado)
                # y0.extend([cc.pos_x],[cc.pos_y],[cc.vel_x],[cc.vel_y])
        return np.array(y0)

    # equacoes_movimento_setup
    def equacoes_movimento_setup(self, y):
        all_positions = []
        all_velocities = []

        # como temos xa,ya,vxa,vya,xb,yb,vxb,vyb,xc,yc,vxc,vyc
        # precisamos de um ponteiro = ptr pra ir em cada valor de forma organizada
        ptr = 0
        for i, cc in enumerate(self.corpos_celestes):
            if i == self.fixed_body_index:
                all_positions.append(np.array(cc.return_pos()))
                all_velocities.append(np.array(cc.return_vel()))
            else:
                all_positions.append(np.array([y[ptr], y[ptr + 1]]))
                all_velocities.append(np.array([y[ptr + 2], y[ptr + 3]]))
                ptr += 4

        dydt = np.zeros_like(y)
        return all_positions, all_velocities, dydt

    # tem que deixar t aqui pelo solve_ivp
    def equacoes_movimento(self, t, y):
        all_positions, all_velocities, dydt = self.equacoes_movimento_setup(y)

        # dessa vez o ptr eh pro vx,vy,ax,ay,...
        ptr = 0
        for i in range(len(self.corpos_celestes)):
            # se for o index do sol ele passa pra prox iteracao
            if i == self.fixed_body_index:
                continue

            acc_i = np.array([0.0, 0.0])

            for j in range(len(self.corpos_celestes)):
                if i == j:
                    continue
                r = all_positions[j] - all_positions[i]
                r2 = np.dot(r, r)
                if r2 < 1e-10:
                    continue
                r3 = r2**1.5
                # aqui eu testei negativo e positivo e ele sem '-' dá
                acc_i += self.G * self.corpos_massas[j] * r / r3

            dydt[ptr] = all_velocities[i][0]
            dydt[ptr + 1] = all_velocities[i][1]
            dydt[ptr + 2] = acc_i[0]
            dydt[ptr + 3] = acc_i[1]

            ptr += 4

        return dydt

    def get_current_state(self, y_in_t):
        current_states = []
        ptr = 0
        for i, cc in enumerate(self.corpos_celestes):
            if i == self.fixed_body_index:
                current_states.append(
                    {
                        "pos_x": cc.pos_x,
                        "pos_y": cc.pos_y,
                        "vel_x": cc.vel_x,
                        "vel_y": cc.vel_y,
                    }
                )
            else:
                current_states.append(
                    {
                        "pos_x": y_in_t[ptr],
                        "pos_y": y_in_t[ptr + 1],
                        "vel_x": y_in_t[ptr + 2],
                        "vel_y": y_in_t[ptr + 3],
                    }
                )
                ptr += 4
        return current_states

    def simular(self):
        for cc in self.corpos_celestes:
            cc.trace = []
        t_eval = np.linspace(0, self.duracao, 15000)
        rtol = 1e-6
        atol = 1e-9
        solver = RK4Solver(
            self.equacoes_movimento, (0, self.duracao), self.y0, t_eval=t_eval
        )
        solucao = solver.solve()

        solucao_array = solucao.y.T

        for step in range(len(t_eval)):
            y_in_t = solucao_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.corpos_celestes[i].pos_x = state["pos_x"]
                self.corpos_celestes[i].pos_y = state["pos_y"]
                self.corpos_celestes[i].vel_x = state["vel_x"]
                self.corpos_celestes[i].vel_y = state["vel_y"]
                self.corpos_celestes[i].trace.append((state["pos_x"], state["pos_y"]))

        return solucao_array

    def criar_plot(self, ax):
        ax.set_xlim(-self.limite_x, self.limite_x)
        ax.set_ylim(-self.limite_y, self.limite_y)
        ax.set_aspect("equal", adjustable="datalim")
        # esse de baixo deixa desproporcional e as esferas viram (), talvez desse pra usar elipse ao inves de circle pra ajustar com a razao tambem, não tenho certeza
        # razao = self.limite_x / self.limite_y
        # ax.set_aspect(razao, adjustable='box')

        ax.set_facecolor("gray")
        # sei que nao eh universo
        ax.set_title("Universo teste")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    def animar(
        self, solucao_array, df_trajetoria_projetil=None, df_sonda_limitada=None
    ):
        if solucao_array is None:
            print("erro em Universo, Animar")
            return
        fig, ax = plt.subplots(figsize=(8, 8))
        self.criar_plot(ax)

        # === Projétil ===
        if df_trajetoria_projetil is not None:
            print("[DEBUG] Projétil carregado com sucesso")
            (linha_proj,) = ax.plot([], [], "--", color="green", lw=2, label="Projétil")
            (ponto_proj,) = ax.plot([], [], "o", color="green", markersize=6)
        else:
            print("[ERRO] df_trajetoria_projetil é None dentro da função animar!")
            linha_proj, ponto_proj = None, None

        # === Sonda com atuadores limitados ===
        if df_sonda_limitada is not None:
            (linha_sonda,) = ax.plot(
                [], [], "--", color="orange", lw=2, label="Sonda Limitada"
            )
            (ponto_sonda,) = ax.plot([], [], "o", color="orange", markersize=6)
        else:
            linha_sonda, ponto_sonda = None, None

        linhas = []
        pontos = []
        for cc in self.corpos_celestes:
            (linha,) = ax.plot(
                [], [], "-", lw=2, label=f"{cc.name} Trajetoria", color=cc.color
            )
            if cc.name == "Lua":
                (ponto,) = ax.plot(
                    [],
                    [],
                    "o",
                    markersize=6,
                    label=f"{cc.name}",
                    markerfacecolor=cc.color,
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                )
            else:
                (ponto,) = ax.plot(
                    [],
                    [],
                    "o",
                    markersize=12,
                    label=f"{cc.name}",
                    markerfacecolor=cc.color,
                )
            linhas.append(linha)
            pontos.append(ponto)

        def update(frame):
            current_states = self.get_current_state(solucao_array[frame])
            for i, cc in enumerate(self.corpos_celestes):
                cc.pos_x = current_states[i]["pos_x"]
                cc.pos_y = current_states[i]["pos_y"]
                cc.vel_x = current_states[i]["vel_x"]
                cc.vel_y = current_states[i]["vel_y"]
                linhas[i].set_data(
                    [p[0] for p in cc.trace[: frame + 1]],
                    [p[1] for p in cc.trace[: frame + 1]],
                )
                pontos[i].set_data(cc.pos_x, cc.pos_y)

            elementos_proj = []
            try:
                if (
                    linha_proj is not None
                    and ponto_proj is not None
                    and df_trajetoria_projetil is not None
                ):
                    if frame < len(df_trajetoria_projetil):
                        linha_proj.set_data(
                            df_trajetoria_projetil["x_proj"][: frame + 1],
                            df_trajetoria_projetil["y_proj"][: frame + 1],
                        )
                        ponto_proj.set_data(
                            df_trajetoria_projetil["x_proj"][frame],
                            df_trajetoria_projetil["y_proj"][frame],
                        )
                        elementos_proj = [linha_proj, ponto_proj]
            except Exception as e:
                print("[ERRO] Projétil:", e)

            elementos_sonda = []
            try:
                if (
                    linha_sonda is not None
                    and ponto_sonda is not None
                    and df_sonda_limitada is not None
                ):
                    if frame < len(df_sonda_limitada):
                        linha_sonda.set_data(
                            df_sonda_limitada["x_sonda"][: frame + 1],
                            df_sonda_limitada["y_sonda"][: frame + 1],
                        )
                        ponto_sonda.set_data(
                            df_sonda_limitada["x_sonda"][frame],
                            df_sonda_limitada["y_sonda"][frame],
                        )
                        elementos_sonda = [linha_sonda, ponto_sonda]
            except Exception as e:
                print("[ERRO] Sonda:", e)

            return linhas + pontos + elementos_proj + elementos_sonda

        anim = FuncAnimation(
            fig,
            update,
            frames=len(solucao_array),
            interval=self.intervalo_animacao,
            blit=False,
            repeat=False,
        )
        plt.legend()
        plt.show()

    # isso aqui eh statico ainda, acho
    def plotar_estatico(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        self.plotar_corpos_celestes(ax)
        self.criar_plot(ax)
        plt.legend()
        plt.show()
