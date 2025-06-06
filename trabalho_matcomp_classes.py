import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
import random
from scipy.interpolate import CubicSpline


class OrbitaAsteroide:
    def __init__(self, massa_planeta, vel_inicial, duracao=100000, intervalo_animacao=50):
        self.G = 6.67430e-11
        self.softening = 1e6
        self.limite = 2e7
        self.massa_planeta = massa_planeta
        self.vel_inicial = vel_inicial
        self.duracao = duracao
        self.intervalo_animacao = intervalo_animacao
        self.zoom_out_animacao = 5
        self.zoom_out_captura = 5

        # Sol
        self.massa_sol = 2e30
        self.posicao_sol_x = 1.5e11
        self.posicao_sol_y = 0.0
        self.posicao_terra_x = 0.0
        self.posicao_terra_y = 0.0

    def capturar_posicao_inicial(self):
        fig, ax = plt.subplots()
        ax.set_title('Clique para definir a posição inicial do asteroide')
        ax.set_xlim(-self.zoom_out_captura * self.limite, self.zoom_out_captura * self.limite)
        ax.set_ylim(-self.zoom_out_captura * self.limite, self.zoom_out_captura * self.limite)
        ax.set_aspect('equal')
        raio_terra = 6.371e6
        terra = Circle((0, 0), raio_terra, color='blue', label='Terra (escala real)')
        ax.add_patch(terra)
        ax.grid(True)
        posicao = []

        def clique(evento):
            if evento.inaxes == ax:
                posicao.append(evento.xdata)
                posicao.append(evento.ydata)
                plt.close()

        fig.canvas.mpl_connect('button_press_event', clique)
        plt.legend()
        plt.show()
        return posicao if len(posicao) == 2 else None

    def equacoes_movimento(self, t, estado):
        x, y, vx, vy = estado
        dx_terra = x - self.posicao_terra_x
        dy_terra = y - self.posicao_terra_y
        dx_sol = x - self.posicao_sol_x
        dy_sol = y - self.posicao_sol_y

        r2_terra = dx_terra**2 + dy_terra**2
        r3_terra = (r2_terra + self.softening**2)**1.5
        r2_sol = dx_sol**2 + dy_sol**2
        r3_sol = (r2_sol + self.softening**2)**1.5

        ax_terra = -self.G * self.massa_planeta * dx_terra / r3_terra
        ay_terra = -self.G * self.massa_planeta * dy_terra / r3_terra
        ax_sol = -self.G * self.massa_sol * dx_sol / r3_sol
        ay_sol = -self.G * self.massa_sol * dy_sol / r3_sol

        return [vx, vy, ax_terra + ax_sol, ay_terra + ay_sol]

    def simular(self):
        pos_inicial = self.capturar_posicao_inicial()
        if pos_inicial is None:
            print("Nenhum clique detectado.")
            return None

        estado_inicial = np.concatenate((pos_inicial, self.vel_inicial))
        t_eval = np.linspace(0, self.duracao, 2000)
        sol = solve_ivp(self.equacoes_movimento, (0, self.duracao), estado_inicial, t_eval=t_eval,
                        rtol=1e-6, atol=1e-9)

        # anima trajetória completa do asteroide
        self.animar_trajetoria(pos_inicial, sol.y[0], sol.y[1])
        return sol.y[0].tolist(), sol.y[1].tolist(), sol.t.tolist()

    def animar_trajetoria(self, pos_inicial, x_data, y_data):
        fig, ax = plt.subplots()
        ax.set_xlim(-self.zoom_out_animacao * self.limite, self.zoom_out_animacao * self.limite)
        ax.set_ylim(-self.zoom_out_animacao * self.limite, self.zoom_out_animacao * self.limite)
        ax.set_aspect('equal')
        ax.set_title('Órbita do asteroide')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        raio_terra = 6.371e6
        terra = Circle((0, 0), raio_terra, color='blue', label='Terra (escala real)')
        ax.add_patch(terra)
        ax.plot(pos_inicial[0], pos_inicial[1], 'go', label='Início')
        linha, = ax.plot([], [], 'r-', lw=2, label='Trajetória')
        ponto, = ax.plot([], [], 'ko')

        def init():
            linha.set_data([], [])
            ponto.set_data([], [])
            return linha, ponto

        def update(frame):
            linha.set_data(x_data[:frame], y_data[:frame])
            ponto.set_data([x_data[frame]], [y_data[frame]])
            return linha, ponto

        anim = FuncAnimation(fig, update, frames=len(x_data), init_func=init,
                             interval=self.intervalo_animacao, blit=False)
        plt.legend()
        plt.show()


class Interceptador:
    def __init__(self, spline_x, spline_y, v_proj, p0_x, p0_y, limite_plot):
        self.spline_x = spline_x
        self.spline_y = spline_y
        self.v_proj = v_proj
        self.p0_x = p0_x
        self.p0_y = p0_y
        self.limite = limite_plot

    def f(self, t):
        dx = self.spline_x(t) - self.p0_x
        dy = self.spline_y(t) - self.p0_y
        return dx**2 + dy**2 - (self.v_proj**2) * t**2

    def df_dt(self, t, h=1e-3):
        return (self.f(t + h) - self.f(t - h)) / (2 * h)

    def newton_raphson(self, t0, tol=1e-3, max_iter=20):
        t = float(t0)
        for _ in range(max_iter):
            ft = self.f(t)
            dft = self.df_dt(t)
            if np.abs(dft) < 1e-8:
                break
            t_new = t - ft / dft
            if np.abs(t_new - t) < tol:
                return t_new
            t = t_new
        return t

    def simular_interceptacao(self, t_ast, x_ast, y_ast):
        """
        t_ast, x_ast, y_ast: arrays de tempos e posições do asteroide (linha spline)
        """
        # Escolhe como chute inicial um tempo = 5% do final de t_ast
        t0_guess = t_ast[int(len(t_ast) * 0.05)]
        t_ref = self.newton_raphson(t0_guess)

        # Ponto de interceptação exato (usando a spline em t_ref)
        x_int = self.spline_x(t_ref)
        y_int = self.spline_y(t_ref)

        # Direção e distância do projétil lançado de (p0_x, p0_y) até (x_int, y_int)
        direcao = np.array([x_int - self.p0_x, y_int - self.p0_y])
        dist = np.linalg.norm(direcao)
        direcao_unit = direcao / dist if dist != 0 else np.array([1.0, 0.0])

        # Número de pontos/frame para a animação (tanto do asteroide quanto do projétil)
        n_pontos = 300

        # Gera o vetor de tempos do projétil de 0 até t_ref em n_pontos
        t_proj = np.linspace(0, t_ref, n_pontos)

        # Posições do projétil (movimento em linha reta)
        x_proj = self.p0_x + self.v_proj * direcao_unit[0] * t_proj
        y_proj = self.p0_y + self.v_proj * direcao_unit[1] * t_proj

        # Posições do asteroide nos mesmos instantes t_proj (avaliando a spline)
        x_ast_proj = self.spline_x(t_proj)
        y_ast_proj = self.spline_y(t_proj)

        # Configuração da figura para animação
        fig, ax = plt.subplots()
        ax.set_xlim(-5 * self.limite, 5 * self.limite)
        ax.set_ylim(-5 * self.limite, 5 * self.limite)
        ax.set_aspect('equal')
        ax.set_title('Animação da Interceptação')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)

        raio_terra = 6.371e6
        terra = Circle((0, 0), raio_terra, color='blue', label='Terra (escala real)')
        ax.add_patch(terra)

        # Linhas que serão atualizadas a cada frame
        ast_line, = ax.plot([], [], 'r-', label='Asteroide')
        proj_line, = ax.plot([], [], 'g-', label='Projétil')
        intersecao, = ax.plot([x_int], [y_int], 'kx', markersize=10, label='Interceptação')

        def init():
            ast_line.set_data([], [])
            proj_line.set_data([], [])
            return ast_line, proj_line, intersecao

        def update(frame):
            ast_line.set_data(x_ast_proj[:frame], y_ast_proj[:frame])
            proj_line.set_data(x_proj[:frame], y_proj[:frame])
            return ast_line, proj_line, intersecao

        anim = FuncAnimation(fig, update, frames=n_pontos, init_func=init,
                             interval=20, blit=True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Calcula o erro final: diferença entre o ponto X e a última posição do projétil
        erro = np.linalg.norm(np.array([x_int, y_int]) - np.array([x_proj[-1], y_proj[-1]]))
        if erro > 1000:
            print(f"Interceptação falhou! Distância final: {erro:.2f} metros")
        else:
            print(f"Interceptação bem-sucedida! Distância final: {erro:.2f} metros")

        return (x_int, y_int), t_ref


if __name__ == "__main__":
    # Parâmetros da simulação
    massa_planeta_terra = 5.972e24
    velocidade_inicial_asteroide = [-20000, 0]  # [vx, vy] inicial do asteroide
    duracao_total_simulacao = 100000
    velocidade_animacao_ms = 1
    velocidade_inicial_projetil = 11000  # módulo da velocidade do projétil

    # 1) Simula órbita do asteroide
    simulador = OrbitaAsteroide(
        massa_planeta_terra,
        velocidade_inicial_asteroide,
        duracao_total_simulacao,
        velocidade_animacao_ms
    )
    resultado = simulador.simular()
    if resultado is None:
        exit()

    trajetoria_x, trajetoria_y, lista_tempos = resultado

    # 2) Seleciona amostras aleatórias para construir a spline
    quantidade_amostras = 150
    indices_amostrados = sorted(random.sample(range(len(trajetoria_x)), quantidade_amostras))
    posicao_x_sensor = [trajetoria_x[i] for i in indices_amostrados]
    posicao_y_sensor = [trajetoria_y[i] for i in indices_amostrados]
    tempos_sensor = [lista_tempos[i] for i in indices_amostrados]

    spline_cubica_x = CubicSpline(tempos_sensor, posicao_x_sensor)
    spline_cubica_y = CubicSpline(tempos_sensor, posicao_y_sensor)

    # 3) Cria vetor de tempos interpolados (linspace entre o menor e o maior tempo de sensor)
    tempos_interpolados = np.linspace(min(tempos_sensor), max(tempos_sensor), 1000)
    posicao_x_interpolada = spline_cubica_x(tempos_interpolados)
    posicao_y_interpolada = spline_cubica_y(tempos_interpolados)

    # 4) Simula interceptação usando a spline
    interceptador = Interceptador(
        spline_cubica_x,
        spline_cubica_y,
        velocidade_inicial_projetil,
        0, 0,  # ponto de lançamento do projétil (origem)
        simulador.limite
    )
    interceptador.simular_interceptacao(tempos_interpolados, posicao_x_interpolada, posicao_y_interpolada)
