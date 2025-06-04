import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import random
from scipy.interpolate import CubicSpline


class OrbitaAsteroide:
    def __init__(self, massa_terra, vel_inicial, duracao=100000, intervalo_animacao=50):
        self.G = 6.67430e-11
        self.softening = 1e6
        self.limite = 2e7

        self.massa_sol = 2e30
        self.massa_terra = massa_terra
        #como a posicao eh muito distante parece que nao aplica força do sol mas aplica sim
        self.posicao_sol_x = 1.5e11
        self.posicao_sol_y = 0.0
        self.posicao_terra_x = 0.0
        self.posicao_terra_y = 0.0

        self.vel_inicial = vel_inicial
        self.duracao = duracao
        self.intervalo_animacao = intervalo_animacao

    def capturar_posicao_inicial(self):
        fig, ax = plt.subplots()
        ax.set_title('Clique para definir a posição inicial do asteroide')
        ax.set_xlim(-self.limite, self.limite)
        ax.set_ylim(-self.limite, self.limite)
        ax.set_aspect('equal')
        ax.plot(0, 0, 'bo', label='Planeta (fixo)')
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

        # t -> terra, s -> sol;
        dxt = x - self.posicao_terra_x
        dyt = y - self.posicao_terra_y
        dxs = x - self.posicao_sol_x
        dys = y - self.posicao_sol_y

        #r^2 = x^2 + y^2
        distancia_presoft_terra = dxt**2 + dyt**2
        distancia_presoft_sol = dxs**2 + dys**2

        #Aqui se faz necessário adcionar o softening para evitar problemas causados pela divisão por zero ou força infinita
        distancia_possoft_terra = (distancia_presoft_terra + self.softening**2)**1.5
        distancia_possoft_sol = (distancia_presoft_sol + self.softening**2)**1.5

        #Agora eh calcular a aceleracao baseado na equacao que desenvolvemos. Fazendo para a Terra (planeta) e o Sol
        aceleracao_xt = -self.G * self.massa_terra * dxt / distancia_possoft_terra
        aceleracao_yt = -self.G * self.massa_terra * dyt / distancia_possoft_terra

        aceleracao_xs = -self.G * self.massa_sol * dxs / distancia_possoft_sol
        aceleracao_ys = -self.G * self.massa_sol * dys / distancia_possoft_sol

        #pra testar aceleracao do sol
        #aceleracao_xt = 0.0
        #aceleracao_yt = 0.0

        # agora eh somar a aceleracao causada pelas forcas
        aceleracao_x_total = aceleracao_xt + aceleracao_xs
        aceleracao_y_total = aceleracao_yt + aceleracao_ys

        return [vx, vy, aceleracao_x_total, aceleracao_y_total]


    def simular(self):
        pos_inicial = self.capturar_posicao_inicial()
        if pos_inicial is None:
            print("Nenhum clique detectado.")
            return None

        estado_inicial = np.concatenate((pos_inicial, self.vel_inicial))
        t_eval = np.linspace(0, self.duracao, 2000)
        sol = solve_ivp(self.equacoes_movimento, (0, self.duracao), estado_inicial, t_eval=t_eval, rtol=1e-6, atol=1e-9)

        self.animar_trajetoria(pos_inicial, sol.y[0], sol.y[1])
        return sol.y[0].tolist(), sol.y[1].tolist(), sol.t.tolist()

    def animar_trajetoria(self, pos_inicial, x_data, y_data):
        fig, ax = plt.subplots()
        ax.set_xlim(-self.limite, self.limite)
        ax.set_ylim(-self.limite, self.limite)
        ax.set_aspect('equal')
        ax.set_title('Órbita do asteroide')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)

        ax.plot(0, 0, 'ro', label='Planeta (fixo)')
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
    def __init__(self, spline_x, spline_y, v_proj, p0_x, p0_y):
        self.spline_x = spline_x
        self.spline_y = spline_y
        self.v_proj = v_proj
        self.p0_x = p0_x
        self.p0_y = p0_y

    def f(self, t):
        dx = self.spline_x(t) - self.p0_x
        dy = self.spline_y(t) - self.p0_y
        return dx**2 + dy**2 - (self.v_proj**2) * t**2

    def df_dt(self, t, h=1e-3):
        return (self.f(t + h) - self.f(t - h)) / (2 * h)

    def newton_raphson(self, t0, tol=1e-3, max_iter=20):
        t = float(t0)  # garantir que é escalar
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

    def simular_interceptacao(self, x_ast, y_ast, t_ast):
        t_ast_escalar = float((t_ast[0] + t_ast[-1]) / 2)  # usa tempo médio
        t_ref = self.newton_raphson(t_ast_escalar)
        x = self.spline_x(t_ref)
        y = self.spline_y(t_ref)
        direcao = np.array([x - self.p0_x, y - self.p0_y])
        dist = np.linalg.norm(direcao)

        t_proj = np.linspace(0, t_ref, 200)
        direcao_unit = direcao / dist
        x_proj = self.p0_x + self.v_proj * direcao_unit[0] * t_proj
        y_proj = self.p0_y + self.v_proj * direcao_unit[1] * t_proj

        plt.figure(figsize=(8, 8))
        plt.plot(x_ast, y_ast, 'r--', label='Asteroide')
        plt.plot(x_proj, y_proj, 'g--', label='Projétil')
        plt.plot([self.p0_x], [self.p0_y], 'bo', label='Planeta (lançamento)')
        plt.plot([x], [y], 'ko', label='Interceptação Refinada')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Interceptação usando Newton-Raphson')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return (x, y), t_ref


if __name__ == "__main__":
    massa_planeta_terra = 5.972e24
    velocidade_inicial_asteroide = [0, 0]
    duracao_total_simulacao = 100000
    velocidade_animacao_ms = 10
    velocidade_inicial_projetil = 3000

    simulador = OrbitaAsteroide(massa_planeta_terra, velocidade_inicial_asteroide, duracao_total_simulacao, velocidade_animacao_ms)
    resultado = simulador.simular()

    if resultado is None:
        exit()

    trajetoria_x, trajetoria_y, lista_tempos = resultado

    quantidade_amostras = 150
    indices_amostrados = sorted(random.sample(range(len(trajetoria_x)), quantidade_amostras))
    posicao_x_sensor = [trajetoria_x[i] for i in indices_amostrados]
    posicao_y_sensor = [trajetoria_y[i] for i in indices_amostrados]
    tempos_sensor = [lista_tempos[i] for i in indices_amostrados]

    spline_cubica_x = CubicSpline(tempos_sensor, posicao_x_sensor)
    spline_cubica_y = CubicSpline(tempos_sensor, posicao_y_sensor)

    tempos_interpolados = np.linspace(min(tempos_sensor), max(tempos_sensor), 1000)
    posicao_x_interpolada = spline_cubica_x(tempos_interpolados)
    posicao_y_interpolada = spline_cubica_y(tempos_interpolados)

    interceptador = Interceptador(spline_cubica_x, spline_cubica_y, velocidade_inicial_projetil, 0, 0)
    interceptador.simular_interceptacao(posicao_x_interpolada, posicao_y_interpolada, tempos_interpolados)