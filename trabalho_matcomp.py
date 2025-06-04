import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import random
from scipy.interpolate import CubicSpline

# ============================
# Função que simula a órbita do asteroide ao redor de um planeta
# ============================
def simular_orbita_asteroide(massa_do_planeta, vetor_velocidade_inicial, duracao_simulacao=100000, intervalo_animacao_ms=50):
    constante_gravitacional = 6.67430e-11  # Constante G (m^3 kg^-1 s^-2)
    parametro_softening = 1e6              # Parâmetro para evitar singularidade na força gravitacional
    limite_espacial = 2e7                  # Define os limites da visualização (em metros)

    # ========== Etapa 1: Capturar a posição inicial do asteroide via clique ==========
    fig, ax = plt.subplots()
    ax.set_title('Clique para definir a posição inicial do asteroide')
    ax.set_xlim(-limite_espacial, limite_espacial)
    ax.set_ylim(-limite_espacial, limite_espacial)
    ax.set_aspect('equal')
    ax.plot(0, 0, 'bo', label='Planeta (fixo)')  # O planeta está fixo no centro (0,0), mostrado em azul
    ax.grid(True)
    posicao_inicial_asteroide = []

    # Função de callback que captura o clique do usuário
    def capturar_clique(evento):
        if evento.inaxes == ax:
            posicao_inicial_asteroide.append(evento.xdata)
            posicao_inicial_asteroide.append(evento.ydata)
            plt.close()  # Fecha o gráfico após o clique

    fig.canvas.mpl_connect('button_press_event', capturar_clique)
    plt.legend()
    plt.show()

    if len(posicao_inicial_asteroide) < 2:
        print("Nenhum clique detectado.")
        return  # Encerra se o usuário não clicar

    # ========== Etapa 2: Resolver a EDO da órbita com solve_ivp ==========
    estado_inicial = np.concatenate((posicao_inicial_asteroide, vetor_velocidade_inicial))
    instantes_de_tempo = np.linspace(0, duracao_simulacao, 2000)

    # Define o sistema dinâmico: movimento sob influência gravitacional
    def equacoes_de_movimento(t, estado):
        x, y, vx, vy = estado
        distancia2 = x**2 + y**2
        distancia3_soft = (distancia2 + parametro_softening**2)**1.5
        aceleracao_x = -constante_gravitacional * massa_do_planeta * x / distancia3_soft
        aceleracao_y = -constante_gravitacional * massa_do_planeta * y / distancia3_soft
        return [vx, vy, aceleracao_x, aceleracao_y]

    # Resolve o sistema de equações diferenciais
    solucao = solve_ivp(
        equacoes_de_movimento,
        (0, duracao_simulacao),
        estado_inicial,
        t_eval=instantes_de_tempo,
        rtol=1e-6,
        atol=1e-9
    )

    # Extrai as posições e os tempos da solução
    trajetoria_x = solucao.y[0]
    trajetoria_y = solucao.y[1]
    tempos_simulados = solucao.t

    # ========== Etapa 3: Animação da órbita do asteroide ==========
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-limite_espacial, limite_espacial)
    ax.set_ylim(-limite_espacial, limite_espacial)
    ax.set_title('Órbita do asteroide')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True)

    ax.plot(0, 0, 'ro', label='Planeta (fixo)')
    ax.plot(posicao_inicial_asteroide[0], posicao_inicial_asteroide[1], 'go', label='Início')
    linha_trajetoria, = ax.plot([], [], 'r-', lw=2, label='Trajetória')
    ponto_asteroide, = ax.plot([], [], 'ko')

    def inicializar_animacao():
        linha_trajetoria.set_data([], [])
        ponto_asteroide.set_data([], [])
        return linha_trajetoria, ponto_asteroide

    def atualizar_animacao(frame):
        x = trajetoria_x[frame]
        y = trajetoria_y[frame]
        linha_trajetoria.set_data(trajetoria_x[:frame], trajetoria_y[:frame])
        ponto_asteroide.set_data([x], [y])
        return linha_trajetoria, ponto_asteroide

    animacao = FuncAnimation(
        fig,
        atualizar_animacao,
        frames=len(trajetoria_x),
        init_func=inicializar_animacao,
        interval=intervalo_animacao_ms,
        blit=False
    )

    plt.legend()
    plt.show()

    return trajetoria_x.tolist(), trajetoria_y.tolist(), tempos_simulados.tolist()


# ============================
# Parâmetros da simulação da órbita
# ============================
massa_planeta_terra = 5.972e24
velocidade_inicial_asteroide = [-6000, 0]
duracao_total_simulacao = 100000
velocidade_animacao_ms = 10

# Executa a simulação da órbita e só continua se o usuário clicou
resultado = simular_orbita_asteroide(
    massa_planeta_terra,
    velocidade_inicial_asteroide,
    duracao_total_simulacao,
    velocidade_animacao_ms
)

if resultado is None:
    exit()  # Encerra o código se não houve clique

trajetoria_x, trajetoria_y, lista_tempos = resultado

# ============================
# Coleta de dados do sensor (amostragem aleatória da órbita)
# ============================
quantidade_amostras = 150
indices_amostrados = sorted(random.sample(range(len(trajetoria_x)), quantidade_amostras))
posicao_x_sensor = [trajetoria_x[i] for i in indices_amostrados]
posicao_y_sensor = [trajetoria_y[i] for i in indices_amostrados]
tempos_sensor = [lista_tempos[i] for i in indices_amostrados]

# ============================
# Interpolação da trajetória com splines cúbicas
# ============================
spline_cubica_x = CubicSpline(tempos_sensor, posicao_x_sensor)
spline_cubica_y = CubicSpline(tempos_sensor, posicao_y_sensor)

tempos_interpolados = np.linspace(min(tempos_sensor), max(tempos_sensor), 1000)
posicao_x_interpolada = spline_cubica_x(tempos_interpolados)
posicao_y_interpolada = spline_cubica_y(tempos_interpolados)

# ============================
# Função que simula o lançamento de um projétil
# ============================
def refinar_tempo_interceptacao_newton(spline_x, spline_y, v_proj, p0_x, p0_y, t0,
                                        max_iter=20, tol=1e-3):
    def f(t):
        x_ast = spline_x(t)
        y_ast = spline_y(t)
        dx = x_ast - p0_x
        dy = y_ast - p0_y
        d2 = dx**2 + dy**2
        return d2 - (v_proj**2) * t**2

    def df_dt(t, h=1e-3):
        return (f(t + h) - f(t - h)) / (2 * h)

    t = t0
    for _ in range(max_iter):
        ft = f(t)
        dft = df_dt(t)
        if abs(dft) < 1e-8:
            print("[WARN] Derivada muito pequena. Encerrando Newton-Raphson.")
            break
        t_new = t - ft / dft
        if abs(t_new - t) < tol:
            return t_new
        t = t_new
    print("[WARN] Newton-Raphson não convergiu após", max_iter, "iterações.")
    return t


# Função que simula a interceptação com refinamento

def simular_projetil_com_newton(posicao_x_asteroide, posicao_y_asteroide, tempos_asteroide,
                                velocidade_projetil, po_x, po_y, spline_x, spline_y):
    raio_do_planeta = 6.371e6

    tempo_impacto = None
    for i in range(len(posicao_x_asteroide)):
        dist = np.hypot(posicao_x_asteroide[i], posicao_y_asteroide[i])
        if dist <= raio_do_planeta:
            tempo_impacto = tempos_asteroide[i]
            break

    if tempo_impacto is None:
        tempo_impacto = tempos_asteroide[-1]
        print("[INFO] O asteroide não colidiu com o planeta durante a simulação.")

    melhor_ponto = None
    melhor_t = None
    menor_erro = float('inf')

    for i in range(len(tempos_asteroide)):
        t_ast = tempos_asteroide[i]
        if t_ast >= tempo_impacto:
            break
        x_ast = posicao_x_asteroide[i]
        y_ast = posicao_y_asteroide[i]
        d = np.hypot(x_ast - po_x, y_ast - po_y)
        tempo_projetil = d / velocidade_projetil
        if tempo_projetil > t_ast:
            continue
        erro = abs(tempo_projetil - t_ast)
        if erro < menor_erro:
            menor_erro = erro
            melhor_ponto = (x_ast, y_ast)
            melhor_t = t_ast

    if melhor_ponto is None:
        print("[ALERTA] Nenhuma interceptação possível antes da colisão.")
        return None

    # Refinar tempo de interceptação com Newton-Raphson
    t_refinado = refinar_tempo_interceptacao_newton(spline_x, spline_y, velocidade_projetil,
                                                    po_x, po_y, melhor_t)

    x_ast = spline_x(t_refinado)
    y_ast = spline_y(t_refinado)
    direcao = np.array([x_ast - po_x, y_ast - po_y])
    dist = np.linalg.norm(direcao)
    
    raio_do_planeta = 6.371e6  # metros
    margem_de_erro = 20000      # 1 km
    limite_colisao = raio_do_planeta + margem_de_erro


    if dist > limite_colisao:
        print("[INFO] Pode não haver colisão precisa.")

    # Gerar trajetórias
    t_proj = np.linspace(0, t_refinado, 200)
    direcao_unitaria = direcao / dist
    x_proj = po_x + velocidade_projetil * direcao_unitaria[0] * t_proj
    y_proj = po_y + velocidade_projetil * direcao_unitaria[1] * t_proj

    # Gráfico
    plt.figure(figsize=(8, 8))
    plt.plot(posicao_x_asteroide, posicao_y_asteroide, 'r--', label='Asteroide')
    plt.plot(x_proj, y_proj, 'g--', label='Projétil')
    plt.plot([po_x], [po_y], 'bo', label='Planeta (lançamento)')
    plt.plot([x_ast], [y_ast], 'ko', label='Interceptação Refinada')
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Interceptação usando Newton-Raphson')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (x_ast, y_ast), t_refinado

# ============================
# Simula o lançamento do projétil
# ============================
velocidade_inicial_projetil = 3000  # m/s
simular_projetil_com_newton(
    posicao_x_interpolada,
    posicao_y_interpolada,
    tempos_interpolados,
    velocidade_inicial_projetil,
    0, 0,
    spline_cubica_x,
    spline_cubica_y
)

