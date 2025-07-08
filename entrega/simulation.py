import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
import random
from scipy.interpolate import CubicSpline


class Corpo_Celeste:
    def __init__(self, massa, raio, color, name, pos_x, pos_y, vel_x = 0.0, vel_y = 0.0, abg=True, awg=True):
        self.massa = massa
        self.raio = raio
        self.color = color
        self.name = name
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        #abg - affected by gravity, se esse corpo celeste vai ter sua aceleração afetada pela gravidade. Padrão False
        self.abg = abg
        #awg - affect with gravity, se esse corpo celeste vai afetar a aceleração dos outros. Padrão True
        self.awg = awg
        self.trace = []

    #essa funcao é pro equacoes_movimento
    #x, y, vx, vy = estado
    def return_estado(self):
        info = [self.pos_x, self.pos_y, self.vel_x, self.vel_y]
        return info
    
    def return_pos(self):
        info = [self.pos_x, self.pos_y]
        return info
    
    def return_vel(self):
        info = [self.vel_x, self.vel_y]
        return info
    
    def return_name(self):
        name = self.name
        return name
    

class Universo:
    def __init__(self, duracao_padrao=100000, ganho_duracao=100,intervalo_animacao=10):
        self.G = 6.67430e-11
        self.duracao = duracao_padrao*ganho_duracao
        self.intervalo_animacao = intervalo_animacao
        #se deixar no criar_plot como equal, datalim ele só ignora o limite menor. mas se usar o razao, box, ele vai considerar o limite menor tambem (porém o raio fica errado)
        #self.limite_x = 2e11
        #self.limite_y = 1e10
        self.limite_x = 1e12
        self.limite_y = 1e12
        self.criar_corpos_celestes()
        self.y0 = self.get_y0()
        

    def criar_corpos_celestes(self):
        self.corpos_celestes = []
        self.corpos_names = []
        self.corpos_massas = []
        #acho que deve ter uma forma melhor de pegar names e massa, ja que eh de todos

        #sempre fazer o sol ser o corpo fixo, no 0,0 com index 0 no corpos_celestes
        self.Sol = Corpo_Celeste(massa=2e30, raio=6.957e8, color='yellow', name='Sol', pos_x=0, pos_y=0.0, abg=False)
        self.fixed_body_name = "Sol"
        self.fixed_body_index = 0
        self.corpos_names.append(self.Sol.name)
        self.corpos_massas.append(self.Sol.massa)
        self.corpos_celestes.append(self.Sol)

        self.Marte = Corpo_Celeste(massa=6.4e23, raio=3.389e6, color='red', name='Marte', pos_x=2.279e11, pos_y=0.0)
        #vou usar vel no y como inicial
        #self.CC_vel_y = np.sqrt(self.G * self.main_cc.mass / self.distance_to_main_cc)
        self.Marte.vel_y = np.sqrt(self.G*self.Sol.massa / self.Marte.pos_x)
        self.corpos_names.append(self.Marte.name)
        self.corpos_massas.append(self.Marte.massa)
        self.corpos_celestes.append(self.Marte)
        
        self.Terra = Corpo_Celeste(massa=5.972e24, raio=6.371e6, color='blue', name='Terra', pos_x=1.496e11, pos_y=0.0)
        self.Terra.vel_y = np.sqrt(self.G*self.Sol.massa / self.Terra.pos_x)
        self.corpos_names.append(self.Terra.name)
        self.corpos_massas.append(self.Terra.massa)
        self.corpos_celestes.append(self.Terra)
        
        self.Lua = Corpo_Celeste(massa=7.346e22, raio=1.737e6, color='white', name='Lua', pos_x=self.Terra.pos_x, pos_y=-3.84e8)
        # logo após definir a posição da Lua
        distancia_lua_terra = abs(self.Lua.pos_y - self.Terra.pos_y)

        v_rel_lua = np.sqrt(self.G * self.Terra.massa / distancia_lua_terra)

        # velocidade da Terra
        v_terra = np.array([self.Terra.vel_x, self.Terra.vel_y])

        # vetor da Terra para a Lua
        vetor_terra_lua = np.array([self.Lua.pos_x - self.Terra.pos_x,
                                    self.Lua.pos_y - self.Terra.pos_y])
        norma = np.linalg.norm(vetor_terra_lua)

        # direção perpendicular (para tangenciar a órbita) - rotação de 90 graus
        direcao_tangente = np.array([-vetor_terra_lua[1], vetor_terra_lua[0]]) / norma

        # velocidade relativa da Lua em relação à Terra (tangencial à sua órbita)
        v_lua_rel = direcao_tangente * v_rel_lua

        # velocidade total da Lua no referencial inercial (sistema solar)
        v_lua_total = v_terra + v_lua_rel

        self.Lua.vel_x = v_lua_total[0]
        self.Lua.vel_y = v_lua_total[1]
        
        self.corpos_names.append(self.Lua.name)
        self.corpos_massas.append(self.Lua.massa)
        self.corpos_celestes.append(self.Lua)

        #depois pesquisa um melhor, to fazendo com pressa porcausa da lista de msd
        self.Asteroide = Corpo_Celeste(massa=9e10, raio=1e5, color='black', name='Asteroide', pos_x=0.75e12, pos_y=2.5e11)
        self.Asteroide.vel_x = -.4e5
        self.Asteroide.vel_y = -3e4
        self.corpos_names.append(self.Asteroide.name)
        self.corpos_massas.append(self.Asteroide.massa)
        self.corpos_celestes.append(self.Asteroide)
        

    #y0 = vetor inicial
    def get_y0(self):
        y0= []
        for index, cc in enumerate(self.corpos_celestes):
            if index != self.fixed_body_index:
                estado = cc.return_estado()
                y0.extend(estado)
                #y0.extend([cc.pos_x],[cc.pos_y],[cc.vel_x],[cc.vel_y])
        return np.array(y0)

    #equacoes_movimento_setup
    def equacoes_movimento_setup(self,y):
        all_positions = []
        all_velocities = []

        #como temos xa,ya,vxa,vya,xb,yb,vxb,vyb,xc,yc,vxc,vyc
        #precisamos de um ponteiro = ptr pra ir em cada valor de forma organizada
        ptr = 0
        for i, cc in enumerate(self.corpos_celestes):
            if i == self.fixed_body_index:
                all_positions.append(np.array(cc.return_pos()))
                all_velocities.append(np.array(cc.return_vel()))
            else:
                all_positions.append(np.array([y[ptr], y[ptr+1]]))
                all_velocities.append(np.array([y[ptr+2], y[ptr+3]]))
                ptr += 4

        dydt = np.zeros_like(y)
        return all_positions, all_velocities, dydt


    #tem que deixar t aqui pelo solve_ivp
    def equacoes_movimento(self, t, y):
        all_positions, all_velocities, dydt = self.equacoes_movimento_setup(y)

        #dessa vez o ptr eh pro vx,vy,ax,ay,...
        ptr = 0
        for i in range(len(self.corpos_celestes)):
            #se for o index do sol ele passa pra prox iteracao
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
            dydt[ptr+1] = all_velocities[i][1]
            dydt[ptr+2] = acc_i[0]
            dydt[ptr+3] = acc_i[1]

            ptr += 4
        
        return dydt

    def get_current_state(self, y_in_t):
        current_states = []
        ptr = 0
        for i, cc in enumerate(self.corpos_celestes):
            if i == self.fixed_body_index:
                current_states.append({'pos_x': cc.pos_x, 'pos_y': cc.pos_y, 'vel_x': cc.vel_x, 'vel_y': cc.vel_y})
            else:
                current_states.append({'pos_x':y_in_t[ptr], 'pos_y':y_in_t[ptr+1], 'vel_x': y_in_t[ptr+2], 'vel_y': y_in_t[ptr+3]})
                ptr += 4
        return current_states
    
    def simular(self):
        for cc in self.corpos_celestes:
            cc.trace = []
        t_eval = np.linspace(0, self.duracao, 2000)
        rtol = 1e-6
        atol = 1e-9
        solucao = solve_ivp(self.equacoes_movimento, (0,self.duracao), self.y0, method='RK45', t_eval=t_eval, rtol=rtol, atol = atol)
        
        solucao_array = solucao.y.T

        for step in range(len(t_eval)):
            y_in_t = solucao_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.corpos_celestes[i].pos_x = state['pos_x']
                self.corpos_celestes[i].pos_y = state['pos_y']
                self.corpos_celestes[i].vel_x = state['vel_x']
                self.corpos_celestes[i].vel_y = state['vel_y']
                self.corpos_celestes[i].trace.append((state['pos_x'], state['pos_y']))

        return solucao_array
    
    def criar_plot(self, ax):
        ax.set_xlim(-self.limite_x, self.limite_x)
        ax.set_ylim(-self.limite_y, self.limite_y)
        ax.set_aspect('equal', adjustable='datalim')
        #esse de baixo deixa desproporcional e as esferas viram (), talvez desse pra usar elipse ao inves de circle pra ajustar com a razao tambem, não tenho certeza
        #razao = self.limite_x / self.limite_y
        #ax.set_aspect(razao, adjustable='box')
        
        ax.set_facecolor('gray')
        #sei que nao eh universo
        ax.set_title('Universo teste')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')


    def animar(self, solucao_array):
        if solucao_array is None:
            print("erro em Universo, Animar")
            return
        
        fig, ax = plt.subplots(figsize=(8,8))
        self.criar_plot(ax)
        
       # === Projétil ===
        if 'df_trajetoria_projetil' in globals() and df_trajetoria_projetil is not None:
            linha_proj, = ax.plot([], [], '--', color='green', lw=2, label='Projétil')
            ponto_proj, = ax.plot([], [], 'o', color='green', markersize=6)
        else:
            linha_proj, ponto_proj = None, None

        # === Sonda com atuadores limitados ===
        if 'df_sonda_limitada' in globals() and df_sonda_limitada is not None:
            linha_sonda, = ax.plot([], [], '--', color='orange', lw=2, label='Sonda Limitada')
            ponto_sonda, = ax.plot([], [], 'o', color='orange', markersize=6)
        else:
            linha_sonda, ponto_sonda = None, None


        linhas = []
        pontos = []
        for cc in self.corpos_celestes:
            linha, = ax.plot([], [], '-', lw=2, label=f'{cc.name} Trajetoria', color=cc.color)
            
            if cc.name == 'Lua':
                ponto, = ax.plot([], [], 'o', markersize=6, label=f'{cc.name}', 
                                markerfacecolor=cc.color, markeredgecolor='black', markeredgewidth=1.0)
            else:
                ponto, = ax.plot([], [], 'o', markersize=12, label=f'{cc.name}', 
                                markerfacecolor=cc.color)
            
            linhas.append(linha)
            pontos.append(ponto)


        def update(frame):
            current_states = self.get_current_state(solucao_array[frame])
            for i, cc in enumerate(self.corpos_celestes):
                cc.pos_x = current_states[i]['pos_x']
                cc.pos_y = current_states[i]['pos_y']
                cc.vel_x = current_states[i]['vel_x']
                cc.vel_y = current_states[i]['vel_y']

                linhas[i].set_data([p[0] for p in cc.trace[:frame+1]],
                                [p[1] for p in cc.trace[:frame+1]])
                pontos[i].set_data(cc.pos_x, cc.pos_y)

            elementos_proj = []
            if 'df_trajetoria_projetil' in globals() and df_trajetoria_projetil is not None:
                if frame < len(df_trajetoria_projetil):
                    linha_proj.set_data(df_trajetoria_projetil['x_proj'][:frame+1],
                                        df_trajetoria_projetil['y_proj'][:frame+1])
                    ponto_proj.set_data(df_trajetoria_projetil['x_proj'][frame],
                                        df_trajetoria_projetil['y_proj'][frame])
                    elementos_proj = [linha_proj, ponto_proj]

            elementos_sonda = []
            if 'df_sonda_limitada' in globals() and df_sonda_limitada is not None:
                if frame < len(df_sonda_limitada):
                    linha_sonda.set_data(df_sonda_limitada['x_sonda'][:frame+1],
                                        df_sonda_limitada['y_sonda'][:frame+1])
                    ponto_sonda.set_data(df_sonda_limitada['x_sonda'][frame],
                                        df_sonda_limitada['y_sonda'][frame])
                    elementos_sonda = [linha_sonda, ponto_sonda]

            return linhas + pontos + elementos_proj + elementos_sonda

        
        anim = FuncAnimation(fig, update, frames=len(solucao_array), interval=self.intervalo_animacao, blit=False, repeat=False)
        plt.legend()
        plt.show()


    #isso aqui eh statico ainda, acho
    def plotar_estatico(self):
        fig, ax = plt.subplots(figsize=(8,8))
        self.plotar_corpos_celestes(ax)
        self.criar_plot(ax)
        plt.legend()
        plt.show()

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

    t_refinado = refinar_tempo_interceptacao_newton(spline_x, spline_y, velocidade_projetil,
                                                    po_x, po_y, melhor_t)

    x_ast = spline_x(t_refinado)
    y_ast = spline_y(t_refinado)
    direcao = np.array([x_ast - po_x, y_ast - po_y])
    dist = np.linalg.norm(direcao)
    
    t_proj = np.linspace(0, t_refinado, 200)
    direcao_unitaria = direcao / dist
    x_proj = po_x + velocidade_projetil * direcao_unitaria[0] * t_proj
    y_proj = po_y + velocidade_projetil * direcao_unitaria[1] * t_proj

    import pandas as pd
    return pd.DataFrame({'x_proj': x_proj, 'y_proj': y_proj, 'tempo': t_proj})


def simular_sonda_com_atuadores_limitados(
    spline_x, spline_y, t_max, dt=10, v_inicial=3e5, a_max=50
):
    tempos = np.arange(0, t_max, dt)
    x_sonda = [0]
    y_sonda = [0]
    vx_sonda = [v_inicial]
    vy_sonda = [0]

    for i in range(1, len(tempos)):
        t = tempos[i]

        # Posição atual
        x = x_sonda[-1]
        y = y_sonda[-1]

        # Velocidade atual
        vx = vx_sonda[-1]
        vy = vy_sonda[-1]

        # Posição alvo (asteroide estimado)
        x_alvo = spline_x(t)
        y_alvo = spline_y(t)

        dx = x_alvo - x
        dy = y_alvo - y
        distancia = np.hypot(dx, dy)
        if distancia == 0:
            ax = ay = 0
        else:
            ax = a_max * (dx / distancia)
            ay = a_max * (dy / distancia)

        # Atualiza velocidade
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        # Atualiza posição
        x_new = x + vx * dt
        y_new = y + vy * dt

        x_sonda.append(x_new)
        y_sonda.append(y_new)
        vx_sonda.append(vx_new)
        vy_sonda.append(vy_new)

    import pandas as pd
    return pd.DataFrame({'tempo': tempos, 'x_sonda': x_sonda, 'y_sonda': y_sonda})


if __name__ == "__main__":
    universo = Universo()
    solucao = universo.simular()

    # === 1. Obter a trajetória do asteroide
    asteroide = next(c for c in universo.corpos_celestes if c.name == "Asteroide")
    trajetoria_x = [p[0] for p in asteroide.trace]
    trajetoria_y = [p[1] for p in asteroide.trace]
    lista_tempos = np.linspace(0, universo.duracao, len(trajetoria_x))

    # === 2. Coleta dos dados do sensor (amostragem aleatória da órbita)
    quantidade_amostras = 150
    indices_amostrados = sorted(random.sample(range(len(trajetoria_x)), quantidade_amostras))
    posicao_x_sensor = [trajetoria_x[i] for i in indices_amostrados]
    posicao_y_sensor = [trajetoria_y[i] for i in indices_amostrados]
    tempos_sensor = [lista_tempos[i] for i in indices_amostrados]

    # === 3. Interpolação da trajetória com splines cúbicas
    spline_cubica_x = CubicSpline(tempos_sensor, posicao_x_sensor)
    spline_cubica_y = CubicSpline(tempos_sensor, posicao_y_sensor)
    
    # === 4.1 Simular a sonda com atuadores limitados
    df_sonda_limitada = simular_sonda_com_atuadores_limitados(
        spline_x=spline_cubica_x,
        spline_y=spline_cubica_y,
        t_max=universo.duracao,
        dt=300,
        v_inicial=3e5,
        a_max=10  # você pode testar 10, 20, 50 m/s²...
    )
    
    


    # === 4. Simular o lançamento do projétil
    df_trajetoria_projetil = simular_projetil_com_newton(
        posicao_x_asteroide=trajetoria_x,
        posicao_y_asteroide=trajetoria_y,
        tempos_asteroide=lista_tempos,
        velocidade_projetil=3e5,  # m/s
        po_x=0, po_y=0,
        spline_x=spline_cubica_x,
        spline_y=spline_cubica_y
    )

    # === 5. Visualizar ou usar como quiser
    if df_trajetoria_projetil is not None:
        print(df_trajetoria_projetil.head())
        # aqui você pode animar ou plotar com matplotlib, por exemplo

    universo.animar(solucao)

#TODO 1 , fazer o planeta ficar do tamanho real, ta em universo->animacao a parte do markersize pro ponto

#TODO 3 , adaptar a parte da interceptação, considerando aquelas alterações que ele pediu
#TODO X , a qualquer momento pode fazer aquela detecção de colisão melhorada, no equacoes movimento tem uma parte que verifica se r2 (distancia) eh muito pequena, pode usar algo parecido com isso