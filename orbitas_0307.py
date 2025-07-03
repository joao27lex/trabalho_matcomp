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
        
        self.Lua = Corpo_Celeste(massa=7.346e22, raio=1.737e6, color='gray', name='Lua', pos_x=self.Terra.pos_x, pos_y=-3.84e8)
        #aqui eu faço pelo pos_y porque quero a distancia da terra pra lua, a lua ta maluca
        self.Lua.vel_y = self.Terra.vel_y + np.sqrt(self.G*self.Sol.massa / abs(self.Lua.pos_y))
        #self.Lua.vel_y = np.sqrt(self.G*self.Sol.massa / abs(self.Lua.pos_y))
        self.corpos_names.append(self.Lua.name)
        self.corpos_massas.append(self.Lua.massa)
        self.corpos_celestes.append(self.Lua)

        #depois pesquisa um melhor, to fazendo com pressa porcausa da lista de msd
        self.Asteroide = Corpo_Celeste(massa=9e10, raio=1e5, color='black', name='Asteroide', pos_x=1e12, pos_y=1e12)
        self.Asteroide.vel_x = -.4e6
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
                r2 = np.sum(r**2)
                if r2 < 1e-10:
                    continue
                r3 = r2**1.5
                # aqui eu testei negativo e positivo e ele sem '-' dá
                acc_i += - self.G * self.corpos_massas[j] * abs(r) / r3
            
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


    def animar(self):
        solucao_array = self.simular()
        if solucao_array is None:
            print("erro em Universo, Animar")
            return
        
        fig, ax = plt.subplots(figsize=(8,8))
        self.criar_plot(ax)

        linhas = []
        pontos = []
        for cc in self.corpos_celestes:
            linha, = ax.plot([],[], '-', lw=1, label=f'{cc.name} Trajetoria')
            #aqui e o todo #1
            ponto, = ax.plot([],[], 'o', markersize=10, label=f'{cc.name}')
            linhas.append(linha)
            pontos.append(ponto)

        def update(frame):
            #current state in this frame
            current_states = self.get_current_state(solucao_array[frame])
            for i, cc in enumerate(self.corpos_celestes):
                cc.pos_x = current_states[i]['pos_x']
                cc.pos_y = current_states[i]['pos_y']
                cc.vel_x = current_states[i]['vel_x']
                cc.vel_y = current_states[i]['vel_y']

                linhas[i].set_data([p[0] for p in cc.trace[:frame+1]],[p[1] for p in cc.trace[:frame+1]])
                pontos[i].set_data(cc.pos_x, cc.pos_y)
            return linhas + pontos
        
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


if __name__ == "__main__":
    universo = Universo()
    universo.animar()

#TODO 1 , fazer o planeta ficar do tamanho real, ta em universo->animacao a parte do markersize pro ponto
#TODO 2 , consertar a lua
#TODO 3 , adaptar a parte da interceptação, considerando aquelas alterações que ele pediu
#TODO X , a qualquer momento pode fazer aquela detecção de colisão melhorada, no equacoes movimento tem uma parte que verifica se r2 (distancia) eh muito pequena, pode usar algo parecido com isso