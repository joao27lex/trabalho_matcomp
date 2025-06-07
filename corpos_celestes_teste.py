import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
import random
from scipy.interpolate import CubicSpline


class Corpo_Celeste:
    def __init__(self, massa, raio, pos_x, pos_y, color, name):
        self.massa = massa
        self.raio = raio
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.color = color
        self.name = name



class Universo:
    def __init__(self):
        self.softening = 1e6
        self.G = 6.67430e-11

        #se deixar no criar_plot como equal, datalim ele só ignora o limite menor. mas se usar o razao, box, ele vai considerar o limite menor tambem (porém o raio fica errado)
        self.limite_x = 2e11
        self.limite_y = 1e10
        
        self.criar_corpos_celestes()
    

    def criar_plot(self, ax):
        ax.set_xlim(-self.limite_x, self.limite_x)
        ax.set_ylim(-self.limite_y, self.limite_y)
        ax.set_facecolor('gray')
        ax.set_aspect('equal', adjustable='datalim')
        #esse de baixo deixa desproporcional e as esferas viram (), talvez desse pra usar elipse ao inves de circle pra ajustar com a razao tambem, não tenho certeza
        #razao = self.limite_x / self.limite_y
        #ax.set_aspect(razao, adjustable='box')


    def criar_corpos_celestes(self):
        self.corpos_celestes = []

        self.Terra = Corpo_Celeste(massa=5.972e24, raio=6.371e6, pos_x=0.0, pos_y=0.0, color='blue', name='Terra')
        self.corpos_celestes.append(self.Terra)
        self.Sol = Corpo_Celeste(massa=2e30, raio=6.957e8, pos_x=1.5e11, pos_y=0.0, color='yellow', name='Sol')
        self.corpos_celestes.append(self.Sol)
        self.Marte = Corpo_Celeste(massa=6.4e23, raio=3.389e6, pos_x=-2.25e8, pos_y=0.0, color='red', name='Marte')
        self.corpos_celestes.append(self.Marte)


    def posicionar_corpos_celestes(self, ax):
        #se quiser aumentar o raio **visualmente
        ganho_raio = 1
        terra = Circle((self.Terra.pos_x, self.Terra.pos_y), self.Terra.raio*ganho_raio, color=self.Terra.color, label='Terra (escala real)')
        ax.add_patch(terra)
        sol = Circle((self.Sol.pos_x, self.Sol.pos_y), self.Sol.raio*ganho_raio, color=self.Sol.color, label='Sol (escala real)')
        ax.add_patch(sol)
        marte = Circle((self.Marte.pos_x, self.Marte.pos_y), self.Marte.raio*ganho_raio, color=self.Marte.color, label='Marte (escala real)')
        ax.add_patch(marte)
        #ax.grid(True)


    #mesma coisa com nomes diferentes, fica a escolha de vocês
    def plotar_corpos_celestes(self, ax):
        #se quiser aumentar o raio **visualmente
        ganho_raio = 1
        for cc in self.corpos_celestes:
            my_cc = Circle((cc.pos_x, cc.pos_y), cc.raio*ganho_raio, color=cc.color, label=cc.name)
            ax.add_patch(my_cc)


    def plotar(self):
        fig, ax = plt.subplots(figsize=(8,8))
        #self.posicionar_corpos_celestes(ax)
        self.plotar_corpos_celestes(ax)
        self.criar_plot(ax)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    universo = Universo()
    universo.plotar()
