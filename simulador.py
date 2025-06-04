import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import pandas as pd

# Constantes
G = 6.67430e-11              # constante gravitacional (m^3 kg^-1 s^-2)
epsilon = 1e6                # tolerância de impacto (m)
massa_planeta = 5.972e24     # massa do planeta (kg)
tempo_final = 100000         # duração da simulação (s)
tempos_amostragem = np.linspace(0, tempo_final, 2000)

# Função do campo gravitacional com variáveis renomeadas
def campo_gravitacional(tempo, estado, massa_central):
    pos_x, pos_y, vel_x, vel_y = estado
    raio2 = pos_x**2 + pos_y**2
    raio_soft3 = (raio2 + epsilon**2)**1.5
    aceleracao_x = -G * massa_central * pos_x / raio_soft3
    aceleracao_y = -G * massa_central * pos_y / raio_soft3
    return [vel_x, vel_y, aceleracao_x, aceleracao_y]

# Simulação do asteroide
pos_inicial_ast = [7e6, 0]
vel_inicial_ast = [-5000, 9000]
estado_inicial_ast = np.concatenate((pos_inicial_ast, vel_inicial_ast))

sol_ast = solve_ivp(
    campo_gravitacional, (0, tempo_final), estado_inicial_ast,
    t_eval=tempos_amostragem, args=(massa_planeta,), rtol=1e-6, atol=1e-9
)

# Simulação do projétil
pos_inicial_proj = [0, 0]
vel_inicial_proj = [12000, 12000]
estado_inicial_proj = np.concatenate((pos_inicial_proj, vel_inicial_proj))

sol_proj = solve_ivp(
    campo_gravitacional, (0, tempo_final), estado_inicial_proj,
    t_eval=tempos_amostragem, args=(massa_planeta,), rtol=1e-6, atol=1e-9
)

# Interpolação com splines
spline_ast_x = CubicSpline(sol_ast.t, sol_ast.y[0])
spline_ast_y = CubicSpline(sol_ast.t, sol_ast.y[1])
spline_proj_x = CubicSpline(sol_proj.t, sol_proj.y[0])
spline_proj_y = CubicSpline(sol_proj.t, sol_proj.y[1])

# Função distância para minimizar
def distancia_entre_corpos(tempo):
    pos_ast = np.array([spline_ast_x(tempo), spline_ast_y(tempo)])
    pos_proj = np.array([spline_proj_x(tempo), spline_proj_y(tempo)])
    return np.linalg.norm(pos_ast - pos_proj)

# Encontrar tempo de interceptação via minimização da distância
resultado = minimize_scalar(distancia_entre_corpos, bounds=(0, tempo_final), method='bounded')
tempo_intercept = resultado.x

# Posições dos corpos no instante de interceptação
pos_ast_intercept = np.array([spline_ast_x(tempo_intercept), spline_ast_y(tempo_intercept)])
pos_proj_intercept = np.array([spline_proj_x(tempo_intercept), spline_proj_y(tempo_intercept)])
distancia_intercept = np.linalg.norm(pos_ast_intercept - pos_proj_intercept)

# Montar resultados em tabela
tabela_resultado = pd.DataFrame({
    "Tempo de interceptação (s)": [tempo_intercept],
    "Posição asteroide (x, y) [m]": [pos_ast_intercept],
    "Posição projétil (x, y) [m]": [pos_proj_intercept],
    "Distância entre corpos (m)": [distancia_intercept]
})

#import ace_tools as tools; tools.display_dataframe_to_user(name="Interceptação por Mínimo de Distância", dataframe=tabela_resultado)
