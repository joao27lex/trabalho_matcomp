import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify

# === 1. Define a função simbólica R(a, b) ===
a, b = symbols('a b')
R = (1 + a + (a**2 - b**2)/2)**2 + (b * (1 + a))**2

# === 2. Calcula gradiente de R: ∂R/∂a e ∂R/∂b ===
dR_da = diff(R, a)
dR_db = diff(R, b)

# === 3. Converte para funções numéricas com lambdify ===
f1 = lambdify((a, b), dR_da, 'numpy')  # ∂R/∂a
f2 = lambdify((a, b), dR_db, 'numpy')  # ∂R/∂b

# === 4. Função do campo vetorial tangente normalizado ===
def vetor_tangente(state):
    a_val, b_val = state
    grad = np.array([f1(a_val, b_val), f2(a_val, b_val)])
    tangente = np.array([grad[1], -grad[0]])  # ortogonal ao gradiente
    norm = np.linalg.norm(tangente)
    return (tangente / norm) * gamma if norm != 0 else np.zeros_like(tangente)

# === 5. Inicialização ===
state = np.array([0.0, 0.0])
dt = 1
gamma = 1e-2
trajetoria = [state.copy()]

# === 6. Integração com Runge-Kutta de 4ª ordem (RK4) ===
for _ in range(1000):
    k1 = vetor_tangente(state)
    k2 = vetor_tangente(state + 0.5 * dt * k1)
    k3 = vetor_tangente(state + 0.5 * dt * k2)
    k4 = vetor_tangente(state + dt * k3)
    state += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    trajetoria.append(state.copy())

# === 7. Plotagem ===
trajetoria = np.array(trajetoria)
plt.figure(figsize=(8, 6))
plt.plot(trajetoria[:, 0], trajetoria[:, 1], 'k.', markersize=2)
plt.title("Curva de Nível de R(a, b) com RK4 (Taylor 4ª ordem)")
plt.xlabel("a")
plt.ylabel("b")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
