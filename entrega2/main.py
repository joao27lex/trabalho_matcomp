import numpy as np
import pandas as pd
from CubicSplineCustom import CubicSplineCustom
from Universo import Universo

# === UTILITÁRIOS MATEMÁTICOS ===


def prever_colisao_pela_spline(
    spline_x, spline_y, raio_terra, terra_x0, terra_y0, duracao_total, n_amostras=500
):
    tempos = np.linspace(0, duracao_total, n_amostras)
    for t in tempos:
        x_ast = spline_x(t)
        y_ast = spline_y(t)
        dx = x_ast - terra_x0
        dy = y_ast - terra_y0
        dist = np.hypot(dx, dy)
        if dist <= 3 * raio_terra:
            print(f"[ALERTA] Colisão prevista em t={t:.2e}, distância={dist:.2e}")
            return True
    return False


def refinar_tempo_interceptacao_newton(
    spline_x, spline_y, v_proj, p0_x, p0_y, t0, max_iter=20, tol=1e-3
):
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


# === SIMULAÇÕES ===


def simular_projetil_com_newton(
    posicao_x_asteroide,
    posicao_y_asteroide,
    tempos_asteroide,
    velocidade_projetil,
    po_x,
    po_y,
    spline_x,
    spline_y,
):
    raio_do_planeta = 6.371e6
    tempo_impacto = None
    for i in range(len(posicao_x_asteroide)):
        dist = np.hypot(posicao_x_asteroide[i] - po_x, posicao_y_asteroide[i] - po_y)
        if dist <= raio_do_planeta:
            tempo_impacto = tempos_asteroide[i]
            break
    if tempo_impacto is None:
        tempo_impacto = tempos_asteroide[-1]

    melhor_ponto = None
    melhor_t = None
    menor_erro = float("inf")

    for i in range(len(tempos_asteroide)):
        t_ast = tempos_asteroide[i]
        if t_ast >= tempo_impacto:
            break
        x_ast = posicao_x_asteroide[i]
        y_ast = posicao_y_asteroide[i]
        d = np.hypot(x_ast - po_x, y_ast - po_y)
        tempo_projetil = d / velocidade_projetil
        erro = abs(tempo_projetil - t_ast)
        if erro < menor_erro:
            menor_erro = erro
            melhor_ponto = (x_ast, y_ast)
            melhor_t = t_ast

    if melhor_ponto is None:
        print("[ALERTA] Nenhuma interceptação possível antes da colisão.")
        return None

    t_refinado = refinar_tempo_interceptacao_newton(
        spline_x, spline_y, velocidade_projetil, po_x, po_y, melhor_t
    )

    if not np.isfinite(t_refinado):
        print("[ERRO] Tempo refinado inválido (NaN ou infinito). Abortando projétil.")
        return None

    x_ast = spline_x(t_refinado)
    y_ast = spline_y(t_refinado)
    direcao = np.array([x_ast - po_x, y_ast - po_y])
    dist = np.linalg.norm(direcao)

    if dist == 0:
        print("[ERRO] Vetor de direção nulo na interceptação. Abortando projétil.")
        return None

    t_proj = np.linspace(0, t_refinado, 200)
    direcao_unitaria = direcao / dist
    x_proj = po_x + velocidade_projetil * direcao_unitaria[0] * t_proj
    y_proj = po_y + velocidade_projetil * direcao_unitaria[1] * t_proj

    return pd.DataFrame({"x_proj": x_proj, "y_proj": y_proj, "tempo": t_proj})


def simular_sonda_com_atuadores_limitados(
    spline_x, spline_y, t_max, dt=10, v_inicial=3e5, a_max=50, po_x=0, po_y=0
):
    tempos = np.arange(0, t_max, dt)
    x_sonda = [po_x]
    y_sonda = [po_y]
    vx_sonda = [v_inicial]
    vy_sonda = [0]

    for i in range(1, len(tempos)):
        t = tempos[i]
        x = x_sonda[-1]
        y = y_sonda[-1]
        vx = vx_sonda[-1]
        vy = vy_sonda[-1]
        x_alvo = spline_x(t)
        y_alvo = spline_y(t)
        dx = x_alvo - x
        dy = y_alvo - y
        distancia = np.hypot(dx, dy)
        if distancia == 0:
            ax = ay = 0
        else:
            fator_proporcional = min(1.0, distancia / 1e9)
            ax = a_max * (dx / distancia) * fator_proporcional
            ay = a_max * (dy / distancia) * fator_proporcional

        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx * dt
        y_new = y + vy * dt

        x_sonda.append(x_new)
        y_sonda.append(y_new)
        vx_sonda.append(vx_new)
        vy_sonda.append(vy_new)

    return pd.DataFrame({"tempo": tempos, "x_sonda": x_sonda, "y_sonda": y_sonda})


# === EXECUÇÃO PRINCIPAL ===

if __name__ == "__main__":
    universo = Universo()
    solucao = universo.simular()

    terra = next(c for c in universo.corpos_celestes if c.name == "Terra")
    terra_x0, terra_y0 = terra.trace[0]

    asteroide = next(c for c in universo.corpos_celestes if c.name == "Asteroide")
    trajetoria_x = [p[0] for p in asteroide.trace]
    trajetoria_y = [p[1] for p in asteroide.trace]
    lista_tempos = np.linspace(0, universo.duracao, len(trajetoria_x))

    spline_real_x = CubicSplineCustom(lista_tempos, trajetoria_x)
    spline_real_y = CubicSplineCustom(lista_tempos, trajetoria_y)

    df_sonda_limitada = simular_sonda_com_atuadores_limitados(
        spline_x=spline_real_x,
        spline_y=spline_real_y,
        t_max=universo.duracao,
        dt=150,
        v_inicial=3e6,
        a_max=300,
        po_x=terra_x0,
        po_y=terra_y0,
    )

    posicao_x_sensor = df_sonda_limitada["x_sonda"].tolist()
    posicao_y_sensor = df_sonda_limitada["y_sonda"].tolist()
    tempos_sensor = df_sonda_limitada["tempo"].tolist()

    spline_estimada_x = CubicSplineCustom(tempos_sensor, posicao_x_sensor)
    spline_estimada_y = CubicSplineCustom(tempos_sensor, posicao_y_sensor)

    colisao_prevista = prever_colisao_pela_spline(
        spline_x=spline_estimada_x,
        spline_y=spline_estimada_y,
        raio_terra=terra.raio,
        terra_x0=terra_x0,
        terra_y0=terra_y0,
        duracao_total=universo.duracao,
    )

    df_trajetoria_projetil = None
    if colisao_prevista:
        df_trajetoria_projetil = simular_projetil_com_newton(
            posicao_x_asteroide=posicao_x_sensor,
            posicao_y_asteroide=posicao_y_sensor,
            tempos_asteroide=tempos_sensor,
            velocidade_projetil=3e6,
            po_x=terra_x0,
            po_y=terra_y0,
            spline_x=spline_estimada_x,
            spline_y=spline_estimada_y,
        )

    if df_trajetoria_projetil is not None:
        universo.animar(
            solucao,
            df_trajetoria_projetil=df_trajetoria_projetil,
            df_sonda_limitada=df_sonda_limitada,
        )
    else:
        print("[INFO] Projétil não foi lançado. Animação sem trajetória de interceptação.")
        universo.animar(solucao, df_sonda_limitada=df_sonda_limitada)
