import numpy as np


class RK4Solver:
    class Solution:
        def __init__(self, t, y):
            self.t = t
            self.y = y

    def __init__(self, fun, t_span, y0, t_eval=None):
        self.fun = fun  # função do sistema: f(t, y)
        self.t0, self.tf = t_span  # intervalo de tempo
        self.y0 = np.array(y0, dtype=float)  # condição inicial
        self.t_eval = (
            np.linspace(self.t0, self.tf, 100)
            if t_eval is None
            else np.array(t_eval, dtype=float)
        )

    def solve(self):
        n = self.t_eval.size
        y = np.zeros((self.y0.size, n), dtype=float)
        y[:, 0] = self.y0

        for i in range(1, n):
            ti, tf_i = self.t_eval[i - 1], self.t_eval[i]
            h = tf_i - ti
            yi = y[:, i - 1]
            k1 = np.array(self.fun(ti, yi))
            k2 = np.array(self.fun(ti + h / 2, yi + h * k1 / 2))
            k3 = np.array(self.fun(ti + h / 2, yi + h * k2 / 2))
            k4 = np.array(self.fun(tf_i, yi + h * k3))
            y[:, i] = yi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.Solution(self.t_eval, y)
