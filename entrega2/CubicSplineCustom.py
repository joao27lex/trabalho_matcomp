import numpy as np


class CubicSplineCustom:
    """
    Spline cúbica natural 1D. Uso igual ao SciPy: cs = CubicSplineCustom(x,y); cs(x_new)
    """

    def __init__(self, x, y, **kwargs):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        n = self.x.size
        h = np.diff(self.x)
        # monta sistema tri-diagonal para M (2a deriv)
        A = np.zeros((n, n), dtype=float)
        b = np.zeros(n, dtype=float)
        A[0, 0] = A[-1, -1] = 1
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            b[i] = 6 * (
                (self.y[i + 1] - self.y[i]) / h[i]
                - (self.y[i] - self.y[i - 1]) / h[i - 1]
            )
        self.M = np.linalg.solve(A, b)

    def __call__(self, x_new):
        xv = np.atleast_1d(x_new)
        yv = np.empty_like(xv, dtype=float)
        for j, t in enumerate(xv):
            if t <= self.x[0]:
                i = 0
            elif t >= self.x[-1]:
                i = self.x.size - 2
            else:
                i = np.searchsorted(self.x, t) - 1
            x0, x1 = self.x[i], self.x[i + 1]
            y0, y1 = self.y[i], self.y[i + 1]
            M0, M1 = self.M[i], self.M[i + 1]
            h = x1 - x0
            A = (x1 - t) / h
            B = (t - x0) / h
            yv[j] = (
                M0 * (A**3) * (h**2) / 6
                + M1 * (B**3) * (h**2) / 6
                + (y0 - M0 * (h**2) / 6) * A
                + (y1 - M1 * (h**2) / 6) * B
            )
        return yv if yv.size > 1 else yv[0]
