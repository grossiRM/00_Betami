import numpy as np

def analytical_solution(z, t, dh=1.0, b0=1.0, ssk=100.0, vk=0.025, n=100, silent=True):
    v = 0.0  ; e = np.exp(1)  ; pi = np.pi ; pi2 = np.pi**2
    for k in range(n):
        fk = float(k) ; tauk = (0.5 * b0) ** 2.0 * ssk / ((2.0 * fk + 1.0) ** 2.0 * vk)
        ep = ((2.0 * fk + 1.0) ** 2 * pi2 * vk * t) / (4.0 * ssk * (0.5 * b0) ** 2.0)
        rad = (2.0 * fk + 1.0) * pi * z / b0             ;  v += ((-1.0) ** fk / (2.0 * fk + 1.0)) * (e**-ep) * np.cos(rad)
        if not silent: print(f"{k:5d} {tauk:20g} {rad:20g} {v:20g}")
    return dh - 4.0 * dh * v / pi