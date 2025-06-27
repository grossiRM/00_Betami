import numpy as np

def analytical_solution(z, t, dh=1.0, b0=1.0, ssk=100.0, vk=0.025, n=100, silent=True):
    v = 0.0  ; e = np.exp(1)  ; pi = np.pi ; pi2 = np.pi**2
    for k in range(n):
        fk = float(k) ; tauk = (0.5 * b0) ** 2.0 * ssk / ((2.0 * fk + 1.0) ** 2.0 * vk)
        ep = ((2.0 * fk + 1.0) ** 2 * pi2 * vk * t) / (4.0 * ssk * (0.5 * b0) ** 2.0)
        rad = (2.0 * fk + 1.0) * pi * z / b0             ;  v += ((-1.0) ** fk / (2.0 * fk + 1.0)) * (e**-ep) * np.cos(rad)
        if not silent: print(f"{k:5d} {tauk:20g} {rad:20g} {v:20g}")
    return dh - 4.0 * dh * v / pi

# to extract ... _______________________________________________________________________________ 
sys.path.append("E:/15_REPOS/00_BETAMI/bin/Flopy_functions")     # analytical ...
from Analitycal_SUB_01 import analytical_solution
cc1 = []  ; nz = 100 ; thick = 1.0 ; 
kv = parameters[list(parameters.keys())[0]]["kv"][0] ; dhalf = thick * 0.5  ; az = np.linspace(-dhalf, dhalf, num=nz)  ; dz = az[1] - az[0]
for tt in cobs["totim"]: 
    cc2 = 0.0
    for jdx, zz in enumerate(az): 
        f = 1.0
        if jdx == 0 or jdx == nz - 1:             f = 0.5
        h = analytical_solution(zz, tt, ssk=skv, vk=kv, n=200, dh=1.0) ;         cc2 += h * skv * f * dz
    cc1.append(cc2)
cc1 = np.array(cc1)