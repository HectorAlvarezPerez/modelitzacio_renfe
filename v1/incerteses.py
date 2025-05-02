import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------------------------------------------------------
#  Análisis de incertidumbre por Monte Carlo para R4 Sur (16 estaciones)
# ------------------------------------------------------------------

# ---------- 1. Definición de estaciones y parámetros nominales ----------
STATIONS = [
    # idx, nombre,                    EI, FL, IL,   PF (pop_ratio)
    ( 0, "Sant Vicenç de Calders",      1, 0, 1,   2259/88500),   # ≈0.03
    ( 1, "El Vendrell",                 0, 0, 0,   36230/88500),  # ≈0.41
    ( 2, "L'Arboç",                     0, 0, 0,    6524/88500),  # ≈0.07
    ( 3, "Els Monjos",                  0, 0, 0,   13800/88500),  # ≈0.16
    ( 4, "Vilafranca del Penedès",      1, 0, 0,   40000/88500),  # ≈0.45
    ( 5, "La Granada",                  0, 0, 0,    4000/88500),  # ≈0.05
    ( 6, "La Pobla de Claramunt",       0, 0, 0,    9885/88500),  # ≈0.11
    ( 7, "Sant Sadurní d'Anoia",        0, 0, 0,   11874/88500),  # ≈0.13
    ( 8, "Gelida",                      0, 0, 0,    8120/88500),  # ≈0.09
    ( 9, "Martorell",                   1, 0, 0,   26106/88500),  # ≈0.29
    (10, "Castellbisbal",               1, 0, 0,   12000/88500),  # ≈0.14
    (11, "El Papiol",                   0, 0, 0,    7500/88500),  # ≈0.08
    (12, "Molins de Rei",               0, 0, 0,   26000/88500),  # ≈0.29
    (13, "Sant Feliu de Llobregat",     0, 0, 0,   46000/88500),  # ≈0.52
    (14, "Sant Joan Despí",             0, 0, 0,   27500/88500),  # ≈0.31
    (15, "Cornellà",                    0, 0, 0,   88500/88500)   # =1.00
]

CV, QV, P    = 43, 6, 3                   # plazas/vagón, vagones, puertas
M             = CV * QV
DT_DWELL      = 1.0
SEG_TIME      = 4.0
SIM_START     = 8 * 60

nominal = {
    'alpha_F':  0.015,
    'alpha_EI': 0.40,
    'alpha_IL': 0.60,
    'alpha_ES': 0.50,
    'beta_QP':  0.0015,
    'beta_FL':  0.30,
    'beta_QV':  0.002
}

def F_base(minuto):
    return 6 if 30 <= minuto < 90 else 4

def commute_multiplier(t_abs):
    if 480 <= t_abs < 600:    return 1.3, 1.0
    if 1020 <= t_abs < 1140:  return 1.0, 1.3
    return 1.0, 1.0

def run_once(params):
    aF, aEI, aIL, aES = params['alpha_F'], params['alpha_EI'], params['alpha_IL'], params['alpha_ES']
    bQP, bFL, bQV     = params['beta_QP'], params['beta_FL'], params['beta_QV']

    def R_pujada(F, EI, IL):
        return max(0, aF*F + aEI*EI + aIL*IL + aES*0)
    def R_baixada(EI, FL):
        return max(0, bQP*(QV*P) + bFL*FL + bQV*QV)

    env  = simpy.Environment()
    ocup = simpy.Container(env, capacity=M, init=0)
    hist = []

    def tren(env):
        for idx, name, EI, FL, IL, PF in STATIONS:
            if idx > 0:
                yield env.timeout(SEG_TIME)
            t_abs = SIM_START + env.now
            F_t   = F_base(env.now)

            up_base  = min(R_pujada(F_t, EI, IL)*PF, P*QV)
            dn_base  = R_baixada(EI, FL)*PF
            mu_up, mu_dn = commute_multiplier(t_abs)
            rup, rdn = up_base*mu_up, dn_base*mu_dn

            inflow  = min((M-ocup.level)*rup*DT_DWELL, P*QV*DT_DWELL)
            outflow = min(ocup.level*rdn*DT_DWELL, ocup.level)

            if outflow>0: yield ocup.get(outflow)
            if inflow>0:  yield ocup.put(inflow)
            hist.append((env.now, name, ocup.level))
            yield env.timeout(DT_DWELL)

    env.process(tren(env))
    env.run(until=len(STATIONS)*(SEG_TIME+DT_DWELL))
    return [level for (_,_, level) in hist]

# ---------- 2. Distribuciones de incertidumbre ----------
distros = {k: norm(loc=nominal[k], scale=0.2*nominal[k]) for k in nominal}

# ---------- 3. Monte Carlo ----------
Nsim    = 10000
results = np.zeros((Nsim, len(STATIONS)))

for i in range(Nsim):
    sample = {k: distros[k].rvs() for k in distros}
    results[i, :] = run_once(sample)

# ---------- 4. Estadísticas ----------
# ← Aquí definimos station_names **antes** de usarlo:
station_names = [s[1] for s in STATIONS]

df_stats = pd.DataFrame({
    'station': station_names,
    'mean':    results.mean(axis=0),
    'std':     results.std(axis=0),
    'p2.5':    np.percentile(results, 2.5, axis=0),
    'p97.5':   np.percentile(results,97.5,axis=0),
    'IC':       np.percentile(results, 97.5, axis=0) - np.percentile(results, 2.5, axis=0), # 95% confidence interval
    'std/mean': results.std(axis=0) / results.mean(axis=0) *100
})

print(df_stats)

# ---------- 5. Gráfica de incertidumbre ----------
x = np.arange(len(station_names))
plt.figure(figsize=(12,5))
plt.plot(x, df_stats['mean'], lw=2, label='Media')
plt.fill_between(x,
                 df_stats['p2.5'],
                 df_stats['p97.5'],
                 color='gray', alpha=0.3, label='95% CI')
plt.xticks(x, station_names, rotation=45, ha='right')
plt.ylabel("Ocupación (personas)")
plt.title("Propagación de incertidumbre en ocupación por estación")
plt.legend()
plt.tight_layout()
plt.show()
