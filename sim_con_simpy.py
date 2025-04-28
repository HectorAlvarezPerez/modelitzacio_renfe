# 16 estaciones R4 (Sant Vicenç → Cornellà)
# -----------------------------------------------------------------
#  pip install simpy matplotlib pandas
# -----------------------------------------------------------------
import simpy, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------- 1. PARÁMETROS GLOBALES ----------
DT_DWELL   = 1.0          # min parado en estación
SEG_TIME   = 4            # min entre estaciones (promedio)

CV, QV, P  = 43, 6, 3    # plazas/vagón, nº vagones, puertas/vagón tren serie 449
M = CV * QV               # capacidad total tren

ES, TA = 0, 0             # sin eventos, tarifa normal
C = np.array([1, 0, 0])   # clima: sol (one-hot)

# Coeficientes base
alpha_F, alpha_EI, alpha_ES, alpha_IL = 0.015, 0.40, 0.50, 0.60
beta_C,  beta_QP, beta_FL, beta_QV    = 0.02,  0.0015, 0.30, 0.002

# Inicio de simulación a las 08:00 h (env.now=0 => 08:00)
SIM_START = 8 * 60  # minutos desde medianoche

# ---------- 2. FUNCIONES AUXILIARES ----------
def F_base(minuto):
    """Frecuencia trenes/hora (R4 típica: 6 en punta, 4 fuera)."""
    return 6 if 30 <= minuto < 90 else 2

def commute_multiplier(t_abs):
    """Factor horas punta según minuto desde medianoche."""
    if 480 <= t_abs < 600:    # 08:00–10:00 (ida)
        return 1.3, 1.0
    if 1020 <= t_abs < 1140:  # 17:00–19:00 (vuelta)
        return 1.0, 1.3
    return 1.0, 1.0

def R_pujada(F, EI, IL):
    return alpha_F*F + alpha_EI*EI + alpha_IL*IL + alpha_ES*ES


def R_baixada(EI, FL):
    clima = beta_C * (C @ np.arange(1, len(C)+1))
    raw   = clima + beta_QP*(QV*P) + beta_FL*FL + beta_QV*QV
    return raw

# ---------- 3. LISTA COMPLETA DE ESTACIONES R4 ----------

STATIONS = [
    # idx, nombre,                     EI, FL, IL,   PF (pop/pop_max)
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



# ---------- 4. PROCESO SIMPY: tren ----------
def tren(env, ocup, hist):
    for idx, name, EI, FL, IL, PF in STATIONS:
        # viaje si no es la primera estación
        if idx > 0:
            yield env.timeout(SEG_TIME)

        # parada: calculo tasas
        t_abs = SIM_START + env.now
        F_t   = F_base(env.now)

        base_up   = R_pujada(F_t, EI, IL)
        base_dn   = R_baixada(EI, FL)

        # ajusta por población
        base_up  *= PF
        base_dn  *= PF

        # ajusta por horas punta
        mu_up, mu_dn = commute_multiplier(t_abs)
        rup = base_up * mu_up
        rdn = base_dn * mu_dn

        # flujos en DT_DWELL
        inflow  = (ocup.capacity - ocup.level) * rup * DT_DWELL
        outflow = ocup.level * rdn * DT_DWELL

        if outflow > 0:
            yield ocup.get(min(outflow, ocup.level))
        free = ocup.capacity - ocup.level
        if inflow > 0:
            yield ocup.put(min(inflow, free))

        # log y dwell-time
        hist.append((env.now, name, ocup.level))
        yield env.timeout(DT_DWELL)


# ---------- 5. EJECUCIÓN ----------
env   = simpy.Environment()
ocup  = simpy.Container(env, capacity=M, init=0)
hist  = []
env.process(tren(env, ocup, hist))
# ejecuta un solo viaje completo a través de todas las estaciones:
env.run(until=(len(STATIONS)*(SEG_TIME+DT_DWELL)))

# ---------- 6. VISUALIZACIÓN por estación ----------
df = pd.DataFrame(hist, columns=["min", "estacion", "O"])
x_labels = df["estacion"]
x_pos    = range(len(x_labels))

plt.figure(figsize=(12, 5))
plt.step(x_pos, df["O"], where="post", lw=2, marker="o")
plt.xticks(x_pos, x_labels, rotation=45, ha="right")
plt.xlabel("Estaciones (Sant Vicenç → Cornellà)")
plt.ylabel("Ocupación (personas)")
plt.title("Ocupación por parada en todo el tramo R4 Sur ")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
