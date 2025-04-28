# r4_sim_oop.py
# -------------------------------------------------
#  pip install simpy matplotlib pandas
# -------------------------------------------------
import simpy, numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass


# ------------------------------ 1. DATOS BÁSICOS ------------------------------
@dataclass
class Station:
    idx: int
    name: str
    EI: int
    FL: int
    IL: int
    PF: float                      # factor población

# 16 paradas del tramo Sant Vicenç → Cornellà
STATIONS = [
    Station( 0,"Sant Vicenç de Calders",1,0,1, 2259/88500),
    Station( 1,"El Vendrell",           0,0,0, 36230/88500),
    Station( 2,"L'Arboç",               0,0,0,  6524/88500),
    Station( 3,"Els Monjos",            0,0,0, 13800/88500),
    Station( 4,"Vilafranca del Penedès",1,0,0, 40000/88500),
    Station( 5,"La Granada",            0,0,0,  4000/88500),
    Station( 6,"La Pobla de Claramunt", 0,0,0,  9885/88500),
    Station( 7,"Sant Sadurní d'Anoia",  0,0,0, 11874/88500),
    Station( 8,"Gelida",                0,0,0,  8120/88500),
    Station( 9,"Martorell",             1,0,0, 26106/88500),
    Station(10,"Castellbisbal",         1,0,0, 12000/88500),
    Station(11,"El Papiol",             0,0,0,  7500/88500),
    Station(12,"Molins de Rei",         0,0,0, 26000/88500),
    Station(13,"Sant Feliu de Llobregat",0,0,0,46000/88500),
    Station(14,"Sant Joan Despí",       0,0,0, 27500/88500),
    Station(15,"Cornellà",              0,0,0, 88500/88500)
]

# ------------------------------ 2. MODELO DE TREN -----------------------------
class Train:
    # parámetros “hardware”
    CV, QV, P = 43, 6, 3       # plazas/vagón, vagones, puertas
    M         = CV*QV
    DT_DWELL  = 1.0
    SEG_TIME  = 4.0
    SIM_START = 8*60           # 08:00

    # coeficientes base
    alpha_F, alpha_EI, alpha_ES, alpha_IL = 0.015, 0.40, 0.50, 0.60
    beta_C,  beta_QP, beta_FL, beta_QV    = 0.02, 0.0015, 0.30, 0.002

    C = np.array([1,0,0])      # clima: sol

    @staticmethod
    def F_base(minuto):              # trenes/hora
        return 6 if 30<=minuto<90 else 2

    @staticmethod
    def commute_mult(t_abs):
        if 480<=t_abs<600:   return 1.3,1.0
        if 1020<=t_abs<1140: return 1.0,1.3
        return 1.0,1.0

    # tasas elementales
    def R_up(self,F,EI,IL):
        return self.alpha_F*F + self.alpha_EI*EI + self.alpha_IL*IL + self.alpha_ES*0
    def R_dn(self,EI,FL):
        clima = self.beta_C * (self.C @ np.arange(1,len(self.C)+1))
        return clima + self.beta_QP*(self.QV*self.P) + self.beta_FL*FL + self.beta_QV*self.QV


# ------------------------------ 3. SIMULADOR OOP ------------------------------
class LineSimulation:
    def __init__(self, stations, train:Train):
        self.stations = stations
        self.train    = train
        self.env      = simpy.Environment()
        self.occup    = simpy.Container(self.env, capacity=train.M, init=0)
        self.history  = []              # (min, station, O)

    def process_train(self, env):
        t = self.train
        for s in self.stations:
            if s.idx>0:
                yield env.timeout(t.SEG_TIME)   # viaje

            # tasas base
            t_abs = t.SIM_START + env.now
            F_t   = t.F_base(env.now)
            up    = t.R_up(F_t, s.EI, s.IL)*s.PF
            dn    = t.R_dn(s.EI, s.FL)*s.PF
            mu_u, mu_d = t.commute_mult(t_abs)
            up, dn = up*mu_u, dn*mu_d

            inflow  = min((t.M-self.occup.level)*up*t.DT_DWELL, t.P*t.QV*t.DT_DWELL)
            outflow = min(self.occup.level*dn*t.DT_DWELL, self.occup.level)

            if outflow: yield self.occup.get(outflow)
            if inflow:  yield self.occup.put(inflow)

            self.history.append((env.now, s.name, self.occup.level))
            yield env.timeout(t.DT_DWELL)

    def run(self):
        self.env.process(self.process_train(self.env))
        self.env.run(until=len(self.stations)*(self.train.SEG_TIME+self.train.DT_DWELL))
        return pd.DataFrame(self.history, columns=["min","station","O"])

# ------------------------------ 4. EJECUCIÓN Y PLOT ---------------------------
if __name__ == "__main__":
    train = Train()
    sim   = LineSimulation(STATIONS, train)
    df    = sim.run()

    x = np.arange(len(df))
    plt.figure(figsize=(12,5))
    plt.step(x, df["O"], where="post", marker="o")
    plt.xticks(x, df["station"], rotation=45, ha="right")
    plt.xlabel("Estaciones (Sant Vicenç → Cornellà)")
    plt.ylabel("Ocupación (personas)")
    plt.title("Ocupación por parada — modelo OOP")
    plt.grid(ls=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
