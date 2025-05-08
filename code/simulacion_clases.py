# 16 estaciones R4 (Sant Vicenç → Cornellà)
# -------------------------------------------------
#  pip install simpy matplotlib pandas
# -------------------------------------------------
import simpy, numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass
from ocupacio_real import ocupacio_inicial, calcular_ocupacion_real

# ------------------------------ 1. CONTEXTO Y CONFIGURACIÓN ------------------------------
@dataclass
class SimulationContext:
    start_time: int = 19*60  # minutos desde medianoche (ej: 8:00h)
    weather: str = "sunny"  # sunny, rain, cloudy
    special_tariff: bool = False
    special_event: bool = False

    peak_hours = [(7*60, 9*60), (17*60, 19*60)]  # 7-9h y 17-19h

    weather_factors = {
        "sunny": 1.0,
        "rain": 1.2,
        "cloudy": 1.1
    }

    def is_peak(self, absolute_minute):
        return any(start <= absolute_minute < end for start, end in self.peak_hours)

    def get_weather_factor(self):
        return self.weather_factors.get(self.weather, 1.0)

# ------------------------------ 2. ESTACIONES ------------------------------
@dataclass
class Station:
    idx: int
    name: str
    EI: int
    FL: int
    IL: int
    PF: float  # factor de población

# Lista de estaciones
STATIONS = [
    Station(0, "Sant Vicenç de Calders", 1, 0, 1, 2259/88500),
    Station(1, "El Vendrell",             0, 0, 0, 36230/88500),
    Station(2, "L'Arboç",                 0, 0, 0, 6524/88500),
    Station(3, "Els Monjos",              0, 0, 0, 13800/88500),
    Station(4, "Vilafranca del Penedès",  1, 0, 0, 40000/88500),
    Station(5, "La Granada",              0, 0, 0, 4000/88500),
    Station(6, "La Pobla de Claramunt",   0, 0, 0, 9885/88500),
    Station(7, "Sant Sadurní d'Anoia",    0, 0, 0, 11874/88500),
    Station(8, "Gelida",                  0, 0, 0, 8120/88500),
    Station(9, "Martorell",               1, 0, 0, 26106/88500),
    Station(10, "Castellbisbal",          1, 0, 0, 12000/88500),
    Station(11, "El Papiol",              0, 0, 0, 7500/88500),
    Station(12, "Molins de Rei",          0, 0, 0, 26000/88500),
    Station(13, "Sant Feliu de Llobregat",0, 0, 0, 46000/88500),
    Station(14, "Sant Joan Despí",        0, 0, 0, 27500/88500),
    Station(15, "Cornellà",               0, 0, 0, 88500/88500),
]

# Frecuencias por grupos de estaciones (minutos entre trenes)
FREQUENCIES = {
    (0, 3):   {"normal": 60, "peak": 30},
    (4, 8):   {"normal": 30, "peak": 30},
    (9, 11):  {"normal": 30, "peak": 15},
    (12, 15): {"normal": 15, "peak": 8}
}

def get_frequency(station_idx, is_peak):
    for (start, end), freq in FREQUENCIES.items():
        if start <= station_idx <= end:
            return freq["peak"] if is_peak else freq["normal"]
    return 60  # por defecto si no está mapeada

# ------------------------------ 3. MODELO DE TREN ------------------------------
class Train:
    CV, QV, P = 43, 6, 3        # plazas/vagón, vagones, puertas
    M = CV * QV                 # capacidad total
    DT_DWELL = 1.0              # tiempo de parada (min)
    SEG_TIME = 4.3              # tiempo entre estaciones (min)


    # coeficientes base
    alpha_F, alpha_EI, alpha_ES, alpha_IL, alpha_T, extra_alpha = 0.34606, 1.43527, 1.35071, 0.93269, 0.93269, 0.93269
    beta_C, beta_QP, beta_FL, beta_EI, extra_beta = 0.26119, 0.0, 0.1000, 0.67248, 0.11480

    def __init__(self, context: SimulationContext):
        self.context = context

    def R_up(self, F, EI, IL):
        return (self.alpha_F*F + self.alpha_EI*EI + self.alpha_IL*IL) * self.alpha_ES * self.alpha_T * self.extra_alpha

    def R_dn(self, EI, FL):
        clima_factor = self.beta_C * self.context.get_weather_factor()
        return (clima_factor + self.beta_QP*(self.QV*self.P) + self.beta_FL*FL + self.beta_EI*EI) * self.extra_beta

# ------------------------------ 4. SIMULADOR OOP ------------------------------
class LineSimulation:
    def __init__(self, stations, train: Train):
        self.stations = stations
        self.train = train
        self.env = simpy.Environment()
        self.occup = simpy.Container(self.env, capacity=train.M, init=ocupacio_inicial)
        self.history = []  # (min, station, O)

    def process_train(self, env):
        t = self.train
        ctx = t.context

        for i, station in enumerate(self.stations):
            if i > 0:
                yield env.timeout(t.SEG_TIME)

            current_time = ctx.start_time + env.now
            is_peak = ctx.is_peak(current_time)
            frequency = get_frequency(station.idx, is_peak)

            # calcular tasas
            up_rate = t.R_up((60/frequency), station.EI, station.IL) * station.PF # is min minutes between trains
            dn_rate = t.R_dn(station.EI, station.FL) * station.PF

            # modificar por hora punta
            up_rate *= 1.3 if is_peak else 1.0
            dn_rate *= 1.3 if is_peak else 1.0

            # flujos
            inflow = min((t.M - self.occup.level) * up_rate * t.DT_DWELL, t.P*t.QV*t.DT_DWELL)
            outflow = min(self.occup.level * dn_rate * t.DT_DWELL, self.occup.level)

            if outflow: yield self.occup.get(outflow)
            if inflow:  yield self.occup.put(inflow)

            self.history.append((env.now, station.name, self.occup.level))
            yield env.timeout(t.DT_DWELL)

    def run(self):
        self.env.process(self.process_train(self.env))
        # total_time = len(self.stations) * (self.train.SEG_TIME + self.train.DT_DWELL) + 10
        self.env.run()
        return pd.DataFrame(self.history, columns=["min", "station", "O"])

# ------------------------------ 5. EJECUCIÓN Y PLOT ------------------------------
if __name__ == "__main__":
    context = SimulationContext(
        start_time=8*60,       # 08:00
        weather="sunny",        # "sunny", "rain", "cloudy"
        special_tariff=False,   
        special_event=False
    )

    train = Train(context)
    sim = LineSimulation(STATIONS, train)
    df = sim.run()
    ocup_real = calcular_ocupacion_real()

    x = np.arange(len(df))
    plt.figure(figsize=(12,5))
    plt.step(x, df["O"], where="post", marker="o")
    plt.xticks(x, df["station"], rotation=45, ha="right")
    plt.xlabel("Estaciones (Sant Vicenç → Cornellà)")
    plt.ylabel("Ocupación (personas)")
    plt.title(f"Ocupación por parada — {context.weather.capitalize()} / Tarifa especial: {context.special_tariff} / Evento: {context.special_event}")
    plt.grid(ls=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    #graficar_resultados(df["0", ocup_real])
    estaciones = list(range(1, len(ocup_real) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(estaciones, ocup_real, marker="o", label="Ocupación real", linewidth=2)
    plt.plot(estaciones, df["O"], marker="s", label="Ocupación simulada", linewidth=2)
    plt.xlabel("Estaciones")
    plt.ylabel("Ocupación")
    plt.title("Comparación de ocupación real vs simulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------ 6. FUNCIÓN PARA IMPORTAR ------------------------------
def run_simulacion(alphas, betas):
    context = SimulationContext(
        start_time=8*60,
        weather="sun",
        special_tariff=False,   
        special_event=False
    )

    train = Train(context)

    train.alpha_F, train.alpha_EI, train.alpha_IL, train.alpha_ES, train.alpha_T, train.extra_alpha = alphas
    train.beta_C, train.beta_QP, train.beta_FL, train.beta_EI, train.extra_beta = betas

    try:
        sim = LineSimulation(STATIONS, train)
        df = sim.run()
        ocupaciones = df["O"].tolist()

        
        if len(ocupaciones) != len(STATIONS):
            raise ValueError(f"Se esperaban {len(STATIONS)} ocupaciones, pero se obtuvieron {len(ocupaciones)}")

        return ocupaciones
    except Exception as e:
        print("Fin de la simulación:", e)
        return [1e6] * len(STATIONS)
