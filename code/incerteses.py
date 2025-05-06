import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass

# ------------------------------------------------------------------
#  Análisis de incertidumbre por Monte Carlo para R4 Sur (16 estaciones)
# ------------------------------------------------------------------

# ------------------------------ 1. CONTEXTO Y CONFIGURACIÓN ------------------------------
@dataclass
class SimulationContext:
    start_time: int = 8*60  # minutos desde medianoche (ej: 8:00h)
    weather: str = "sunny"  # sunny, rain, cloudy

    weather_factors = {
        "sunny": 1.0,
        "rain": 1.2,
        "cloudy": 1.1
    }
    
    peak_hours = [(7*60, 9*60), (17*60, 19*60)]  # 7-9h y 17-19h
    
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
    Station(0, "Sant Vicenç de Calders", 1, 0, 1, 0.03),
    Station(1, "El Vendrell",             0, 0, 0, 0.41),
    Station(2, "L'Arboç",                 0, 0, 0, 0.07),
    Station(3, "Els Monjos",              0, 0, 0, 0.16),
    Station(4, "Vilafranca del Penedès",  1, 0, 0, 0.45),
    Station(5, "La Granada",              0, 0, 0, 0.05),
    Station(6, "La Pobla de Claramunt",   0, 0, 0, 0.11),
    Station(7, "Sant Sadurní d'Anoia",    0, 0, 0, 0.13),
    Station(8, "Gelida",                  0, 0, 0, 0.09),
    Station(9, "Martorell",               1, 0, 0, 0.29),
    Station(10, "Castellbisbal",          1, 0, 0, 0.14),
    Station(11, "El Papiol",              0, 0, 0, 0.08),
    Station(12, "Molins de Rei",          0, 0, 0, 0.29),
    Station(13, "Sant Feliu de Llobregat",0, 0, 0, 0.52),
    Station(14, "Sant Joan Despí",        0, 0, 0, 0.31),
    Station(15, "Cornellà",               0, 0, 0, 1.00),
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
    SEG_TIME = 4.0              # tiempo entre estaciones (min)
    
    def __init__(self, params=None):
        # Usar los parámetros proporcionados
        self.alpha_F = params['alpha_F']
        self.alpha_EI = params['alpha_EI']
        self.alpha_IL = params['alpha_IL']
        self.alpha_ES = params['alpha_ES']
        self.alpha_T = params['alpha_T']
        self.extra_alpha = params['extra_alpha']
        self.beta_C = params['beta_C']
        self.beta_QP = params['beta_QP']
        self.beta_FL = params['beta_FL']
        self.beta_EI = params['beta_EI']
        self.extra_beta = params['extra_beta']

    def R_up(self, F, EI, IL):
        return max(0, (self.alpha_F*F + self.alpha_EI*EI + self.alpha_IL*IL) * self.alpha_ES * self.alpha_T * self.extra_alpha)

    def R_dn(self, EI, FL, weather_factor):
        clima_factor = self.beta_C * weather_factor
        return max(0, (clima_factor + self.beta_QP*(self.QV*self.P) + self.beta_FL*FL + self.beta_EI*EI) * self.extra_beta)

# ------------------------------ 4. SIMULADOR PARA MONTE CARLO ------------------------------
class MonteCarloSimulation:
    def __init__(self, stations, params=None):
        self.stations = stations
        self.train = Train(params)
        self.context = SimulationContext()

    def run_once(self):
        env = simpy.Environment()
        occup = simpy.Container(env, capacity=self.train.M, init=0)
        history = []

        def process_train(env):
            # Aseguramos que todas las estaciones se procesen
            for i, station in enumerate(self.stations):
                try:
                    if station.idx > 0:
                        yield env.timeout(self.train.SEG_TIME)
                    
                    current_time = self.context.start_time + env.now
                    is_peak = self.context.is_peak(current_time)
                    freq = get_frequency(station.idx, is_peak)
                    
                    # Calcular tasas como en simulacion_clases.py
                    up_rate = self.train.R_up((60/freq), station.EI, station.IL) * station.PF
                    dn_rate = self.train.R_dn(station.EI, station.FL, self.context.get_weather_factor()) * station.PF
                    
                    # Modificar por hora punta
                    up_mult, dn_mult = 1.0, 1.0
                    if is_peak:
                        up_mult, dn_mult = 1.3, 1.3
                    
                    up_rate *= up_mult
                    dn_rate *= dn_mult
                    
                    # Calcular flujos
                    inflow = min((self.train.M - occup.level) * up_rate * self.train.DT_DWELL, 
                                self.train.P * self.train.QV * self.train.DT_DWELL)
                    outflow = min(occup.level * dn_rate * self.train.DT_DWELL, occup.level)
                    
                    if outflow > 0:
                        yield occup.get(outflow)
                    if inflow > 0:
                        yield occup.put(inflow)
                    
                    history.append((env.now, station.name, occup.level))
                    yield env.timeout(self.train.DT_DWELL)
                except Exception as e:
                    print(f"Error en estación {station.name}: {e}")
                    # Asegurarse de que esta estación aparezca en el historial
                    history.append((env.now, station.name, occup.level if hasattr(occup, 'level') else 0))
        
        env.process(process_train(env))
        env.run(until=len(self.stations)*(self.train.SEG_TIME + self.train.DT_DWELL))
        
        # Verificar que todas las estaciones estén en el historial
        if len(history) != len(self.stations):
            print(f"ADVERTENCIA: El historial tiene {len(history)} registros pero hay {len(self.stations)} estaciones")
            # Si falta alguna estación, completamos con el último valor conocido o cero
            if len(history) < len(self.stations):
                last_level = history[-1][2] if history else 0
                for i in range(len(history), len(self.stations)):
                    station_missing = self.stations[i]
                    history.append((env.now + i, station_missing.name, last_level))
        
        return [level for (_, _, level) in history]

# ------------------------------ 5. ANÁLISIS DE MONTE CARLO ------------------------------
def run_monte_carlo_analysis(n_sims=10000):
    # Definir parámetros nominales basados en el modelo de simulacion_clases.py
    nominal = {
        'alpha_F': 0.96819,
        'alpha_EI': 0.31339,
        'alpha_IL': 0.22120,
        'alpha_ES': 1.06358,
        'alpha_T': 1.06358,
        'extra_alpha': 1.06358,
        'beta_C': 0.08999,
        'beta_QP': 0.01902,
        'beta_FL': 0.10000,
        'beta_EI': 0.29517,
        'extra_beta': 0.11893
    }
    
    # Crear distribuciones de incertidumbre (20% de variación)
    distros = {k: norm(loc=nominal[k], scale=0.2*nominal[k]) for k in nominal}
    
    # Ejecutar simulaciones de Monte Carlo
    results = np.zeros((n_sims, len(STATIONS)))
    
    print(f"Ejecutando {n_sims} simulaciones de Monte Carlo...")
    for i in range(n_sims):
        if i % 1000 == 0 and i > 0:
            print(f"Completadas {i} simulaciones...")
        
        # Muestrear parámetros de las distribuciones
        sample = {k: distros[k].rvs() for k in distros}
        
        # Ejecutar simulación con estos parámetros
        sim = MonteCarloSimulation(STATIONS, sample)
        result_values = sim.run_once()
        
        # Verificar que la dimensión coincide
        if len(result_values) != len(STATIONS):
            print(f"ADVERTENCIA: La simulación devolvió {len(result_values)} valores pero hay {len(STATIONS)} estaciones")
            # Rellenar con ceros si falta algún valor
            if len(result_values) < len(STATIONS):
                result_values = np.pad(result_values, (0, len(STATIONS) - len(result_values)), 'constant')
            else:
                result_values = result_values[:len(STATIONS)]
        
        results[i, :] = result_values
    
    # Calcular estadísticas
    station_names = [s.name for s in STATIONS]
    df_stats = pd.DataFrame({
        'station': station_names,
        'mean': results.mean(axis=0),
        'std': results.std(axis=0),
        'p2.5': np.percentile(results, 2.5, axis=0),
        'p97.5': np.percentile(results, 97.5, axis=0),
        'IC': np.percentile(results, 97.5, axis=0) - np.percentile(results, 2.5, axis=0),
        'std/mean': results.std(axis=0) / results.mean(axis=0) * 100
    })
    
    return df_stats, results

# ------------------------------ 6. VISUALIZACIÓN ------------------------------
def plot_uncertainty(df_stats):
    x = np.arange(len(df_stats))
    plt.figure(figsize=(12, 5))
    plt.plot(x, df_stats['mean'], lw=2, label='Media')
    plt.fill_between(x,
                     df_stats['p2.5'],
                     df_stats['p97.5'],
                     color='gray', alpha=0.3, label='95% CI')
    plt.xticks(x, df_stats['station'], rotation=45, ha='right')
    plt.ylabel("Ocupación (personas)")
    plt.title("Propagación de incertidumbre en ocupación por estación")
    plt.legend()
    plt.grid(ls=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ------------------------------ 7. EJECUCIÓN PRINCIPAL ------------------------------
if __name__ == "__main__":
    # Ejecutar análisis de Monte Carlo con menos simulaciones para pruebas
    df_stats, results = run_monte_carlo_analysis(n_sims=100)
    print(df_stats)
    plot_uncertainty(df_stats)
    
    # Para análisis completo, usar más simulaciones
    # df_stats, results = run_monte_carlo_analysis(n_sims=10000)