# compare_real_route_weighted.py
# -----------------------------------------------------------------
#  pip install pandas matplotlib
# -----------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# -------- 1. Tramo, horarios y “otros” trenes --------------------
route_stations = [
    "sant vicenc de calders","el vendrell","l'arboc","els monjos",
    "vilafranca del penedes","la granada","lavern-subirats",
    "sant sadurni d'anoia","gelida","martorell","castellbisbal",
    "el papiol","molins de rei","sant feliu de llobregat",
    "sant joan despi","cornella"
]

orig_times = {
    "sant vicenc de calders":"08:00","el vendrell":"08:05","l'arboc":"08:10",
    "els monjos":"08:16","vilafranca del penedes":"08:22","la granada":"08:26",
    "lavern-subirats":"08:31","sant sadurni d'anoia":"08:35","gelida":"08:40",
    "martorell":"08:48","castellbisbal":"08:54","el papiol":"09:02",
    "molins de rei":"09:06","sant feliu de llobregat":"09:11",
    "sant joan despi":"09:15","cornella":"09:17"
}

station_times = {}
for station, t_str in orig_times.items():
    hh, mm = map(int, t_str.split(":"))
    if mm < 30:
        start = f"{hh:02d}:00"
        end   = f"{hh:02d}:30"
    else:
        start = f"{hh:02d}:30"
        end   = f"{(hh+1)%24:02d}:00"
    station_times[station] = f"{start} - {end}"


pop = {
    "sant vicenc de calders":2259,"el vendrell":36230,"l'arboc":6524,
    "els monjos":13800,"vilafranca del penedes":40000,"la granada":4000,
    "lavern-subirats":9885,"sant sadurni d'anoia":11874,"gelida":8120,
    "martorell":26106,"castellbisbal":12000,"el papiol":7500,
    "molins de rei":26000,"sant feliu de llobregat":46000,
    "sant joan despi":27500,"cornella":88500
}
pop_max = max(pop.values())
PF = {k: v/pop_max for k,v in pop.items()} 

# líneas adicionales que paran en cada estación en la misma franja
other_trains = {
    "sant vicenc de calders": ["R2s"],
    "martorell":              ["R8"],
    "castellbisbal":          ["R8"]
}

double_trains = {
    "sant vicenc de calders": 1,
    "el vendrell": 1,
    "l'arboc": 1,
    "els monjos": 1,
    "vilafranca del penedes": 1,
    "la granada": 1,
    "lavern-subirats": 1,
    "sant sadurni d'anoia": 1,
    "gelida": 1,
    "martorell": 2,
    "castellbisbal": 2,
    "el papiol": 2,
    "molins de rei": 2,
    "sant feliu de llobregat": 2,
    "sant joan despi": 2,
    "cornella": 2
}



# -------- 2. Parámetros de reparto -------------------------------
def lambda_r4(i):
    """Coeficiente de reparto de pasajeros entre trenes."""
    return 1.0 if i == 0 else 0.4
# lambda_r4 = 0.6   # R4 de vuelta atrae 60 % del boarding de nuestro R4 ida
k_other   = 0.6   # líneas ajenas (R2s, R8) atraen 30 %
N         = 38  # len(route_stations)

# m = 0.3 m + (1-2*m)*
def w_up(i): return max(0.3, 1 - i/(N-1))
def w_dn(i): return max(0.3, (1 - w_up(i)))

# -------- 3. Cargar CSV de conteos -------------------------------
df = pd.read_csv("../dades/dades_R4_ordenades_prova.csv", sep=";")
df["HORA_STR"] = df["TRAMO_HORARIO"]

records, ocup = [], 0
print(df["HORA_STR"])
for i, est in enumerate(route_stations):
    h_str = station_times[est]
    row   = df[(df["NOMBRE_ESTACION"]==est) & (df["HORA_STR"]==h_str)]
    up_tot  = int(row["VIAJEROS_SUBIDOS"].iloc[0])  if not row.empty else 0
    dn_tot  = int(row["VIAJEROS_BAJADOS"].iloc[0])  if not row.empty else 0

    n_other = len(other_trains.get(est, []))

    pujan  = int(up_tot * w_up(i) * PF[est] * lambda_r4(n_other) / double_trains[est]) 
    baixen = int(dn_tot * w_dn(i) * PF[est] * lambda_r4(n_other) / double_trains[est])

    if i == 0:   # en la cabecera nadie baja de nuestro tren
        baixen = 0
        pujan = up_tot

    ocup += pujan - baixen
    records.append((est, h_str, up_tot, dn_tot, pujan, baixen, ocup))

# -------- 4. Tabla y gráfica -------------------------------------

cols = ["Estació","Hora","Suben_tot","Bajan_tot","Suben_nuestro",
        "Bajan_nuestro","Ocupación"]
df_route = pd.DataFrame(records, columns=cols)
print(df_route.to_string(index=False))

x = range(len(route_stations))
plt.figure(figsize=(9,4))
plt.step(x, df_route["Ocupación"], where="post", marker="o")
plt.xticks(x, [s.title() for s in route_stations], rotation=45, ha="right")
plt.ylabel("Ocupación (pers)")
plt.title("Ocupación estimada para nuestro R4 (08:00 – 09:17)")
plt.grid(ls=":", alpha=.6)
plt.tight_layout()
plt.show()

# -------- 4. Función para importar -------------------------------------
def calcular_ocupacion_real():
    # from pathlib import Path
    # if not Path("./dades/dades_R4_ordenades_prova.csv").exists():
    #     raise FileNotFoundError("Falta el archivo 'dades_R4_ordenades_prova.csv'")
    
    return df_route["Ocupación"].tolist()
