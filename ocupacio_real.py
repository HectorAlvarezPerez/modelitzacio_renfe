# compare_real_route.py

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1) Definimos las estaciones y sus horarios de paso
route_stations = [
    "sant vicenc de calders",
    "el vendrell",
    "l'arboc",
    "els monjos",
    "vilafranca del penedes",
    "la granada",
    "lavern-subirats",
    "sant sadurni d'anoia",
    "gelida",
    "martorell",
    "castellbisbal",
    "el papiol",
    "molins de rei",
    "sant feliu de llobregat",
    "sant joan despi",
    "cornella"
]


# Tiempo de paso original por estación
orig_times = {
    "sant vicenc de calders":  "08:00",
    "el vendrell":             "08:05",
    "l'arboc":                 "08:10",
    "els monjos":              "08:16",
    "vilafranca del penedes":  "08:22",
    "la granada":              "08:26",
    "lavern-subirats":         "08:31",
    "sant sadurni d'anoia":    "08:35",
    "gelida":                  "08:40",
    "martorell":               "08:48",
    "castellbisbal":           "08:54",
    "el papiol":               "09:02",
    "molins de rei":           "09:06",
    "sant feliu de llobregat": "09:11",
    "sant joan despi":         "09:15",
    "cornella":                "09:17"
}

# Construimos intervalos automáticos
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

# print(station_times)


# 2) Cargamos el CSV y preparamos las horas
csv_path = "./dades_R4_ordenades_prova.csv"
df = pd.read_csv(csv_path, sep=";")

# calculamos subidos - bajados (cambio de ocupación)
df["VEL_OCUPACION"] = df["VIAJEROS_SUBIDOS"] - df["VIAJEROS_BAJADOS"]

# extraemos la hora de inicio de cada tramo como objeto time
# def hora_inicio(tramo):
#     return datetime.strptime(tramo.split('-')[0].strip(), "%H:%M").time()

# df["HORA_ORDEN"] = df["TRAMO_HORARIO"].apply(hora_inicio)

# 3) Recorremos las estaciones del tramo y acumulamos la ocupación

# Añadimos esta línea tras calcular HORA_ORDEN:
df["HORA_STR"] = df["TRAMO_HORARIO"]
records = []
ocup_actual = 0

for idx, est in enumerate(route_stations):
    hora_str = station_times[est]
    # filtramos por estación y por cadena de hora
    
    # print(df["NOMBRE_ESTACION"]==est)
    # print(df["NOMBRE_ESTACION"], est)
    # print(df["HORA_STR"]==hora_str)
    # print(df["HORA_STR"][:10])
    # print(hora_str)
    row = df[
        (df["NOMBRE_ESTACION"] == est) &
        (df["HORA_STR"] == hora_str)
    ]

    
    if not row.empty:
        pujen  = int(row["VIAJEROS_SUBIDOS"].iloc[0])
        baixen = int(row["VIAJEROS_BAJADOS"].iloc[0])
    else:
        pujen, baixen = 0, 0

    # print(pujen, baixen)
    if idx == 0:
        ocup_actual = pujen

    ocup_actual += pujen - baixen
    records.append((est, hora_str, pujen, baixen, ocup_actual))
    

# … (construcción de df_route y gráfica igual que antes)


df_route = pd.DataFrame(records, columns=[
    "Estación", "Hora", "Pujan", "Baixen", "Ocupación"
])

# 4) Mostrar la tabla resultante
print(df_route.to_string(index=False))

# 5) Gráfica escalera de ocupación
x = range(len(route_stations))
plt.figure(figsize=(8, 4))
plt.step(x, df_route["Ocupación"], where="post", marker="o", lw=2)
plt.xticks(x, route_stations, rotation=45, ha="right")
plt.xlabel("Estación")
plt.ylabel("Ocupación (personas)")
plt.title("Ocupación real en tramo R4 (08:00–08:26)")
plt.grid(ls=":", alpha=0.6)
plt.tight_layout()
plt.show()
