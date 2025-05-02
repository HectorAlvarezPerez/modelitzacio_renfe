import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np 
import os 

# Cargar datos
df = pd.read_csv("./dades/dades_R4_ordenades_prova.csv", sep=";")

# Calcular cambio de ocupación (sin .diff)
df["VEL_OCUPACION"] = df["VIAJEROS_SUBIDOS"] - df["VIAJEROS_BAJADOS"]

# Convertir tramos horarios a orden cronológico
def convertir_a_hora_inicio(tramo):
    inicio = tramo.split('-')[0].strip()
    return datetime.strptime(inicio, "%H:%M").time()

df["HORA_ORDEN"] = df["TRAMO_HORARIO"].apply(convertir_a_hora_inicio)

# Ordenar tramos horarios
tramos_ordenados = df.sort_values("HORA_ORDEN")["TRAMO_HORARIO"].unique()

# Obtener estaciones únicas
estaciones = df["NOMBRE_ESTACION"].unique()

# Graficar una figura por estación
for estacion in estaciones:
    df_est = df[df["NOMBRE_ESTACION"] == estacion]
    
    # Reindexar para que estén todos los tramos, rellenando con 0 si falta alguno
    df_est = df_est.set_index("TRAMO_HORARIO").reindex(tramos_ordenados).reset_index()
    df_est["VEL_OCUPACION"] = df_est["VEL_OCUPACION"].fillna(0)

    # Crear gráfico individual
    plt.figure(figsize=(15, 6))
    plt.plot(df_est["TRAMO_HORARIO"], df_est["VEL_OCUPACION"], 
             marker='o', linestyle='-', color='blue', markersize=5)
    
    plt.title(f"Cambio de ocupación horaria - Estación: {estacion.title()}", fontsize=14, pad=20)
    plt.xlabel("Tramo horario", fontsize=12)
    plt.ylabel("Cambio en ocupación\n(Subidos - Bajados)", fontsize=12)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    nombre_archivo = f"{estacion.strip().replace(' ', '_')}_ocupacion.png"
    ruta_archivo   = os.path.join("./plots_velOcup", nombre_archivo)
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')

    # plt.show()
    # plt.close()
    
