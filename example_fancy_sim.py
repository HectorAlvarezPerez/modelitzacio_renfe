import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# — Datos ya simulados
stations = ["Sant Vicenç", "El Vendrell", "L'Arboç", "Els Monjos",
            "Vilafranca", "La Granada"]
occupancies = [ 32,  28,  25,   20,   35,    22]  # tu lista de O tras cada parada

# 1) Preparar la figura
fig, ax = plt.subplots(figsize=(10,4))
x = np.arange(len(stations))
bars = ax.bar(x, occupancies, color="lightgray")
train_marker, = ax.plot([], [], marker="s", markersize=20, color="black")  # el “tren”

ax.set_xticks(x)
ax.set_xticklabels(stations, rotation=45, ha="right")
ax.set_ylim(0, max(occupancies)*1.2)
ax.set_ylabel("Ocupación (pers.)")
ax.set_title("Simulación R4: tren en marcha")

# 2) Función de inicialización
def init():
    train_marker.set_data([], [])
    for b in bars:
        b.set_color("lightgray")
    return bars + (train_marker,)

# 3) Función de animación: frame i → estación i
def animate(i):
    # Colorea solo la barra actual
    for idx, b in enumerate(bars):
        b.set_color("orange" if idx == i else "lightgray")
    # Mueve el marcador al centro de la barra i, a la altura de la barra
    train_marker.set_data(x[i], occupancies[i] + max(occupancies)*0.05)
    return bars + (train_marker,)

# 4) Crear animación
anim = FuncAnimation(fig, animate, frames=len(stations),
                     init_func=init, blit=True, interval=1000)

# 5) Mostrarla (en Jupyter pone anim)
plt.show()
