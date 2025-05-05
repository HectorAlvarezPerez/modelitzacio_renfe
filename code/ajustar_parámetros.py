# ajustar_parametros.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from simulacion_clases import run_simulacion
from ocupacio_real import calcular_ocupacion_real

ALPHA_KEYS = ["alpha_F", "alpha_EI", "alpha_IL", "alpha_ES", "alpha_T"]
BETA_KEYS  = ["beta_C", "beta_QP", "beta_FL", "beta_QV", "beta_EI"]

# ------------------ Función objetivo ------------------

def funcion_objetivo(params):
    n_alpha = len(ALPHA_KEYS)
    alphas = params[:n_alpha]
    betas  = params[n_alpha:]

    try:
        ocup_sim = run_simulacion(alphas, betas)

    except Exception as e:
        print("Fin de la simulación:", e)
        return 1e6

    ocup_real = calcular_ocupacion_real()

    if len(ocup_sim) != len(ocup_real):
        raise ValueError(f"Longitudes distintas: sim={len(ocup_sim)}, real={len(ocup_real)}")

    rmse = np.sqrt(mean_squared_error(ocup_real, ocup_sim))
    return rmse

# ------------------ Optimización ------------------

def ajustar_parametros():
    x0 = [0.1] * (len(ALPHA_KEYS) + len(BETA_KEYS))
    bounds = [(0.0001, 2.0)] * len(x0)

    print("Iniciando optimización...")
    res = minimize(funcion_objetivo, x0, method="L-BFGS-B", bounds=bounds)

    if res.success:
        print(f"\nOptimización exitosa. Error mínimo: {res.fun:.4f}")
        alpha_opt = dict(zip(ALPHA_KEYS, res.x[:len(ALPHA_KEYS)]))
        beta_opt  = dict(zip(BETA_KEYS, res.x[len(ALPHA_KEYS):]))

        print("\nAlphas óptimos:")
        for k, v in alpha_opt.items():
            print(f"  {k}: {v:.5f}")
        print("\nBetas óptimos:")
        for k, v in beta_opt.items():
            print(f"  {k}: {v:.5f}")

        return alpha_opt, beta_opt, res.fun
    else:
        raise RuntimeError(f"Optimización fallida: {res.message}")

# ------------------ Graficar resultados ------------------

def graficar_resultados(ocup_real, ocup_sim):
    estaciones = list(range(1, len(ocup_real) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(estaciones, ocup_real, marker="o", label="Ocupación real", linewidth=2)
    plt.plot(estaciones, ocup_sim, marker="s", label="Ocupación simulada", linewidth=2)
    plt.xlabel("Estaciones")
    plt.ylabel("Ocupación")
    plt.title("Comparación de ocupación real vs simulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Ejecutar ------------------

if __name__ == "__main__":
    alpha_opt, beta_opt, error_min = ajustar_parametros()

    alphas = [alpha_opt[k] for k in ALPHA_KEYS]
    betas  = [beta_opt[k] for k in BETA_KEYS]

    ocup_sim  = run_simulacion(alphas, betas)
    ocup_real = calcular_ocupacion_real()

    graficar_resultados(ocup_real, ocup_sim)
