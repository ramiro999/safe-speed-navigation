import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Ruta del archivo CSV
csv_path = "../metrics/metric_results_test.csv"  # Asegúrate de colocar la ruta correcta

# Cargar el archivo CSV
df = pd.read_csv(csv_path)

# Calcular la media y desviación estándar de cada métrica
mean_values = df.iloc[:, 1:].mean()  # Promedio por métrica
std_values = df.iloc[:, 1:].std()    # Desviación estándar por métrica

# Mostrar los resultados en consola
print("\n📊 Media de cada métrica:")
print(mean_values)

print("\n📉 Desviación estándar de cada métrica:")
print(std_values)

# Opcional: Guardar los resultados en un archivo CSV
summary_path = "metric_summary_test.csv"
summary_df = pd.DataFrame({"Métrica": mean_values.index, "Media": mean_values.values, "Desviación Estándar": std_values.values})
summary_df.to_csv(summary_path, index=False)

print(f"\n✅ Resultados guardados en: {summary_path}")

# Crear gráficos de barras para la media y la desviación estándar

# Gráfico de Media
plt.figure(figsize=(8, 5))
plt.bar(mean_values.index, mean_values.values, color='blue', alpha=0.7)
plt.xlabel("Métricas")
plt.ylabel("Media")
plt.title("Media de las Métricas")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.savefig("mean_metrics_train.png")

# Gráfico de Desviación Estándar
plt.figure(figsize=(8, 5))
plt.bar(std_values.index, std_values.values, color='red', alpha=0.7)
plt.xlabel("Métricas")
plt.ylabel("Desviación Estándar")
plt.title("Desviación Estándar de las Métricas")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.savefig("desviacion_metrics_train.png")


#### -------------------------------------------- ####
# Definir las métricas que queremos analizar
metrics = ["mae", "rmse", "imae", "irmse"]

# Cargar los archivos CSV

# Entrenamiento

# files = {
#     "road": "../metrics_analysis/results_depth/kitti/train/road/metric_summary_train_road.csv",
#     "residential": "../metrics_analysis/results_depth/kitti/train/residential/metric_summary_train_residential.csv",
#     "person": "../metrics_analysis/results_depth/kitti/train/person/metric_summary_train_person.csv",
#     "city": "../metrics_analysis/results_depth/kitti/train/city/metric_summary_train_city.csv",
#     "campus": "../metrics_analysis/results_depth/kitti/train/campus/metric_summary_train_campus.csv",
# }

# Validacion

files = {
     "road": "../metrics_analysis/results_depth/kitti/val/road/metric_summary_val_road.csv",
     "residential": "../metrics_analysis/results_depth/kitti/val/residential/metric_summary_val_residential.csv",
#     "person": "../metrics_analysis/results_depth/kitti/val/person/metric_summary_train_person.csv",
     "city": "../metrics_analysis/results_depth/kitti/val/city/metric_summary_val_city.csv",
     "campus": "../metrics_analysis/results_depth/kitti/val/campus/metric_summary_val_campus.csv",
}


# Leer los archivos y almacenarlos en un diccionario
data = {key: pd.read_csv(file) for key, file in files.items()}

# Crear un DataFrame vacío para almacenar los datos consolidados
summary_df = pd.DataFrame(columns=["Class"] + metrics)

# Extraer las métricas de la columna "Media"
for key, df in data.items():
    metric_values = df[df["Métrica"].isin(metrics)]["Media"].values  # Obtener valores de métricas
    summary_df.loc[len(summary_df)] = [key] + list(metric_values)

# Graficar Barras Agrupadas para MAE y RMSE
plt.figure(figsize=(10, 6))
plt.tight_layout()
plt.style.use('seaborn-v0_8-colorblind')
x = np.arange(2)  # Posiciones en X para MAE y RMSE
width = 0.15  # Ancho de barras

for i, cls in enumerate(summary_df["Class"]):
    plt.bar(x + i * width, summary_df.iloc[i, 1:3], width=width, label=cls)

plt.xticks(x + width, ["mae", "rmse"], fontsize=14)  # Etiquetas en eje X centradas y tamaño de fuente aumentado
plt.xlabel("Métrica", fontsize=14)
plt.ylabel("Error promedio de estimación de profundidad", fontsize=14)
#plt.title("Comparación de MAE y RMSE entre escenas")
plt.legend(title="Escenas")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("mae_rmse_comparison.png", dpi=300)  # Guardar la figura con alta resolución

# Graficar Barras Agrupadas para IMAE y IRMSE
plt.figure(figsize=(10, 6))
x = np.arange(2)  # Posiciones en X para IMAE y IRMSE

for i, cls in enumerate(summary_df["Class"]):
    plt.bar(x + i * width, summary_df.iloc[i, 3:], width=width, label=cls)

plt.xticks(x + width, ["imae", "irmse"], fontsize=14)  # Etiquetas en eje X centradas
plt.xlabel("Métrica", fontsize=14)
plt.ylabel("Error promedio de estimación de profundidad", fontsize=14)
#plt.title("Comparación de IMAE y IRMSE entre escenas")
plt.legend(title="Escenas")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("imae_irmse_comparison.png", dpi=300)

