import pandas as pd
import matplotlib.pyplot as plt

# Ruta del archivo CSV
csv_path = "../metrics/results/kitti/person/metric_results_train.csv"  # Asegúrate de colocar la ruta correcta

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
summary_path = "metric_summary_train.csv"
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
plt.savefig("mean_metrics_train.png")

# Gráfico de Desviación Estándar
plt.figure(figsize=(8, 5))
plt.bar(std_values.index, std_values.values, color='red', alpha=0.7)
plt.xlabel("Métricas")
plt.ylabel("Desviación Estándar")
plt.title("Desviación Estándar de las Métricas")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("desviacion_metrics_train.png")
