import numpy as np
import matplotlib.pyplot as plt
import torch
from detr.image_processing import preprocess_image, plot_detr_results
from detr.model_loader import load_detr_model
import os
from matplotlib import cm

# Cargar el modelo (si es necesario para procesamiento adicional)
model = load_detr_model()

# Asegurarse de que exista el directorio de salida
output_directory = 'outputs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def calculate_lookahead_distance(mu, t, l, B, cog, wheelbase, turning_angle, object_height=None, object_distance=None, image_path=None):
    # Parámetros fijos para el modelo (algunos se han convertido en parámetros de la función)
    g = 9.81
    a = -9  # Desaceleración durante el frenado [m/s^2]
    Tper = t  # Tiempo de percepción
    Tact = l  # Latencia
    d_offset = B  # Distancia buffer [m]
    w = wheelbase  # Ancho del vehículo
    turningCar = turning_angle  # Ángulo de giro del vehículo

    # Rangos de velocidad
    v_mph = np.arange(1, 151)  # Velocidades de 1 a 150 mph
    v_kph = v_mph * 1.609  # Conversion de mph a kph
    v_mtps = v_mph * (1609.34 / 3600)  # Conversion de mph a metros por segundo

    # Inicializar distancias para la primera gráfica
    d_per = np.zeros_like(v_mtps)
    d_act = np.zeros_like(v_mtps)
    d_brake = np.zeros_like(v_mtps)
    d_look_stop = np.zeros_like(v_mtps)
    K_roll = np.zeros_like(v_mtps)
    K_slip = np.zeros_like(v_mtps)
    turning = np.zeros_like(v_mtps)
    d_swerve = np.zeros_like(v_mtps)
    d_look_swerve = np.zeros_like(v_mtps)

    # Calcular las distancias de frenado y maniobra para la primera gráfica
    for v in range(1, 151):
        d_per[v - 1] = v_mtps[v - 1] * (2 * Tper)
        d_act[v - 1] = v_mtps[v - 1] * Tact
        d_brake[v - 1] = -v_mtps[v - 1] ** 2 / (2 * a)
        d_look_stop[v - 1] = d_offset + d_per[v - 1] + d_act[v - 1] + d_brake[v - 1]
        K_roll[v - 1] = (g * (w / 2)) / (cog * v_mtps[v - 1] ** 2)
        K_slip[v - 1] = (mu * g) / (v_mtps[v - 1] ** 2)
        turning[v - 1] = max(1 / (min(K_roll[v - 1], K_slip[v - 1])), turningCar)
        d_swerve[v - 1] = np.real(np.sqrt(turning[v - 1] ** 2 - (turning[v - 1] - w) ** 2))
        d_look_swerve[v - 1] = d_offset + d_per[v - 1] + d_act[v - 1] + d_swerve[v - 1]

    # Generar la primera gráfica (Lookahead Distance)
    plt.figure(figsize=(10, 6))
    plt.plot(v_kph, d_look_stop, linewidth=2, label='Stopping Distance [m]')
    plt.plot(v_kph, d_look_swerve, linewidth=2, label='Swerve Distance [m]')
    plt.grid(True)
    plt.xlabel('Vehicle speed [kph]')
    plt.ylabel('Lookahead distance [m]')
    plt.ylim(0, 800)
    plt.legend()
    plt.tight_layout()
    plot_path1 = os.path.join('outputs', 'Lookahead_Distance_For_Stopping_Distance(kph).png')
    plt.savefig(plot_path1, bbox_inches="tight")
    plt.close()

    # Variables y cálculos para la segunda gráfica (AOV)
    HFOV = np.zeros_like(v_mtps)
    hc = cog  # Altura del centro de gravedad [m]
    thetaSlope = np.deg2rad(15)  # Ángulo de la pendiente
    thetaMin = np.arctan(hc / d_look_stop)
    thetaMax = np.arctan(hc / d_offset)
    VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax)

    for v in range(1, 151):
        HFOV[v - 1] = d_look_stop[v - 1] / turning[v - 1]

    # Generar la segunda gráfica (Angle of View)
    plt.figure(figsize=(10, 6))
    plt.plot(v_kph, HFOV * 1e3, linewidth=2, label='HAOV [miliradians]')
    plt.plot(v_kph, VFOV * 1e3, linewidth=2, label='VAOV [miliradians]')
    plt.xlabel('Vehicle speed [kph]')
    plt.ylabel('Angle of View (AOV) [miliradians]')
    plt.title('Angle of View (AOV) vs Vehicle Speed')
    plt.grid()
    plt.legend()
    plot_path2 = os.path.join('outputs', 'HFOV_VFOV_Miliradians.png')
    plt.savefig(plot_path2, bbox_inches="tight")
    plt.close()

    # Gráfico de IFOV para obstáculos positivos
    hp = 0.1  # Altura del obstáculo positivo
    IFOVp = np.arctan(hc / d_look_stop) - np.arctan((hc - hp) / d_look_stop)

    plt.figure()
    plt.plot(v_kph, (IFOVp * 10 ** 3), linewidth=2, label='IFOV Positive [miliradians]')
    plt.xlabel('Vehicle speed [kph]')
    plt.ylabel('Instantaneous Field of View [miliradians]')
    plt.title('Instantaneous Field of View vs Vehicle Speed')
    plt.grid()
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plot_path3 = os.path.join('outputs', 'IFOV_Miliradians.png')
    plt.savefig(plot_path3, bbox_inches="tight")
    plt.close()

    # Gráfico de IFOV para obstáculos positivos con diferentes alturas
    stoppingDistance = np.arange(1, 1001)
    if object_height is not None:
        hp_values = np.array([object_height])  # Usar la altura del objeto si se proporciona
    else:
        hp_values = np.arange(0.1, 1.1, 0.1)  # Usar valores por defecto si no se proporciona la altura del objeto
    IFOVp = np.zeros((len(hp_values), len(stoppingDistance)))

    for i, hp in enumerate(hp_values):
        IFOVp[i, :] = np.arctan(hc / stoppingDistance) - np.arctan((hc - hp) / stoppingDistance)

    plt.figure()
    colors = cm.get_cmap('viridis', len(hp_values))
    for i, hp in enumerate(hp_values):
        plt.plot(stoppingDistance, IFOVp[i, :] * 10 ** 3, color=colors(i), linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title('Positive Obstacle IFOV')
    plt.xlabel('Sensor distance to the scene [m]')
    plt.ylabel('IFOV [milliradians]')
    plt.ylim([10 ** -4, 10 ** 3])
    hp_labels = [f'{h:.1f}' for h in hp_values]
    plt.legend(hp_labels, title="Object size [m]", loc='lower left')
    plt.tight_layout()
    plot_path4 = os.path.join('outputs', 'Positive_Obstacle_IFOV.png')
    plt.savefig(plot_path4, bbox_inches="tight")
    plt.close()

    # Detección opcional si se proporciona una imagen
    if image_path is not None:
        image, image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(image_tensor)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8

        bboxes = outputs['pred_boxes'][0, keep].numpy()
        labels = probas[keep].argmax(-1).numpy()

        fig_detr = plot_detr_results(image, bboxes, labels)
        return fig_detr, plot_path1, plot_path2, plot_path3, plot_path4

    return None, plot_path1, plot_path2, plot_path3, plot_path4
