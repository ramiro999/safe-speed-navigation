# lookahead_calculator.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from image_processing import preprocess_image, plot_detr_results
from model_loader import load_detr_model
import os

model = load_detr_model()

def calculate_lookahead_distance(mu, t, l, B, image_path=None):
    # Parámetros fijos para el modelo
    g = 9.81
    a = -9
    Tper = 0.1
    Tact = 0.25
    d_offset = 2
    w = 1.82
    cog = 0.767
    turningCar = 7.62

    # Rangos de velocidad
    v_mph = np.arange(1, 151)
    v_kph = v_mph * 1.609
    v_mtps = v_mph * (1609.34 / 3600)

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
        d_per[v-1] = v_mtps[v-1] * (2 * Tper)
        d_act[v-1] = v_mtps[v-1] * Tact
        d_brake[v-1] = -v_mtps[v-1]**2 / (2 * a)
        d_look_stop[v-1] = d_offset + d_per[v-1] + d_act[v-1] + d_brake[v-1]
        K_roll[v-1] = (g * (w / 2)) / (cog * v_mtps[v-1]**2)
        K_slip[v-1] = (mu * g) / (v_mtps[v-1]**2)
        turning[v-1] = max(1 / (min(K_roll[v-1], K_slip[v-1])), turningCar)
        d_swerve[v-1] = np.real(np.sqrt(turning[v-1]**2 - (turning[v-1] - w)**2))
        d_look_swerve[v-1] = d_offset + d_per[v-1] + d_act[v-1] + d_swerve[v-1]

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
    plot_path1 = 'Lookahead_Distance_For_Stopping_Distance(kph).png'
    #plt.savefig(plot_path1)
    plt.close()

    # Variables y cálculos para la segunda gráfica (AOV)
    HFOV = np.zeros_like(v_mtps)
    hc = 2
    thetaSlope = np.deg2rad(15)
    thetaMin = np.arctan(hc / d_look_stop)
    thetaMax = np.arctan(hc / d_offset)
    VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax)

    for v in range(1, 101):
        HFOV[v-1] = d_look_stop[v-1] / turning[v-1]

    # Generar la segunda gráfica (Angle of View)
    plt.figure(figsize=(10, 6))
    plt.plot(v_kph, HFOV * 1e3, linewidth=2, label='HAOV [miliradians]')
    plt.plot(v_kph, VFOV * 1e3, linewidth=2, label='VAOV [miliradians]')
    plt.xlabel('Vehicle speed [kph]')
    plt.ylabel('Angle of View (AOV) [miliradians]')
    plt.title('Angle of View (AOV) vs Vehicle Speed')
    plt.grid()
    plt.legend()
    plot_path2 = 'HFOV_VFOV_Miliradians.png'
    #plt.savefig(plot_path2)
    plt.close()

    # Guardar los datos en un archivo de texto
    txt_path = 'Lookahead_Distance_For_Swerve_Stop.txt'
    with open(txt_path, 'w') as f:
        f.write('Stopping Distance [m]\n')
        for x, y in zip(v_kph, d_look_stop):
            f.write(f'SPIE: Vehicle speed [kph]: {x} -- Lookahead distance [m]: {y}\n')
        f.write('Swerve Distance [m]\n')
        for x, y in zip(v_kph, d_look_swerve):
            f.write(f'vT: Vehicle speed [kph]: {x} -- Lookahead distance [m]: {y}\n')

    # Procesar la imagen si se proporciona
    if image_path is not None:
        image, image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(image_tensor)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8

        bboxes = outputs['pred_boxes'][0, keep].numpy()
        labels = probas[keep].argmax(-1).numpy()

        fig_detr = plot_detr_results(image, bboxes, labels)
        return fig_detr, plot_path1, plot_path2, txt_path

    return None, plot_path1, plot_path2, txt_path
