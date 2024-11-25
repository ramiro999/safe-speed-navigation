# lookahead_calculator.py:
import numpy as np
import plotly.graph_objects as go
import torch
from detr.image_processing import preprocess_image, plot_detr_results
from detr.model_loader import load_detr_model

# Cargar el modelo
model = load_detr_model()

def calculate_lookahead_distance(mu, t, l, B, cog, wheelbase, turning_angle, object_height=None, object_distance=None, image_path=None):
    # Parámetros fijos para el modelo 
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

    # Gráfico interactivo 1: Lookahead Distance
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=v_kph, y=d_look_stop, mode='lines', name='Stopping Distance [m]', line=dict(color='aqua', width=3)))
    fig1.add_trace(go.Scatter(x=v_kph, y=d_look_swerve, mode='lines', name='Swerve Distance [m]', line=dict(color='orange', width=3)))
    fig1.update_layout(
        title='Lookahead Distance for Stopping and Swerving',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Lookahead distance [m]',
        yaxis=dict(range=[0, 800]),
        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        plot_bgcolor='rgba(0,0,0,0)',    # Fondo del plot transparente
        font=dict(color='white')         # Texto en blanco
    )

    # Variables y cálculos para la segunda gráfica (AOV)
    HFOV = np.zeros_like(v_mtps)
    hc = cog  # Altura del centro de gravedad [m]
    thetaSlope = np.deg2rad(15)  # Ángulo de la pendiente
    thetaMin = np.arctan(hc / d_look_stop)
    thetaMax = np.arctan(hc / d_offset)
    VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax)

    for v in range(1, 151):
        HFOV[v - 1] = d_look_stop[v - 1] / turning[v - 1]

    # Gráfico interactivo 2: Angle of View (AOV)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=v_kph, y=HFOV * 1e3, mode='lines', name='HAOV [miliradians]', line=dict(color='lime', width=3)))
    fig2.add_trace(go.Scatter(x=v_kph, y=VFOV * 1e3, mode='lines', name='VAOV [miliradians]', line=dict(color='red', width=3)))
    fig2.update_layout(
        title='Angle of View (AOV) vs Vehicle Speed',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Angle of View (AOV) [miliradians]',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    # Gráfico interactivo 3: IFOV vs Vehicle Speed
    hp = 0.1  # Altura del obstáculo positivo
    IFOVp = np.arctan(hc / d_look_stop) - np.arctan((hc - hp) / d_look_stop)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=v_kph, y=IFOVp * 1e3, mode='lines', name='IFOV Positive [miliradians]', line=dict(color='magenta', width=3)))
    fig3.update_layout(
        title='Instantaneous Field of View vs Vehicle Speed',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Instantaneous Field of View [miliradians]',
        xaxis=dict(
            type='log',
            range=[0, 3],  # Rango adecuado para mostrar velocidades entre 1 a 1000 (en logaritmo base 10)
            dtick=1  # Espaciado claro para las etiquetas
        ),
        yaxis=dict(
            type='log',
            range=[-1, 3],  # Ajustar para que la escala muestre bien los valores
            dtick=1  # Espaciado claro para las etiquetas en escala logarítmica
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    # Gráfico interactivo 4: IFOV para obstáculos positivos con diferentes alturas
    looakheadDistance = np.arange(1, 1001)
    if object_height is not None:
        hp_values = np.array([object_height])  # Usar la altura del objeto si se proporciona
    else:
        hp_values = np.arange(0.1, 1.1, 0.1)  # Usar valores por defecto si no se proporciona la altura del objeto
    IFOVp = np.zeros((len(hp_values), len(looakheadDistance)))

    for i, hp in enumerate(hp_values):
        IFOVp[i, :] = np.arctan(hc / looakheadDistance) - np.arctan((hc - hp) / looakheadDistance)

    fig4 = go.Figure()
    for i, hp in enumerate(hp_values):
        fig4.add_trace(go.Scatter(x=looakheadDistance, y=IFOVp[i, :] * 1e3, mode='lines', name=f'Object Height: {hp:.1f} m', line=dict(width=3)))

    # Añadir una leyenda personalizada para la altura del objeto seleccionado
    if object_height is not None:
        fig4.add_trace(go.Scatter(
            x=[1], y=[2], mode='text', name='Object Height',
            text=[f'Object Height: {object_height:.2f} m'],
            textposition='top right',
            showlegend=False
        ))

    # Actualizar diseño de la gráfica para una mejor visualización
    fig4.update_layout(
        title='Positive Obstacle IFOV',
        xaxis_title='Sensor distance to the scene [m]',
        yaxis_title='IFOV [milliradians]',
        xaxis_type='log',
        yaxis_type='log',
        xaxis=dict(dtick=1, title=dict(standoff=20)),
        yaxis=dict(dtick=1, title=dict(standoff=20), range=[-1, 3.5]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    # Retornar las gráficas interactuables de Plotly
    return fig1, fig2, fig3, fig4
