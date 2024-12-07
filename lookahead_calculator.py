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
    d_offset = B  # Distancia de buffer [m]
    w = wheelbase  # Ancho del vehículo
    turningCar = turning_angle  # Ángulo de giro del vehículo
    object_height = object_height * (0.1) # Convertir la altura del objeto a metros, tipicamente el GSD de KITTI es 0.1.
    hp = object_height  # Altura del obstáculo positivo

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

    # Variables y cálculos para la segunda gráfica (AOV)
    HFOV = np.zeros_like(v_mtps)
    hc = 1.65  # Altura de la camara, dataset KITTI 1.65 m
    thetaSlope = np.deg2rad(15)  # Ángulo de la pendiente
    thetaMin = np.arctan(hc / d_look_stop)  # Ángulo debajo del horizonte determinado por la distancia de frenado
    thetaMax = np.arctan(hc / d_offset)  # Ángulo debajo del horizonte determinado por la longitud de la base B (longitud del automóvil)
    VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax)

    for v in range(1, 151):
        HFOV[v - 1] = d_look_stop[v - 1] / turning[v - 1]

    # Gráfico interactivo 1: Lookahead Distance
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=v_kph, y=HFOV * 1e3, mode='lines', name='HAOV [miliradians]', line=dict(color='lime', width=3)))
    fig1.add_trace(go.Scatter(x=v_kph, y=VFOV * 1e3, mode='lines', name='VAOV [miliradians]', line=dict(color='red', width=3)))
    
    # Cotas de las velocidades a 30 km/h y 60 km/h
    v_30_kph = 30
    v_60_kph = 60

    # Índice de los valres más cercanos a 30 km/h y 60 km/h
    index_30 = (np.abs(v_kph - v_30_kph)).argmin()
    index_60 = (np.abs(v_kph - v_60_kph)).argmin()

    vfov_30 = VFOV[index_30] * 1e3
    vfov_60 = VFOV[index_60] * 1e3
    
    fig1.add_trace(go.Scatter(
        x=[v_kph[index_30]], y=[vfov_30], mode='markers+text', name='VFOV at 30 km/h',
        text=[f'30 km/h: {vfov_30:.2f} miliradians'],
        textposition='top right',
        marker=dict(color='yellow', size=10),
        showlegend=True
    ))

    fig1.add_trace(go.Scatter(
        x=[v_kph[index_60]], y=[vfov_60], mode='markers+text', name='VFOV at 60 km/h',
        text=[f'60 km/h: {vfov_60:.2f} miliradians'],
        textposition='top right',
        marker=dict(color='cyan', size=10),
        showlegend=True
    ))

    # Lineas delimitadoras para las velocidades
    fig1.add_shape(
        type='line',
        x0=v_kph[index_30],
        y0=0,
        x1=v_kph[index_30],
        y1=max(VFOV) * 1e3,
        line=dict(color='yellow', width=2, dash='dash')
    )

    fig1.add_shape(
        type='line',
        x0=v_kph[index_60],
        y0=0,
        x1=v_kph[index_60],
        y1=max(VFOV) * 1e3,
        line=dict(color='cyan', width=2, dash='dash')
    )

    fig1.update_layout(
        title='Angle of View (AOV) VS Vehicle Speed',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Angle of View (AOV) [miliradians]',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        autosize=True,
        margin=dict(t=40, b=40, l=40, r=40)
    )

    # Gráfico interactivo 2: IFOV para obstáculos positivos con diferentes alturas
    looakheadDistance = np.arange(1, 1001)

    if object_height is not None:
        hp_values = np.array([object_height])  # Usar la altura del objeto si se proporciona
    else:
        hp_values = np.arange(0.1, 1.1, 0.1)  # Usar valores por defecto si no se proporciona la altura del objeto
    IFOVp = np.zeros((len(hp_values), len(looakheadDistance)))

    for i, hp in enumerate(hp_values):
        IFOVp[i, :] = np.arctan(hc / looakheadDistance) - np.arctan((hc - hp) / looakheadDistance)

    fig2 = go.Figure()
    for i, hp in enumerate(hp_values):
        fig2.add_trace(go.Scatter(x=looakheadDistance, y=IFOVp[i, :] * 1e3, mode='lines', name=f'Object Height: {hp:.1f} meters', line=dict(width=3)))

    # Leyenda de la altura del objeto
    if object_height is not None:
        fig2.add_trace(go.Scatter(
            x=[1], y=[2], mode='text', name='Object Height',
            text=[f'Object Height: {object_height:.2f} meters'],
            textposition='top right',
            showlegend=False
        ))

    # Cota para el objeto seleccionado
    if object_distance is not None:
        selected_ifov = np.arctan(hc / object_distance) - np.arctan((hc - object_height) / object_distance)
        fig2.add_trace(go.Scatter(
            x=[object_distance], y=[selected_ifov * 1e3], mode='markers+text', name='Selected Object',
            text=[f'IFOV: {selected_ifov * 1e3:.2f} miliradians\nDistance: {object_distance:.2f} meters'],
            textposition='top right',
            marker=dict(color='yellow', size=15),
            showlegend=True
        ))

    # Linea delimitadora
    fig2.add_shape(
        type='line',
        x0=object_distance,
        y0=1,
        x1=object_distance,
        y1=max(IFOVp.flatten()) * 1e3,
        line=dict(color='yellow', width=2, dash='dash')
    )

    fig2.update_layout(
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

    # Gráfico interactivo 3: Lookahead Distance for Stopping
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=v_kph, y=d_look_stop, mode='lines', name='Stopping Distance [m]', line=dict(color='aqua', width=3)))

    # Añadir cotas para las velocidades a 30 km/h y 60 km/h
    fig3.add_trace(go.Scatter(
        x=[v_kph[index_30]], y=[d_look_stop[index_30]], mode='markers+text', name='Stopping Distance at 30 km/h',
        text=[f'30 km/h: {d_look_stop[index_30]:.2f} m'],
        textposition='top right',
        marker=dict(color='yellow', size=10),
        showlegend=True
    ))

    fig3.add_trace(go.Scatter(
        x=[v_kph[index_60]], y=[d_look_stop[index_60]], mode='markers+text', name='Stopping Distance at 60 km/h',
        text=[f'60 km/h: {d_look_stop[index_60]:.2f} m'],
        textposition='top right',
        marker=dict(color='cyan', size=10),
        showlegend=True
    ))

    # Lineas delimitadoras horizontales para las velocidades
    fig3.add_shape(
        type='line',
        x0=0,
        y0=d_look_stop[index_30],
        x1=max(v_kph),
        y1=d_look_stop[index_30],
        line=dict(color='yellow', width=2, dash='dash')
    )

    fig3.add_shape(
        type='line',
        x0=0,
        y0=d_look_stop[index_60],
        x1=max(v_kph),
        y1=d_look_stop[index_60],
        line=dict(color='cyan', width=2, dash='dash')
    )

    # Línea delimitadora horizontal para la distancia del objeto seleccionado (nueva cota)
    if object_distance is not None:

        # Indice del valor más cercano a la distancia del objeto para la cota
        index_obj_distance = (np.abs(d_look_stop - object_distance)).argmin()

        fig3.add_trace(go.Scatter(
        x=[v_kph[index_obj_distance]], y=[d_look_stop[index_obj_distance]], 
        mode='markers+text', 
        name='Selected Object',
        text=[f'Object Distance: {object_distance:.2f} m'],
        textposition='top right',
        marker=dict(color='magenta', size=15),
        showlegend=True
        ))

        fig3.add_shape(
            type='line',
            x0=0,
            y0=object_distance,
            x1=max(v_kph),
            y1=object_distance,
            line=dict(color='magenta', width=2, dash='dash'),
            name="Object Distance"
        )

    fig3.update_layout(
        title='Lookahead Distance for Stopping',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Lookahead distance [m]',
        yaxis=dict(range=[0, 200]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    # Gráfico interactivo 4: Lookahead Distance for Swerving
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=v_kph, y=d_look_swerve, mode='lines', name='Swerving Distance [m]', line=dict(color='orange', width=3)))

    # Añadir cotas para las velocidades a 30 km/h y 60 km/h
    fig4.add_trace(go.Scatter(
        x=[v_kph[index_30]], y=[d_look_swerve[index_30]], mode='markers+text', name='Swerving Distance at 30 km/h',
        text=[f'30 km/h: {d_look_swerve[index_30]:.2f} m'],
        textposition='top right',
        marker=dict(color='yellow', size=10),
        showlegend=True
    ))

    fig4.add_trace(go.Scatter(
        x=[v_kph[index_60]], y=[d_look_swerve[index_60]], mode='markers+text', name='Swerving Distance at 60 km/h',
        text=[f'60 km/h: {d_look_swerve[index_60]:.2f} m'],
        textposition='top right',
        marker=dict(color='cyan', size=10),
        showlegend=True
    ))

    # Lineas delimitadoras horizontales para las velocidades
    fig4.add_shape(
        type='line',
        x0=0,
        y0=d_look_swerve[index_30],
        x1=max(v_kph),
        y1=d_look_swerve[index_30],
        line=dict(color='yellow', width=2, dash='dash')
    )

    fig4.add_shape(
        type='line',
        x0=0,
        y0=d_look_swerve[index_60],
        x1=max(v_kph),
        y1=d_look_swerve[index_60],
        line=dict(color='cyan', width=2, dash='dash')
    )

# Línea delimitadora horizontal para la distancia del objeto seleccionado (nueva cota)
    if object_distance is not None:

        # Indice del valor más cercano a la distancia del objeto para la cota
        index_obj_distance = (np.abs(d_look_stop - object_distance)).argmin()

        fig4.add_trace(go.Scatter(
        x=[v_kph[index_obj_distance]], y=[d_look_stop[index_obj_distance]], 
        mode='markers+text', 
        name='Selected Object',
        text=[f'Object Distance: {object_distance:.2f} m'],
        textposition='top right',
        marker=dict(color='magenta', size=15),
        showlegend=True
        ))

        fig4.add_shape(
            type='line',
            x0=0,
            y0=object_distance,
            x1=max(v_kph),
            y1=object_distance,
            line=dict(color='magenta', width=2, dash='dash'),
            name="Object Distance"
        )

    fig4.update_layout(
        title="Lookahead Distance for Swerving",
        xaxis_title="Vehicle speed [km/h]",
        yaxis_title="Lookahead distance [m]",
        yaxis=dict(range=[0, 200]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    # Retornar las gráficas interactuables de Plotly
    return fig1, fig2, fig3, fig4
    
    
