# safety_calculator.py:
import numpy as np
import plotly.graph_objects as go
import torch
from detr.image_processing import preprocess_image, plot_detr_results
from detr.model_detr import load_detr_model

"""
Script para calcular las distancias de frenado y esquiva para un vehículo autónomo en función de los parámetros de entrada.
"""

# Cargar el modelo
model = load_detr_model()

def set_plot_style(is_dark_mode=False):
    """
    Define los estilos de diseño para gráficos Plotly en modo oscuro o claro.
    
    Args:
        is_dark_mode (bool): Si es True, aplica el tema oscuro; de lo contrario, aplica el tema claro.

    Returns:
        dict: Parámetros de estilo para el diseño de gráficos.
    """
    if is_dark_mode:
        return {
            'paper_bgcolor': 'rgba(0,0,0,0)',  # Fondo del papel transparente (oscuro)
            'plot_bgcolor': 'rgba(0,0,0,0)',  # Fondo del gráfico transparente (oscuro)
            'font': dict(color='white'),      # Texto blanco
            'line_color': {
                'lime': 'lime',
                'red': 'red',
                'yellow': 'yellow',
                'cyan': 'cyan',
                'magenta': 'magenta',
                'orange': 'orange',
                'aqua': 'aqua'
            }
        }
    else:
        return {
            'paper_bgcolor': 'rgba(255,255,255,1)',  # Fondo del papel blanco
            'plot_bgcolor': 'rgba(240,240,240,1)',  # Fondo del gráfico claro
            'font': dict(color='black'),            # Texto negro
            'line_color': {
                'lime': '#32CD32',
                'red': '#FF4500',
                'yellow': '#FFD700',
                'cyan': '#00FFFF',
                'magenta': '#FF00FF',
                'orange': '#FFA500',
                'aqua': '#00CED1'
            }
        }

def calculate_lookahead_distance(mu, t, l, B, cog, wheelbase, turning_angle, object_height=None, object_distance=None, image_path=None):
    """
    Calcula las distancias de frenado y esquiva para un vehículo autónomo en función de los parámetros de entrada.
    
    Parámetros:
    - mu: Coeficiente de fricción entre los neumáticos y la carretera.
    - t: Tiempo de procesamiento de los datos.
    - l: Tiempo de latencia que tarda el sistema del vehiculo autonomo demora en realizar la maniobra.
    - B: Distancia de desplazamiento debida a la longitud del coche.
    - cog: Centro de gravedad del vehículo.
    - wheelbase: Distancia entre los ejes delantero y trasero del vehículo.
    - turning_angle: Ángulo de giro del vehículo.
    - object_height: Altura del objeto en metros.
    - object_distance: Distancia del objeto al sistema óptico en metros.
    - image_path: Ruta de la imagen a procesar.

    Retorna:
    - fig1: Gráfico interactivo de Ángulo de Vista (AOV) vs Velocidad del vehículo.
    - fig2: Gráfico interactivo de Campo de Visión Instantáneo (IFOV) para obstáculos positivos con diferentes alturas.
    - fig3: Gráfico interactivo de Distancia de frenado.
    - fig4: Gráfico interactivo de Distancia de esquiva.
    """


    is_dark_mode = False
    style = set_plot_style(is_dark_mode)
    
    # Parámetros fijos para el modelo 
    g = 9.81
    a = - mu * g  # Desaceleración durante el frenado [m/s^2]
    Tper = t  # Tiempo de percepción [s]
    Tact = l  # Latencia [s]
    d_offset = B  # Distancia de desplazamiento debida a la longitud del coche. [m]
    w = wheelbase  # Ancho del vehículo [m]
    turningCar = turning_angle  # Radio de giro del vehículo [m]

    object_distance = object_distance # Distancia del objeto al sistema óptico en metros
    object_distance_mm = object_distance * 1000  # Distancia del objeto al sistema óptico en milímetros
    hp = object_height # Altura del objeto en metros
   

    pi = 3.141592653589793

    speed_limit = 251

    # Rangos de velocidad
    v_kph = np.arange(1, 251)  # Velocidades de 1 a 150 km/h
    v_mtps = v_kph / 3.6  # Conversion de kph a m/s

    # Inicialización de las distancias y constantes
    d_per = np.zeros_like(v_mtps)
    d_act = np.zeros_like(v_mtps)
    d_brake = np.zeros_like(v_mtps)
    d_look_stop = np.zeros_like(v_mtps)
    K_roll = np.zeros_like(v_mtps)
    K_slip = np.zeros_like(v_mtps)
    turning = np.zeros_like(v_mtps)
    d_swerve = np.zeros_like(v_mtps)
    d_look_swerve = np.zeros_like(v_mtps)

    # Calcular las velocidades para distancias de frenado y maniobra para la primera gráfica
    for v in range(1, 251):
        # Distancias de frenado
        d_per[v - 1] = v_mtps[v - 1] * (2 * Tper)
        d_act[v - 1] = v_mtps[v - 1] * Tact
        d_brake[v - 1] = - v_mtps[v - 1] ** 2 / (2 * a)
        d_look_stop[v - 1] = d_offset + d_per[v - 1] + d_act[v - 1] + d_brake[v - 1]
        
        # Distancias de esquiva
        K_roll[v - 1] = (g * (w / 2)) / (cog * v_mtps[v - 1] ** 2)
        K_slip[v - 1] = (mu * g) / (v_mtps[v - 1] ** 2)
        turning[v - 1] = max(1 / (min(K_roll[v - 1], K_slip[v - 1])), turningCar)
        d_swerve[v - 1] = np.real(np.sqrt(turning[v - 1] ** 2 - (turning[v - 1] - w) ** 2))
        d_look_swerve[v - 1] = d_offset + d_per[v - 1] + d_act[v - 1] + d_swerve[v - 1]


    # Variables y cálculos para la primera gráfica (AOV vs Velocidad)
    HFOV = np.zeros_like(v_mtps)
    hc = 1.65  # Altura de la cámara promedio, dataset KITTI 1.65 m
    thetaSlope = np.deg2rad(15)  # Ángulo de la pendiente, por lo general 15 grados
    
    thetaMin = np.arctan(hc / d_look_stop)  # Ángulo debajo del horizonte determinado por la distancia de frenado
    thetaMax = np.arctan(hc / d_offset)  # Ángulo debajo del horizonte determinado por la longitud de la base B (longitud del automóvil)
    VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax)

    for v in range(1, 251):
        HFOV[v - 1] = d_look_stop[v - 1] / turning[v - 1]

    selected_ifov = None
    safe_speed_stop = None
    safe_speed_swerve = None
    safe_speed_stop_haov = None
    safe_speed_swerve_haov = None
    safe_speed_stop_vaov = None
    safe_speed_swerve_vaov = None
    object_height_meters = hp
    
    if object_distance is not None:
        selected_ifov = np.arctan(hc / object_distance) - np.arctan((hc - hp) / object_distance)
    
    if object_distance is not None:
        index_obj_distance = (np.abs(d_look_stop - object_distance)).argmin()
        index_obj_distance = min(index_obj_distance, len(d_look_stop) - 1)  # Corregir índice fuera de rango

        if d_look_stop[index_obj_distance] > object_distance:
            index_obj_distance = max(0, index_obj_distance - 1)  # Evitar índices negativos
        safe_speed_stop = v_kph[index_obj_distance] if index_obj_distance >= 0 else None

    if object_distance is not None:
        index_obj_distance = (np.abs(d_look_swerve - object_distance)).argmin()
        index_obj_distance = min(index_obj_distance, len(d_look_swerve) - 1)  # Corregir índice fuera de rango

        if d_look_swerve[index_obj_distance] > object_distance:
            index_obj_distance = max(0, index_obj_distance - 1)  # Evitar índices negativos
        safe_speed_swerve = v_kph[index_obj_distance] if index_obj_distance >= 0 else None

    if safe_speed_stop:
        idx = np.abs(v_kph - safe_speed_stop).argmin()
        idx = min(idx, len(v_kph) - 1)
        safe_speed_stop_haov = np.degrees(HFOV[idx])
        safe_speed_stop_vaov = np.degrees(VFOV[idx])

    if safe_speed_swerve:
        idx = np.abs(v_kph - safe_speed_swerve).argmin()
        idx = min(idx, len(v_kph) - 1)
        safe_speed_swerve_haov = np.degrees(HFOV[idx])
        safe_speed_swerve_vaov = np.degrees(VFOV[idx])

    # Gráfico 1: Angle of View (AOV)
    HFOV = d_look_stop / turning
    VFOV = 2 * thetaSlope + np.minimum(thetaMin, thetaMax)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=v_kph, y=np.degrees(HFOV), mode='lines', name='HAOV [degrees]',
                            line=dict(color=style['line_color']['lime'], width=3)))

    if safe_speed_stop is not None:
        fig1.add_trace(go.Scatter(
            x=[safe_speed_stop], 
            y=[np.degrees(HFOV[np.abs(v_kph - safe_speed_stop).argmin()])],
            mode='markers+text',
            name='Safe Speed (HAOV)',
            textposition='top right',
            marker=dict(color='magenta', size=15),
            showlegend=True
        ))

    # VAOV con cotas
    fig1.add_trace(go.Scatter(x=v_kph, y=np.degrees(VFOV), mode='lines', name='VAOV [degrees]',
                              line=dict(color=style['line_color']['red'], width=3)))
    if safe_speed_stop is not None:
        fig1.add_trace(go.Scatter(
            x=[safe_speed_stop], 
            y=[np.degrees(VFOV[np.abs(v_kph - safe_speed_stop).argmin()])],
            mode='markers+text',
            name='Safe Speed (VAOV)',
            textposition='top right',
            marker=dict(color='purple', size=15),
            showlegend=True
        ))

    fig1.update_layout(
        title='Angle of View (AOV) VS Vehicle Speed',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Angle of View (AOV) [degrees]',
        paper_bgcolor=style['paper_bgcolor'],
        plot_bgcolor=style['plot_bgcolor'],
        font=style['font']
    )

    # Gráfico 2: IFOV para obstáculos positivos ------------------------------------------------------------
    looakheadDistance = np.arange(1, 1001)  # Lookahead distance in meters          
    IFOVp = np.zeros(len(looakheadDistance))  # Solo un conjunto de valores, porque hp es escalar

    # Calcular IFOV para la altura del objeto
    for i, distance in enumerate(looakheadDistance):
        IFOVp[i] = np.arctan(hc / distance) - np.arctan((hc - hp) / distance)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=looakheadDistance,
        y=IFOVp * (180/ np.pi),   # Grados
        mode='lines',
        name=f'Object Size: {hp:.4f} meters',
        line=dict(width=3)
    ))

    # Cota para el objeto seleccionado
    if object_distance is not None:
        selected_ifov = np.arctan(hc / object_distance) - np.arctan((hc - hp) / object_distance)
        fig2.add_trace(go.Scatter(
            x=[object_distance],
            y=[selected_ifov * (180/np.pi)],  # Grados
            mode='markers+text',
            name='Selected Object',
            text=[f'IFOV: {selected_ifov * (180/np.pi):.2f} degrees<br>Distance: {object_distance:.2f} meters'],
            textposition='top right',
            marker=dict(color='yellow', size=15),
            showlegend=True
        ))

        # Línea delimitadora para la distancia del objeto seleccionado
        fig2.add_shape(
            type='line',
            x0=object_distance,
            y0=0.1,  # Mínimo valor de IFOV en grados
            x1=object_distance,
            y1=max(IFOVp) * (180/np.pi),  # Máximo valor de IFOV en grados
            line=dict(color='yellow', width=2, dash='dash')
        )

    fig2.update_layout(
        title='Obstacle IFOV',
        xaxis_title='Sensor distance to the scene [m]',
        yaxis_title='IFOV [degrees]',
        xaxis_type='log',  # Escala logarítmica en x
        yaxis_type='log',  # Escala logarítmica en y
        xaxis=dict(
            title=dict(standoff=20),
            dtick=1
        ),
        yaxis=dict(
            type='log',
            range=[-3, 3],  # Ajustar el rango del eje Y (10^-3 a 10^3)
            title=dict(standoff=20)
        ),
        paper_bgcolor=style['paper_bgcolor'],
        plot_bgcolor=style['plot_bgcolor'],
        font=style['font']
    )

    # Gráfico 3: Velocidad para realizar la maniobra de frenado ---------------------------------------------------------------------------------
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=v_kph, y=d_look_stop, mode='lines', name='Stopping Distance [m]',
                              line=dict(color=style['line_color']['aqua'], width=3)))
    

    # Línea delimitadora horizontal para la distancia del objeto seleccionado (nueva cota)
    if object_distance is not None:

        # Indice del valor más cercano a la distancia del objeto para la cota
        index_obj_distance = (np.abs(d_look_stop - object_distance)).argmin()
        if d_look_stop[index_obj_distance] > object_distance:
            index_obj_distance -= 1

        if index_obj_distance < 0:
            fig3.add_trace(go.Scatter
            (
            x=[0], y=[0], mode='text', name='Stopping Not Possible',
            text=['Stopping not possible for distance'],
            textposition='top right',
            textfont=dict(color='red', size=15),
            showlegend=False
            ))

        else:

            fig3.add_vrect(
                x0=speed_limit, x1=max(v_kph),
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Unsafe Zone", annotation_position="top left"
            )

            fig3.add_trace(go.Scatter(
                x=[v_kph[index_obj_distance]], y=[d_look_stop[index_obj_distance]],
                mode='markers+text',
                name='Selected Object',
                text=[f'Object Distance: {object_distance:.2f} m <br>Safe speed: {v_kph[index_obj_distance]:.2f} km/h'],
                textposition='top right',
                textfont=dict(size=10),
                marker=dict(color='magenta', size=15),
                showlegend=True
            ))

            if v_kph[index_obj_distance] < speed_limit:
                fig3.add_shape(
                    type='line',
                    x0=0,
                    y0=d_look_stop[index_obj_distance],
                    x1=max(v_kph),
                    y1=d_look_stop[index_obj_distance],
                    line=dict(color='magenta', width=2, dash='dash'),
                    name="Object Distance"
                )

    fig3.update_layout(
        title='Safe navigation speed for Stopping',
        xaxis_title='Vehicle speed [km/h]',
        yaxis_title='Lookahead distance [m]',
        yaxis=dict(range=[0, 150]),
        paper_bgcolor=style['paper_bgcolor'],
        plot_bgcolor=style['plot_bgcolor'],
        font=style['font']
    )

    # Gráfico 4: Velocidad para realizar la maniobra de esquiva ----------------------------------------------------------------------------
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=v_kph, y=d_look_swerve, mode='lines', name='Swerving Distance [m]',
                              line=dict(color=style['line_color']['orange'], width=3)))
    
    # Línea delimitadora horizontal para la distancia del objeto seleccionado (nueva cota)
    if object_distance is not None:

        # Indice del valor más cercano a la distancia del objeto para la cota
        index_obj_distance = (np.abs(d_look_swerve - object_distance)).argmin()
        if d_look_swerve[index_obj_distance] > object_distance:
            index_obj_distance -= 1

        if index_obj_distance < 0:
            # 🔴 Mostrar solo el mensaje de "Swerving Not Possible"
            fig4.add_trace(go.Scatter(
                x=[0], y=[0], mode='text', name='Swerving Not Possible',
                text=['Swerving not possible for distance'],
                textposition='top right',
                textfont=dict(color='red', size=15),
                showlegend=False
            ))

        else:
            # Agregar la zona insegura solo si se puede esquivar
            fig4.add_vrect(
                x0=speed_limit, x1=max(v_kph),
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Unsafe Zone", annotation_position="top left"
            )

            if index_obj_distance > speed_limit:
                fig4.add_trace(go.Scatter(
                    x=[v_kph[index_obj_distance]], y=[d_look_swerve[index_obj_distance]],
                    mode='text',
                    text=['Unsafe: Speed exceeds limit'],
                    textposition='top right',
                    textfont=dict(color='red', size=15),
                    showlegend=False
                ))
                fig4.add_shape(
                    type='line',
                    x0=0,
                    y0=object_distance,
                    x1=speed_limit,
                    y1=object_distance,
                    line=dict(color='magenta', width=2, dash='dash'),
                    name="Object Distance"
                )

            else:
                fig4.add_trace(go.Scatter(
                    x=[v_kph[index_obj_distance]], y=[d_look_swerve[index_obj_distance]], 
                    mode='markers+text', 
                    name='Selected Object',
                    text=[f'Object Distance: {object_distance:.2f} m <br>Safe speed: {v_kph[index_obj_distance]:.2f} km/h'],
                    textposition='top right',
                    marker=dict(color='magenta', size=15),
                    textfont=dict(size=10),
                    showlegend=True
                ))

                if v_kph[index_obj_distance] < speed_limit:
                    fig4.add_shape(
                        type='line',
                        x0=0,
                        y0=d_look_swerve[index_obj_distance],
                        x1=max(v_kph),
                        y1=d_look_swerve[index_obj_distance],
                        line=dict(color='magenta', width=2, dash='dash'),
                        name="Object Distance"
                    )

    fig4.update_layout(
        title="Safe navigation speed for Swerving",
        xaxis_title="Vehicle speed [km/h]",
        yaxis_title="Lookahead distance [m]",
        yaxis=dict(range=[0, 150]),
        paper_bgcolor=style['paper_bgcolor'],
        plot_bgcolor=style['plot_bgcolor'],
        font=style['font']
    )



    return (
        fig1, fig2, fig3, fig4,
        selected_ifov,
        safe_speed_stop,
        safe_speed_swerve,
        safe_speed_stop_haov,
        safe_speed_swerve_haov,
        safe_speed_stop_vaov,
        safe_speed_swerve_vaov,
        object_height_meters
    )
