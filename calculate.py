import numpy as np
import math

image_width = 800  # Ancho de la imagen en píxeles
image_fov = 50  # Campo de visión horizontal en grados


f = image_width / (2.0 * math.tan(image_fov * math.pi / 360.0))
print(f)

sensorSize = 7.13 # Tamaño del sensor en mm
focalLength = f * sensorSize
print(focalLength)