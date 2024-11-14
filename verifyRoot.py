import os

root_path = "/home/ramiro-avila/simulation-gradio/stereo/NMRF/datasets/KITTI"
if os.path.exists(root_path):
    print("La ruta existe y es accesible.")
else:
    print(f"La ruta '{root_path}' no existe o no es accesible.")


import sys
sys.path.append('/home/ramiro-avila/simulation-gradio/stereo/NMRF')

from stereo.NMRF.nmrf.data.datasets import KITTI

root_path = "/home/ramiro-avila/simulation-gradio/stereo/NMRF/datasets/KITTI"
dataset = KITTI(root=root_path, split='testing', image_set='kitti_2015')
print(f"Tamaño del dataset: {len(dataset.image_list)}")

