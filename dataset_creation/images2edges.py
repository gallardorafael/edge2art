# Este script convierte todas las imágenes del directorio en sus versiones
# de bordes.
# La transformación utiliza la implementacion de Canny Edge Detection de OpenCV

import cv2
import numpy as np
import os
from os.path import join, isfile, splitext, isdir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Directorio con las images originales")
parser.add_argument("--output_dir", required=True, help="Directorio de guardado")
a = parser.parse_args()

def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)

def preprocess(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray_image, (3, 3), 0)

def draw_edges():
    count = 0
    # Se crean los directorios por estilo para las imágenes de bordes.
    if not isdir(a.output_dir):
        os.mkdir(a.output_dir)
    for artwork_name in os.listdir(a.input_dir):
        count += 1
        print("Processing %d images \n"%(count))
        artwork_path = os.path.join(a.input_dir, artwork_name)
        image = preprocess(artwork_path)
        edge = 255 -  auto_canny(image)
        target_path = os.path.join(a.output_dir, artwork_name)
        print(target_path)
        cv2.imwrite(target_path, edge)

draw_edges()
