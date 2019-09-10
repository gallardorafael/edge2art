# Este script se utilizó para extraer el dataset de un video.
# Extrae cada frame de un video, hace el resize a 256x256 y guarda el dataset.
# input_

import cv2
import numpy as np
import argparse
import os
from os.path import isfile, exists

parser = argparse.ArgumentParser()
parser.add_argument("--input_video", required=True, help="Path al video")
parser.add_argument("--output_dir", required=True, help="Directorio donde guardar el dataset extraído")
a = parser.parse_args()

CROP_SIZE = 256
DOWNSAMPLE_RATIO = 4

def resize_out(image):
    # Tamaño a 256x256
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize

def extraer_dataset():
        # Directorio del archivo.
        if isfile(a.input_video):
            video_path = a.input_video
        else:
            print("No existe el video.")
            return

        #Definición del CODEC
        cap = cv2.VideoCapture(video_path)
        if not exists(a.output_dir):
            os.mkdir(a.output_dir)

        save_path = a.output_dir

        # OpenCV
        contador = 0
        while cap.isOpened():
            # Obtenemos el frame.
            ret, frame = cap.read()
            if ret == True:
                if contador % 2 == 0:
                    data_path = save_path+"/"+str(contador)+".png"
                    print(data_path+"\n")
                    frame_resize = resize_out(frame)
                    cv2.imwrite(data_path, frame_resize)
                contador += 1
            else:
                break

        cap.release()
        print('Ha terminado la generación del dataset.')
        cv2.destroyAllWindows()

extraer_dataset()
