# edge2art
Proyecto basado en el modelo de traducción de imagen a imagen ***[Pix2Pix](https://phillipi.github.io/pix2pix/)***: Redes Adversarias Generativas Condicionales.

***edge2art*** se ve inspirado por ***[CycleGAN](https://junyanz.github.io/CycleGAN/)***. ***edge2art*** es capaz de crear arte utilizando el modelo **Pix2Pix**.

## Instalación

### Requisitos:
* tensorflow == 1.5 (o tensorflow-gpu == 1.5)
* tkinter
* python-opencv
* PIL
* imutils
* numpy
* scipy

### Instrucciones:
```sh
# Clonar el repositorio
git clone https://github.com/gallardorafael/edge2art.git
cd edge2art
``` 

## Entrenamiento: 
Redimensionar las imágenes a un tamaño de 256x256 píxeles:
```sh
cd Creación\ del\ Dataset/
python3 preprocess.py --input_dir <path al directorio que contiene las imágenes originales> --output_dir <path del directorio dónde guardar las imágenes redimensionadas> --operation 
```

Extraer los bordes de las imágenes redimensionadas:
```sh
cd Creación\ del\ Dataset/
python3 images2edges.py --input_dir <path al directorio con las imágenes normales> --output_dir <ṕath al directorio de guardado de las imágenes de bordes>
```

Crear el dataset AtoB para Pix2Pix:
```sh
cd Creación\ del\ Dataset/
python3 edge2art_dataset.py --a_dir <path al directorio con las imágenes de bordes> --b_dir <path al directorio con las imágenes normales> --output_dir <path al directorio de guardado del dataset de entrenamiento>
```
 
Entrenamiento del modelo:
```sh
python3 pix2pix.py --input_dir <path del directorio con las imágenes de entrenamiento> --output_dir <path al directorio dónde guardar el modelo> --mode train --max_epochs 200 --which_direction AtoB
```

Reducir y exportar el modelo:
```sh
cd Reducción\ de\ Modelos/
python3 reduce_model.py --model-input <path al directorio con el modelo entrenado previamente> --model-output <path al directorio dónde guardar el modelo reducido>

python3 freeze_model.py --model_folder <path al directorio que contiene el modelo reducido>
```
  ** El modelo se guardará con el nombre "frozen_model.pb" en este mismod directorio.
  ** Deberá mover los modelos exportados a una carpeta llamada "models" que se ubique en el mismo directorio que el archivo edge2art.py

## Experimentando con edge2art
Correr la GUI de edge2art:
```sh
python3 edge2art.py
```
 ** La interfaz gráfica permite varias opciones. 
