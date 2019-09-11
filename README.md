# edge2art
Proyecto basado en el modelo de traducción de imagen a imagen ***[Pix2Pix](https://phillipi.github.io/pix2pix/)***: Redes Adversarias Generativas Condicionales.

## Funcionalidades
***edge2art*** es capaz de traducir cualquier imagen en un estilo artistico utilizando la extracción de bordes para favorecer la generación de contenido de la cGAN. Desde la GUI de edge2art es posible:
* Convertir imágenes en arte
* Ver video en tiempo real en algún estilo artístico
* Traducir videos en animaciones artisticas

Los resultados se pueden apreciar en las siguientes demostraciones: 
![Rococo](https://github.com/gallardorafael/edge2art/blob/master/docs/dot_rococo.gif)
![Ukiyo](https://github.com/gallardorafael/edge2art/blob/master/docs/robot_ukiyo.gif)

Para una demostración completa ver el [video](link_al_video) demostrativo en Youtube.

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

### Datasets
#### Dataset de ejemplo
El dataset de la corriente Rococó, se encuentra disponible para su descarga en: [Rococó](https://drive.google.com/open?id=1lSdUbAf-OFjQCVrgRM8L1nL0R2dnV4ng), este dataset tiene un total de 2089 obras de arte en la corriente artística conocida como Rococó. El dataset está listo comenzar el entrenamiento de pix2pix con dirección AtoB. Para dar una idea del tiempo necesario para entrenar esta corriente se presentan los tiempos de entrenamiento en diversas plataformas:
* TensorFlow con Intel Core i7 6700HQ: ~108 horas.
* TensorFlow con Nvidia GTX 960 : ~59 horas.
* TensorFlow con Nvidia Tesla K80: ~30 horas.
* TensorFlow con Nvidia Tesla T4: ~10 horas.

Tomar en consideración previo al entrenamiento.

#### Crear tus propios datasets
Sin embargo, existe la posibilidad de crear datasets con las imágenes que se deseen, este proceso está detallado en la sección de ***Entrenamiento***. 

#### Wikiart Dataset Completo
Adicional a esto, se puede descargar el dataset de Wikiart completo [aquí](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip). Este dataset contiene una gran variedad de corrientes artísticos y fue utilizado para el entrenamiento de [ArtGAN](https://github.com/cs-chan/ArtGAN). Para entrenar con alguna de las corrientes en este dataset, es necesario primero aplicar el preprocesamiento descrito en la sección siguiente.

#### Modelos pre-entrenados
Para facilitar la utilización de edge2art están disponibles 3 modelos pre entrenados para su descarga, estos modelos se estan "congelados" y se pueden descargar en las siguientes ligas, estos modelos se deben guardar dentro de una carpeta llamada "frozen_models":
* [Rococó](https://drive.google.com/open?id=1EMYiRRHVmGDPkruFzhvVijlH3effR2pH)
* [Ukiyo-e](https://drive.google.com/open?id=1gBifqL0b1wnrtVJCSiWIqe46wcwg6vwI)
* [Fauvismo](https://drive.google.com/open?id=1ZSYB4CqPyRmr0xNjvK25-UpXYt6RInuT)
* [Van Gogh](https://drive.google.com/open?id=1cCL8K9OUha6ME7l_jBYva7bmU-5tePAe)

## Entrenamiento: 
### Instrucciones:
Clonar el repositorio:
```sh
# Clonar el repositorio
git clone https://github.com/gallardorafael/edge2art.git
cd edge2art
``` 

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

La estructura general de las imágenes de entrenamiento es la siguiente:
![atob](https://github.com/gallardorafael/edge2art/blob/master/docs/ab.png)

Por ejemplo:


![ejemplo](https://github.com/gallardorafael/edge2art/blob/master/docs/abejemplo.png)

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
edge2art tiene una GUI simple que guía al usuario por sus 3 funcionalidades distintas, no es necesario nada más que añadir los archivos de entrada y observar como se renderiza el resultado.

![gui](https://github.com/gallardorafael/edge2art/blob/master/docs/gui.png)

Correr la GUI de edge2art:
```sh
python3 edge2art.py
```

