# edge2art
This is the repository for the paper [edge2art: Edges to Artworks Translation with Conditional
Generative Adversarial Networks](http://www.aniei.org.mx/Archivos/Memorias/Libro_Electronico_CNCIIC2019.pdf), published in the book: [Avances en la Transformación Digital hacia la industria 4.0](http://www.aniei.org.mx/Archivos/Memorias/Libro_Electronico_CNCIIC2019.pdf), pp. 213--222.

Project based on the Pix2Pix model ***[Pix2Pix](https://phillipi.github.io/pix2pix/)***: Image to Image Translation with Condicional Generative Adversarial Networks

## Functionalities
***edge2art*** is able to translate any image into an artistic style by extrating the edges of the image in order to favor the generated content by the cGAN. The edge2art GUI, users can:
* Convert any image into an artwork
* See a real-time artistic version of their webcam video
* Translate common videos into artistic animations

The following GIFs are demonstrations of the results obtained by edge2art: 
![Fauvism](https://github.com/gallardorafael/edge2art/blob/master/docs/dodge_fauvism.gif)
![Ukiyo](https://github.com/gallardorafael/edge2art/blob/master/docs/robot_ukiyo.gif)

A fulll demonstration is available in the following youtube [video](https://youtu.be/7BrNVLDM1dE).

## Installation

### Requirements:
* tensorflow == 1.14.0 (o tensorflow-gpu 1.14.0)
* python3-tk
* opencv-python
* Pillow
* imutils
* numpy
* scipy

### Clone the repo:
```sh
# Cloning the repository
git clone https://github.com/gallardorafael/edge2art.git
cd edge2art
``` 
### Datasets
#### Example dataset
The example dataset is from the Rococo artistic style, it is available for download in: [Rococó](https://drive.google.com/open?id=1Q4lUnhlGs10tsJFisWfO4-HmFdbxFai1), this dataset has a total of 2089 Rococo artworks, the dataset is ready to begin the training phase with pix2pix in AtoB direction. Estimated training times are ahead: 
* TensorFlow with Intel Core i7 6700HQ: ~108 horas.
* TensorFlow with Nvidia GTX 960 : ~59 horas.
* TensorFlow with Nvidia Tesla K80: ~30 horas.
* TensorFlow with Nvidia Tesla T4: ~10 horas.

Consider the times before begin the training.

#### Create your own datasets
You can create your own datasets with every set of images you want. This process is full detailed in ***Training*** section.

##### Van Gogh Dataset
In order to obtain a Van Gogh dataset, the frames of the trailers of the movie "Loving Vincent" were extracted. Then, a pre-processing phase were applied before training. The Van Gogh dataset extracted from the movie is available for download [here](https://drive.google.com/open?id=11yYohJwZMdZzq7QRZtQqQF1QIeHCxlf6). Images in the dataset are already sized to 256x256 pixels.

If you want to extract the frames of every other video, the script to achieve this was included inside the dataset_creation folder.
```sh
cd dataset_creation
python3 extract_video_frames.py --input_video <path to the video file> --output_dir <path to save the frames>
```

#### Pre-trained models
In order to ease the test and use of edge2art, 4 pre-trained models are available to download and use. These models are frozen and can be downloaded in the following links.
* [Rococó](https://drive.google.com/open?id=1EMYiRRHVmGDPkruFzhvVijlH3effR2pH)
* [Ukiyo-e](https://drive.google.com/open?id=1gBifqL0b1wnrtVJCSiWIqe46wcwg6vwI)
* [Fauvismo](https://drive.google.com/open?id=1ZSYB4CqPyRmr0xNjvK25-UpXYt6RInuT)
* [Van Gogh](https://drive.google.com/open?id=1cCL8K9OUha6ME7l_jBYva7bmU-5tePAe)
NOTE: Pre-trained models should be saved into the "frozen_models" folder.

#### Full Wikiart dataset
The full wikiart dataset can be downloaded [here](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip). The full dataset contains a big variety of artistic styles and was used to train [ArtGAN](https://github.com/cs-chan/ArtGAN). If you want to train edge2art in any of these styles, you should first apply a pre-processing phase, described in the following section.

## Training: 
### Instructions:
Resize images to 256x256 pixels:
```sh
cd dataset_creation/
python3 preprocess.py --input_dir <path to the original images folder> --output_dir <path to save resized images> --operation resize
```

Extract the edges from the resized images:
```sh
cd dataset_creation/
python3 images2edges.py --input_dir <path to the resized images folder> --output_dir <path to save the extracted edges>
```

The structure of the training images is the following: 
![atob](https://github.com/gallardorafael/edge2art/blob/master/docs/ab.png)

For example:

![ejemplo](https://github.com/gallardorafael/edge2art/blob/master/docs/abejemplo.png)

Creating an AtoB dataset for pix2pix:
```sh
cd dataset_creation/
python3 edge2art_dataset.py --a_dir <path to the edges folder> --b_dir <path to the resized images> --output_dir <path to save the training images)>
```
 
Training the model:
```sh
python3 pix2pix.py --input_dir <path to the folder with training images> --output_dir <path to save the trained model> --mode train --max_epochs 200 --which_direction AtoB
```
NOTE: The max_epochs parameter depends on your own needs.

Reducing and frozing the model:
```sh
cd model_reduction/
python3 reduce_model.py --model-input <path to the folder with the trained model> --model-output <path to save the reduced model>

python3 freeze_model.py --model_folder <path to save the frozen model>
```
  ** The model will be saved as "frozen_model.pb" in the specified folder.
  ** You must move the frozen models to the "frozen_models" folder inside the edge2art folder.

## Testing edge2art

### Using the Tkinter GUI

edge2art includes an easy-to-use GUI with 3 functionalities: Real time visualization, image translation or video translation. Just select the desired style and have fun!

![gui](https://github.com/gallardorafael/edge2art/blob/master/docs/gui.png)

Running the edge2art GUI:
```sh
python3 edge2art.py
```

### Running edge2art from command line
edg2art also includes a python script, in case you don't want to use the GUI.

#### Options for the --style parameter:
* rococo
* ukiyo
* fauvism
* vangogh

Running edge2art_script in real time mode:
```sh
python3 edge2art_script.py --mode realtime --style <style>
```

Running edge2art_script in image translation mode:
```sh
python3 edge2art_script.py --mode photo --style <style> --input_dir <path to image file> --output_dir <path to save the translated image> 
```
Running edge2art_script in video translation mode:
```sh
python3 edge2art_script.py --mode video --style <style> --input_dir <path to video file> --output_dir <path to save the translated video> 
```

## Citations
If you use this work or the edge2art datasets and you would like to cite:
```
@inproceedings{Gallardo2019edge,
	title={edge2art: Edges to Artworks Translation with Conditional Generative Adversarial Networks},
	author={Gallardo-García, Rafael and Beltrán-Martínez, Beatriz and Cerón Garnica, Carmen and Vilariño Ayala, Darnes},
	booktitle={Avances en la Transformación Digital hacia la industria 4.0},
	pages={213--222},
	year={2018},
	organization={Congreso Nacional e Internacional de Informática y Computación de la ANIEI},
 	publisher={Alfaomega Grupo Editor}
}
```
