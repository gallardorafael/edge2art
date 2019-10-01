import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
from imutils import video

CROP_SIZE = 256
DOWNSAMPLE_RATIO = 4

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["photo", "video", "realtime"])
parser.add_argument("--style", required=True, choices=["rococo", "ukiyo", "fauvism", "vangogh"])
parser.add_argument("--input_dir", required=False, help="PATH de la imagen o el video a traducir")
parser.add_argument("--output_dir", required=False, help="PATH en donde guardar la imagen o el video")
a = parser.parse_args()


def load_graph(graph_filename):
    #Carga el modelo en la memoria.
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)

def resize_in(image):
    # Tamaño a 256x256
    height, width = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize

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

def real_time():
    # TensorFlow
    if a.style == "rococo":
        graph = load_graph('frozen_models/frozen_rococo.pb')
    elif a.style == "ukiyo":
        graph = load_graph('frozen_models/frozen_ukiyo.pb')
    elif a.style == "vangogh":
        graph = load_graph('frozen_models/frozen_vg.pb')
    elif a.style == "fauvism":
        graph = load_graph('frozen_models/frozen_fauvism.pb')
    else:
        print("ERROR al seleccionar el ESTILO")
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # OpenCV
    cap = cv2.VideoCapture(0)
    fps = video.FPS().start()
    while True:
        # Obtenemos el frame.
        ret, frame = cap.read()
        # Se reduce el tamaño del frame a uno procesable por pix2pix
        frame_resize = resize_out(frame)
        # Se aplica pre procesamiento del frame.
        gray_image = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        # Se extraen los bordes.
        edge = 255 -  auto_canny(gaussian_image)
        edge_color = edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        black_image = np.zeros(edge.shape, np.uint8)
        # Se genera la predicción.
        combined_image = np.concatenate([edge, black_image], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([frame_resize, edge_color, image_bgr], axis=1)

        cv2.imshow('Tiempo real', image_normal)

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()

    sess.close()
    cap.release()
    cv2.destroyAllWindows()


def load_image():
    # Ruta de la imagen a transformar
    image_path = a.input_dir
    # Cambiamos el tamaño de la imagen a 256x256 px
    # Salvamos la imagen original.
    image = resize_out(cv2.imread(image_path))
    # Pre procesamos la imagen
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    # Obtenemos los bordes de la imagen
    edge = 255 -  auto_canny(gaussian_image)
    edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    black_image = np.zeros(edge.shape, np.uint8)
    # TensorFlow
    if a.style  == "rococo":
        graph = load_graph('frozen_models/frozen_rococo.pb')
    elif a.style == "ukiyo":
        graph = load_graph('frozen_models/frozen_ukiyo.pb')
    elif a.style == "vangogh":
        graph = load_graph('frozen_models/frozen_vg.pb')
    elif a.style == "fauvism":
        graph = load_graph('frozen_models/frozen_fauvism.pb')
    else:
        print("ERROR al seleccionar el ESTILO")

    # Se cargan los tensores.
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # Se genera la predicción.
    combined_image = np.concatenate([edge, black_image], axis=1)
    image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
    generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
    image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
    image_normal = np.concatenate([image, edge_color, image_bgr], axis=1)
    save_path = a.output_dir
    cv2.imwrite(save_path,image_normal)
    cv2.imshow('Output', cv2.imread(save_path))
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def translate_video():
        # TensorFlow
        if a.style  == "rococo":
            graph = load_graph('frozen_models/frozen_rococo.pb')
        elif a.style == "ukiyo":
            graph = load_graph('frozen_models/frozen_ukiyo.pb')
        elif a.style == "vangogh":
            graph = load_graph('frozen_models/frozen_vg.pb')
        elif a.style == "fauvism":
            graph = load_graph('frozen_models/frozen_fauvism.pb')
        else:
            print("ERROR al seleccionar el ESTILO")
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        output_tensor = graph.get_tensor_by_name('generate_output/output:0')
        sess = tf.Session(graph=graph)

        # Directorio del archivo.
        video_path = a.input_dir

        #Definición del CODEC
        cap = cv2.VideoCapture(video_path)
        save_path = a.output_dir
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        out = cv2.VideoWriter(save_path, fourcc, 25, (768,256))

        # OpenCV

        if cap.isOpened() == False:
            print('Imposible obtener los datos del video.')
        while True:
            # Obtenemos el frame.
            ret, frame = cap.read()
            if ret == True:
                # Se reduce el tamaño del frame a uno procesable por pix2pix
                frame_resize = resize_out(frame)
                # Se aplica pre procesamiento del frame.
                gray_image = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
                gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
                # Se extraen los bordes.
                edge = 255 -  auto_canny(gaussian_image)
                edge_color = edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                black_image = np.zeros(edge.shape, np.uint8)
                # Se genera la predicción.
                combined_image = np.concatenate([edge, black_image], axis=1)
                image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
                generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
                image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
                image_normal = np.concatenate([frame_resize, edge_color, image_bgr], axis=1)

                #Se escribe el cuadro en al salida.
                out.write(image_normal)

                cv2.imshow('Procesando...', image_normal)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        sess.close()
        cap.release()
        out.release()
        print('Ha terminado la traducción.')
        cv2.destroyAllWindows()

if __name__ == "__main__":

    if a.mode == "photo":
        load_image()
    elif a.mode == "video":
        translate_video()
    elif a.mode == "realtime":
        real_time()
    else:
        print("ERROR en la selección de MODO")
