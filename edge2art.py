import tkinter as tk
from random import randint
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
from imutils import video

CROP_SIZE = 256
DOWNSAMPLE_RATIO = 4
global preview

# Diseño de la ventana principal.
root = tk.Tk()
root.title("edge2art")
root.geometry("256x256")
root.resizable(0,0)

or_bg_image = Image.open("gui/backgrounds/"+str(randint(1,20))+".png")
bg_image = ImageTk.PhotoImage(or_bg_image)
background = tk.Label(root, image=bg_image)
background.place(x=0, y=0, relwidth=1, relheight=1)

# Seleccionar estilo(modelo)
var = tk.StringVar(root)
var.set("Selecciona un estilo")
dropDownMenu = tk.OptionMenu(root, var,
                                "Ukiyo-e",
                                "Rococó",
                                "Fauvismo",
                                "Van Gogh",
                                )
dropDownMenu.place(x = 37, y = 20)

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
    if var.get() == "Rococó":
        graph = load_graph('frozen_models/frozen_rococo.pb')
    elif var.get() == "Ukiyo-e":
        graph = load_graph('frozen_models/frozen_ukiyo.pb')
    else:
        graph = load_graph('frozen_models/frozen_fauvism.pb')
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
    image_path = askopenfilename()
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
    if var.get() == "Rococó":
        graph = load_graph('frozen_models/frozen_rococo.pb')
    elif var.get() == "Ukiyo-e":
        graph = load_graph('frozen_models/frozen_ukiyo.pb')
    else:
        graph = load_graph('frozen_models/frozen_fauvism.pb')

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
    save_path = askdirectory(title='Directorio de guardado')
    save_path = save_path+"/"+var.get()+"_transformed.png"
    cv2.imwrite(save_path,image_normal)
    cv2.imshow('Output', cv2.imread(save_path))
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def translate_video():
        # TensorFlow
        if var.get() == "Rococó":
            graph = load_graph('frozen_models/frozen_rococo.pb')
        elif var.get() == "Ukiyo-e":
            graph = load_graph('frozen_models/frozen_ukiyo.pb')
        elif var.get() == "Van Gogh":
            graph = load_graph('frozen_models/frozen_vg.pb')
        else:
            graph = load_graph('frozen_models/frozen_fauvism.pb')
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        output_tensor = graph.get_tensor_by_name('generate_output/output:0')
        sess = tf.Session(graph=graph)

        # Directorio del archivo.
        video_path = askopenfilename()

        #Definición del CODEC
        cap = cv2.VideoCapture(video_path)
        save_path = askdirectory(title='Directorio de guardado')
        save_path = save_path+"/video_transformed.avi"
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

# Opciones de la interfaz.
# Funcionalidad de dibujo libre para transformar.
btn_tiempo_real = tk.Button(root, text="Tiempo Real", command = real_time).place(x=37, y = 80)

# Funcionalidad de insertar imagen para transformar.
btn_insertar_imagen = tk.Button(root, text="Traducir Imagen", command = load_image).place(x=37, y=140)

# Funcionalidad de guardar la Imagen
btn_video = tk.Button(root, text="Traducir Video", command = translate_video).place(x=37, y=190)


root.mainloop()
