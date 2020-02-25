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

# Desing of the main windows
root = tk.Tk()
root.title("edge2art")
root.geometry("256x256")
root.resizable(0,0)

or_bg_image = Image.open("gui/backgrounds/"+str(randint(1,20))+".png")
bg_image = ImageTk.PhotoImage(or_bg_image)
background = tk.Label(root, image=bg_image)
background.place(x=0, y=0, relwidth=1, relheight=1)

# Select style (model)
var = tk.StringVar(root)
var.set("Select a style")
dropDownMenu = tk.OptionMenu(root, var,
                                "Ukiyo-e",
                                "Rococo",
                                "Fauvism",
                                "Van Gogh",
                                )
dropDownMenu.place(x = 37, y = 20)

def load_graph(graph_filename):
    # Load the pre-trained model into the memory
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
    # Size to 256x256
    height, width = image.shape
    if height != width:
        # Crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize

def resize_out(image):
    # Size to 256x256
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
    if var.get() == "Rococo":
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
        # Get the actual frame
        ret, frame = cap.read()
        # Reducing the size of the frame to 256x256
        frame_resize = resize_out(frame)
        # Pre-processing the frame
        gray_image = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        # Extracting the borders (edges)
        edge = 255 -  auto_canny(gaussian_image)
        edge_color = edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        black_image = np.zeros(edge.shape, np.uint8)
        # SGenerating predictions
        combined_image = np.concatenate([edge, black_image], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([frame_resize, edge_color, image_bgr], axis=1)

        cv2.imshow('Real time', image_normal)

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()

    sess.close()
    cap.release()
    cv2.destroyAllWindows()


def load_image():
    # Path to the image to translate
    image_path = askopenfilename()
    # Size of the image to 256x256
    # Saving the original image
    image = resize_out(cv2.imread(image_path))
    # Pre-processing image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    # Getting edges
    edge = 255 -  auto_canny(gaussian_image)
    edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    black_image = np.zeros(edge.shape, np.uint8)
    # TensorFlow
    if var.get() == "Rococo":
        graph = load_graph('frozen_models/frozen_rococo.pb')
    elif var.get() == "Ukiyo-e":
        graph = load_graph('frozen_models/frozen_ukiyo.pb')
    else:
        graph = load_graph('frozen_models/frozen_fauvism.pb')

    # Loading tensors
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # Generating predictions
    combined_image = np.concatenate([edge, black_image], axis=1)
    image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
    generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
    image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
    image_normal = np.concatenate([image, edge_color, image_bgr], axis=1)
    save_path = askdirectory(title='Folder to save:')
    save_path = save_path+"/"+var.get()+"_transformed.png"
    cv2.imwrite(save_path,image_normal)
    cv2.imshow('Output', cv2.imread(save_path))
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def translate_video():
        # TensorFlow
        if var.get() == "Rococo":
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

        # DPath to the video file
        video_path = askopenfilename()

        # Defining the video codec
        cap = cv2.VideoCapture(video_path)
        save_path = askdirectory(title='Folder to save:')
        save_path = save_path+"/video_transformed.avi"
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        out = cv2.VideoWriter(save_path, fourcc, 25, (768,256))

        # OpenCV

        if cap.isOpened() == False:
            print('Imposible to get the video data')
        while True:
            # Getting the actual frame
            ret, frame = cap.read()
            if ret == True:
                # Reducing the size to 256x256
                frame_resize = resize_out(frame)
                # Pre-processing to the frame
                gray_image = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
                gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
                # Extracting edges
                edge = 255 -  auto_canny(gaussian_image)
                edge_color = edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                black_image = np.zeros(edge.shape, np.uint8)
                # Generating predictions
                combined_image = np.concatenate([edge, black_image], axis=1)
                image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
                generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
                image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
                image_normal = np.concatenate([frame_resize, edge_color, image_bgr], axis=1)

                # Ploting the frame
                out.write(image_normal)

                cv2.imshow('Processing...', image_normal)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        sess.close()
        cap.release()
        out.release()
        print('Finished!')
        cv2.destroyAllWindows()

# UI elements
# Real time button
btn_tiempo_real = tk.Button(root, text="Real time", command = real_time).place(x=37, y = 80)

# Image translation button
btn_insertar_imagen = tk.Button(root, text="Image translation", command = load_image).place(x=37, y=140)

# Video translation button
btn_video = tk.Button(root, text="Video translation", command = translate_video).place(x=37, y=190)


root.mainloop()
