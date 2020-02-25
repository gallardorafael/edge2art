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
parser.add_argument("--input_dir", required=False, help="PATH to the image/video to translate")
parser.add_argument("--output_dir", required=False, help="PATH to the folder to save the outputs")
a = parser.parse_args()


def load_graph(graph_filename):
    # Loading model in memory
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
        # crop to correct ratio
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
    if a.style == "rococo":
        graph = load_graph('frozen_models/frozen_rococo.pb')
    elif a.style == "ukiyo":
        graph = load_graph('frozen_models/frozen_ukiyo.pb')
    elif a.style == "vangogh":
        graph = load_graph('frozen_models/frozen_vg.pb')
    elif a.style == "fauvism":
        graph = load_graph('frozen_models/frozen_fauvism.pb')
    else:
        print("ERROR: Select the correct STYLE")
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # OpenCV
    cap = cv2.VideoCapture(0)
    fps = video.FPS().start()
    while True:
        # Getting the actual frame
        ret, frame = cap.read()
        # Reducing the image size to 256x256
        frame_resize = resize_out(frame)
        # Pre-processing of the frame
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
    image_path = a.input_dir
    # Resizing image to 256x256 and saving original image
    image = resize_out(cv2.imread(image_path))
    # PPre-processing image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    # Extracting edges
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
        print("ERROR: Select the correct STYLE")

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
            print("ERROR: Select the correct STYLE")
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        output_tensor = graph.get_tensor_by_name('generate_output/output:0')
        sess = tf.Session(graph=graph)

        # Path to the video file
        video_path = a.input_dir

        # Defining the video codec
        cap = cv2.VideoCapture(video_path)
        save_path = a.output_dir
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        out = cv2.VideoWriter(save_path, fourcc, 25, (768,256))

        # OpenCV

        if cap.isOpened() == False:
            print('ERROR: Imposible to get the video data.')
        while True:
            # Getting the actual frame
            ret, frame = cap.read()
            if ret == True:
                # Resizing frame to 256x256
                frame_resize = resize_out(frame)
                # SPre-processing the frame
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

if __name__ == "__main__":

    if a.mode == "photo":
        load_image()
    elif a.mode == "video":
        translate_video()
    elif a.mode == "realtime":
        real_time()
    else:
        print("ERROR: Select the correct MODE")
