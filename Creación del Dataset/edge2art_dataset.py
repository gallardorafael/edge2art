# Este script crea el dataset AtoB para entrenar los modelos.
# a_dir es el directorio con las imágenes A
# b_dir es el directorio con las imágenes B
# output_dir es el directorio dónde se guardarán los resultados.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import threading
import tfimage as im
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a_dir", required=True, help="Directorio de la carpeta con las images A de training")
parser.add_argument("--b_dir", required=True, help="Directorio de la carpeta con las images A de training")
parser.add_argument("--output_dir", required=True, help="Directorio donde guardar el dataset de entrenamiento")
a = parser.parse_args()

edge_pool = None

def create_paths():
    if not os.exists(a.a_dir):
        print("No existe el directorio A")
        return
    if not os.exists(a.b_dir):
        print("No existe el directorio B")
        return

    input_dir = a.a_dir
    b_dir = a.b_dir

    if not os.exists(a.output_dir):
        os.mkdir(a.output_dir)
    output_dir = a.output_dir

    input_paths = []
    output_paths = []
    b_dir_paths = []

    for style in os.listdir(input_dir):
        style_path = os.path.join(input_dir, style)
        input_paths.append(style_path)

    for style in os.listdir(input_dir):
        style_path = os.path.join(output_dir, style)
        output_paths.append(style_path)

    for style in os.listdir(b_dir):
        style_path = os.path.join(b_dir, style)
        b_dir_paths.append(style_path)

    return input_paths, output_paths, b_dir_paths

def combine(src, src_path, b_dir):
    # find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("Missing sibling for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        print("Error path: ", src_path)
        raise Exception("Different sizes")

    # convert both images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    # remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]

    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    return np.concatenate([src, sibling], axis=1)

complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now

def create_style_dataset(output_dir, input_dir, b_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src_paths = []
    dst_paths = []

    skipped = 0
    for src_path in im.find(input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(output_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)

    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths)

    print("processing %d files" % total)

    global start
    start = time.time()

    with tf.compat.v1.Session() as sess:
        for src_path, dst_path in zip(src_paths, dst_paths):
            src = im.load(src_path)
            dst = combine(src, src_path, b_dir)
            im.save(dst, dst_path)
            complete()

create_style_dataset(a.output_dir, a.a_dir, a.b_dir)
