import configparser
import os
import sys
import time
import pandas as pd
import random
import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from setup_env import setup_unsplash
import copy

# import ocve
OCVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocve"))
sys.path.append(OCVE_DIR)
import ocve

CONFIG = configparser.ConfigParser()
PROFILE = CONFIG["DEFAULT"]

DIR_COUNT = 0
FILE_COUNT = 0

# download background images
setup_unsplash()

def load_brick(brick_path):
    img = ocve.read_img(brick_path)

    out_w = int(PROFILE["out_width"])
    out_h = int(PROFILE["out_height"])

    ar_min = float(PROFILE["brick_ar_min"])
    ar_max = float(PROFILE["brick_ar_max"])
    ar = np.random.normal(loc=1.0, scale=0.05)

    brick_w, brick_h = 32, 32
    if ar >= 1.0:
        brick_w = out_w
        brick_h = int(out_h / ar)
    else:
        brick_w = int(out_w / ar)
        brick_h = out_w

    scl_min = float(PROFILE["brick_scale_min"])
    scl_max = float(PROFILE["brick_scale_max"])
    scl = random.random() * (scl_max - scl_min) + scl_min

    brick_w = int(brick_w * scl)
    brick_h = int(brick_h * scl)

    img = cv2.resize(img, (brick_w, brick_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = ocve.colour_filter(img, (70, 70, 70, 255), (72, 72, 72, 255))
    return img

def load_background():
    bg_dir = PROFILE["background_dir"]
    bg_filename = random.choice(os.listdir(bg_dir))
    bg_path = os.path.join(bg_dir, bg_filename)

    out_w = int(PROFILE["out_width"])
    out_h = int(PROFILE["out_height"])

    img = ocve.read_img(bg_path)
    if img.shape[0] < out_w or img.shape[1] < out_h:
        img = cv2.resize(img, img.shape*2)
    
    # subset image
    img = ocve.random_sub_image(img, out_w, out_h)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # bg blur
    blur_k_min = int(PROFILE["bg_blur_min"])
    blur_k_max = int(PROFILE["bg_blur_max"])
    k = random.randint(blur_k_min, blur_k_max)
    if (k <= 1):
        if k % 2 == 0:
            k += 1
        bg = cv2.blur(bg, (k, k))
            
    return img

def post_processing(img):
    # add noise
    g_min = float(PROFILE["gaussian_strength_min"])
    g_max = float(PROFILE["gaussian_strength_max"])
    g_strength = (random.random() * (g_max - g_min)) + g_min
    g_std_min = float(PROFILE["gaussian_std_min"])
    g_std_max = float(PROFILE["gaussian_std_max"])
    g_std = (random.random() * (g_std_max - g_std_min)) + g_std_min
    img = ocve.gaussian_noise(
        img,
        strength=g_strength,
        mean=0,
        std=g_std)
    img = ocve.salt_pepper_noise(img, freq=0.02, b_w=True)
    
    # flip on axis
    if (random.randint(0, 1)):
        img = cv2.flip(img, 0)
    if (random.randint(0, 1)):
        img = cv2.flip(img, 1)
    return img

def apply_transform(in_path, out_path):
    print(f"DEBUG: Generating @ {out_path}")
    
    img = load_brick(in_path)
    bg = load_background()
    
    img = ocve.random_insert_image(bg, img)
    img = post_processing(img)
    ocve.save_img(out_path, img)
            
def dir_search(in_dir, out_dir):
    global DIR_COUNT, FILE_COUNT
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for (root, dirs, files) in os.walk(in_dir):
        for dir in dirs:
            DIR_COUNT += 1
            class_dir = os.path.join(out_dir, dir)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
        for file in files:
            FILE_COUNT += 1
            base, ext = os.path.splitext(file)
            rel_dir = os.path.basename(root) 
            in_path = os.path.join(root, file)
            for i in range(int(PROFILE["gens_per_base"])):
                filename = base + f"_{i}" + ext
                out_path = os.path.join(out_dir, rel_dir, filename)
                apply_transform(in_path, out_path)

def load_cfg():
    CONFIG.read("sample.ini")

def save_cfg():
    CONFIG["DEFAULT"] = {
        # input data
        "brick_dir": "data/bricks/2_bricks",
        "output_dir": "data/output",
        "background_dir": "data/unsplash",
        "gens_per_base": "2",
        
        # image composition
        "brick_ar_min": "0.9",
        "brick_ar_max": "1.1",
        "brick_scale_min": "0.5",
        "brick_scale_max": "1.0",
        
        # post processing noise
        "s&p_strength": "0.05",
        "gaussian_strength_min": "0.0",
        "gaussian_strength_max": "1.0",
        "gaussian_std_min": "10",
        "gaussian_std_max": "25",
        "bg_blur_min": "0",
        "bg_blur_max": "11",
        
        # output resolution (ai expects: 224 x 224 x 3)
        "out_width": "224",
        "out_height": "224",
    }
    with open("sample.ini", "w") as configfile:
        CONFIG.write(configfile)

def main():
    start_app = time.time()

    save_cfg()
    load_cfg()
    
    dir_search(PROFILE["brick_data_dir"], PROFILE["output_data_dir"])
    
    end_app = time.time()
    dt = end_app-start_app
    
    try:
        print(f"Program Duration = {dt:.2f}s @ {dt/FILE_COUNT:.5f}")
        print(f"Data = {FILE_COUNT} files in {DIR_COUNT} dirs")
    except ZeroDivisionError:
        print(f"Time = {dt:.5f}")


if __name__ == "__main__":
    main()
