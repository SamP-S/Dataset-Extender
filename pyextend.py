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
PWD = os.getcwd()

DIR_COUNT = 0
FILE_COUNT = 0

# download background images
setup_unsplash()

def grab_sub_image(img, w, h):
    max_x = img.shape[0] - w
    max_y = img.shape[1] - h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return img[x:x+w, y:y+h]

def apply_transform(in_path, out_path):
    print(f"DEBUG: Generating @ {out_path}")
    
    t = time.time()

    # convert to RGBA
    img = ocve.read_img(in_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = ocve.colour_filter(img, (70, 70, 70, 255), (72, 72, 72, 255))
    
    print(f"img load {time.time() - t}")
    
    # load bg
    bg_filename = random.choice(os.listdir("data/unsplash"))
    bg_path = os.path.join("data/unsplash", bg_filename)
    bg_img = ocve.read_img(bg_path)
    bg = copy.deepcopy(bg_img)
    print(f"bg load {time.time() - t}")
    
    # bg grab sub img
    out_w = int(PROFILE["out_width"])
    out_h = int(PROFILE["out_height"])
    bg = grab_sub_image(bg, out_w, out_h)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    print(f"bg sub img {time.time() - t}")
    
    # bg blur
    blur_k_min = int(PROFILE["bg_blur_min"])
    blur_k_max = int(PROFILE["bg_blur_max"])
    k = random.randint(blur_k_min, blur_k_max)
    if (k <= 1):
        if k % 2 == 0:
            k += 1
        bg = cv2.blur(bg, (k, k))
    print(f"bg blur {time.time() - t}")
    
    # overlay brick over bg
    x_sub = random.randint(0, bg.shape[0] - img.shape[0])
    y_sub = random.randint(0, bg.shape[1] - img.shape[1])
    img = ocve.insert_image(bg, img, x_sub, y_sub)
    print(f"insert brick {time.time() - t}")
     
    # add noise
    g_min = float(PROFILE["gaussian_strength_min"])
    g_max = float(PROFILE["gaussian_strength_max"])
    g_strength = (random.random() * (g_max - g_min)) + float(g_min)
    g_std_min = float(PROFILE["gaussian_std_min"])
    g_std_max = float(PROFILE["gaussian_std_max"])
    g_std = (random.random() * (g_std_max - g_std_min)) + float(g_std_min)
    img = ocve.gaussian_noise(
        img,
        strength=g_strength,
        mean=0,
        std=g_std)
    img = ocve.poisson_noise(img)
    img = ocve.salt_pepper_noise(img, freq=0.02, b_w=True)
    
    # flip on x
    if (random.randint(0, 1)):
        img = cv2.flip(img, 0)
    # flip on y
    if (random.randint(0, 1)):
        img = cv2.flip(img, 1)
    
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
        "brick_data_dir": os.path.join(PWD, "data/bricks/26_bricks"),
        "output_data_dir": os.path.join(PWD, "data/output"),
        "gens_per_base": "10",
        
        # image composition
        "brick_ar_min": "0.5",
        "brick_ar_max": "2",
        "brick_scale_min": "0.3",
        "brick_scale_max": "1.0",
        
        # post processing noise
        "s&p_strength": "0.05",
        "gaussian_strength_min": "0.0",
        "gaussian_strength_max": "1.0",
        "gaussian_std_min": "10",
        "gaussian_std_max": "25",
        "bg_blur_min": "0",
        "bg_blur_max": "5",
        
        # output resolution (ai expects: 224 x 224 x 3)
        "out_width": "600",
        "out_height": "600",
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
