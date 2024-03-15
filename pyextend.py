import configparser
import os
import sys
import time
import pandas as pd
import random as r
import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import copy

from functions import *
from setup_env import setup_unsplash

DIR_COUNT = 0
FILE_COUNT = 0

def str_to_int_tuple(s):
    return tuple(map(int, s.split(',')))

def load_brick(brick_path):
    img = load_img(brick_path)

    out_w = int(CFG["GENERAL"]["out_width"])
    out_h = int(CFG["GENERAL"]["out_height"])

    ar_min = float(CFG["BRICK"]["ar_min"])
    ar_max = float(CFG["BRICK"]["ar_max"])
    ar = r.normalvariate(mu=1.0, sigma=0.05)

    brick_w, brick_h = 32, 32
    if ar >= 1.0:
        brick_w = out_w
        brick_h = int(out_h / ar)
    else:
        brick_w = int(out_w / ar)
        brick_h = out_w

    scl_min = float(CFG["BRICK"]["scale_min"])
    scl_max = float(CFG["BRICK"]["scale_max"])
    scl = r.random() * (scl_max - scl_min) + scl_min

    brick_w = int(brick_w * scl)
    brick_h = int(brick_h * scl)

    img = cv2.resize(img, (brick_w, brick_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bg_colour_min = str_to_int_tuple(CFG["BRICK"]["bg_colour_min"])
    bg_colour_max = str_to_int_tuple(CFG["BRICK"]["bg_colour_max"])
    img = colour_filter(img, bg_colour_min, bg_colour_max)
    return img

def load_background():
    bg_dir = CFG["PATHS"]["backgrounds"]
    bg_filename = r.choice(os.listdir(bg_dir))
    bg_path = os.path.join(bg_dir, bg_filename)

    out_w = int(CFG["GENERAL"]["out_width"])
    out_h = int(CFG["GENERAL"]["out_height"])

    img = load_img(bg_path)
    if img.shape[0] < out_w or img.shape[1] < out_h:
        img = cv2.resize(img, img.shape*2)
    
    # subset image
    img = random_sub_image(img, out_w, out_h)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # bg blur
    if int(CFG["BACKGROUND"]["blurring"]):
        blur_k_min = int(CFG["BACKGROUND"]["blur_min"])
        blur_k_max = int(CFG["BACKGROUND"]["blur_max"])
        k = r.randint(blur_k_min, blur_k_max)
        if (k <= 1):
            if k % 2 == 0:
                k += 1
            bg = cv2.blur(bg, (k, k))
            
    return img

def post_processing(img):
    # add noise
    if int(CFG["POSTPROCESS"]["gaussian_noise"]):
        g_min = float(CFG["POSTPROCESS"]["gaussian_strength_min"])
        g_max = float(CFG["POSTPROCESS"]["gaussian_strength_max"])
        g_strength = (r.random() * (g_max - g_min)) + g_min
        g_std_min = float(CFG["POSTPROCESS"]["gaussian_std_min"])
        g_std_max = float(CFG["POSTPROCESS"]["gaussian_std_max"])
        g_std = (r.random() * (g_std_max - g_std_min)) + g_std_min
        img = gaussian_noise(
            img,
            strength=g_strength,
            mean=0,
            std=g_std)
        
    if int(CFG["POSTPROCESS"]["sp_bw_noise"]):
        freq = int(CFG["POSTPROCESS"]["sp_bw_frequency"])
        img = salt_pepper_noise(img, freq=freq, b_w=True)
        
    if int(CFG["POSTPROCESS"]["sp_rgb_noise"]):
        freq = int(CFG["POSTPROCESS"]["sp_rgb_frequency"])
        img = salt_pepper_noise(img, freq=freq, b_w=False)
    
    if int(CFG["POSTPROCESS"]["brighten"]):
        b_min = int(CFG["POSTPROCESS"]["brightness_min"])
        b_max = int(CFG["POSTPROCESS"]["brightness_max"])
        b = r.randint(b_min, b_max)
        img = brightness(img, b)
    
    if int(CFG["POSTPROCESS"]["contrast"]):
        c_min = float(CFG["POSTPROCESS"]["contrast_min"])
        c_max = float(CFG["POSTPROCESS"]["contrast_max"])
        c = r.uniform(c_min, c_max)
        img = contrast(img, c)

    if int(CFG["POSTPROCESS"]["mirroring"]):
        if (r.randint(0, 1)):
            img = cv2.flip(img, 0)
        if (r.randint(0, 1)):
            img = cv2.flip(img, 1)
    return img

def apply_transform(in_path, out_path):
    print(f"DEBUG: Generating @ {out_path}")
    
    img = load_brick(in_path)
    bg = load_background()
    
    img = random_insert_image(bg, img)
    img = post_processing(img)
    save_img(out_path, img)
            
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
            for i in range(int(CFG["GENERAL"]["gens_per_base"])):
                filename = base + f"_{i}" + ext
                out_path = os.path.join(out_dir, rel_dir, filename)
                apply_transform(in_path, out_path)

def load_cfg():
    global CFG, CFG_NAME
    CFG = configparser.ConfigParser()
    path = "sample.ini"
    if len(sys.argv) != 1:
        path = sys.argv[1]
    if not os.path.exists(path):
        save_cfg()
    CFG.read(path)
    CFG_NAME = os.path.basename(path).split(".")[0] 

def save_cfg():
    CFG["GENERAL"] = {
        "gens_per_base": "2",
        # output resolution (ai expects: 224 x 224 x 3)
        "out_width": "224",
        "out_height": "224",
    }
    CFG["PATHS"] = {
        "bricks": "data/bricks/10_bricks",
        "backgrounds": "data/backgound",
        "unsplash": "data/unsplash",
        "output": "data/output", 
    }
    CFG["BACKGROUND"] = {
        "blurring": "1",
        "blur_min": "0",
        "blur_max": "11",
    }
    CFG["BRICK"] = {
        "ar_warping": "1",
        "ar_min": "0.9",
        "ar_max": "1.1",
        
        "scaling": "1",
        "scale_min": "0.5",
        "scale_max": "1.0",
        
        "bg_colour_min": "0, 0, 0, 255",
        "bg_colour_max": "2, 2, 2, 255",
    }
    CFG["POSTPROCESS"] = {
        "sp_bw_noise": "1",
        "sp_bw_frequency": "0.02",
        "sp_rgb_noise": "1",
        "sp_rgb_frequency": "0.02",
        
        "gaussian_noise": "1",
        "gaussian_strength_min": "0.0",
        "gaussian_strength_max": "1.0",
        "gaussian_std_min": "10",
        "gaussian_std_max": "25",

        "brighten": "1",
        "brightness_min": "-20",
        "brightness_max": "20",

        "contrast": "1",
        "contrast_min": "0.8",
        "contrast_max": "1.2",
        
        "mirroring": "1",
    }
    with open("sample.ini", "w") as configfile:
        CFG.write(configfile)

def main():
    start_app = time.time()

    load_cfg()
    # download background images
    setup_unsplash(path_bg=CFG["PATHS"]["backgrounds"], path_csv=CFG["PATHS"]["unsplash"])
    
    # setup and create output dir
    if not os.path.exists(CFG["PATHS"]["output"]):
        os.mkdir(CFG["PATHS"]["output"])
    out_dir = os.path.join(CFG["PATHS"]["output"], CFG_NAME)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir_search(CFG["PATHS"]["bricks"], out_dir)
    
    end_app = time.time()
    dt = end_app-start_app
    
    try:
        print(f"Program Duration = {dt:.2f}s @ {dt/FILE_COUNT:.5f}")
        print(f"Data = {FILE_COUNT} files in {DIR_COUNT} dirs")
    except ZeroDivisionError:
        print(f"Time = {dt:.5f}")


if __name__ == "__main__":
    main()
