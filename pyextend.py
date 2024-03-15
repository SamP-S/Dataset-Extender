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

def load_brick(brick_path):
    print(f"Brick @ {brick_path}")
    img = load_img(brick_path)

    out_w = int(CFG["GENERAL"]["output_width"])
    out_h = int(CFG["GENERAL"]["output_height"])
    brick_w, brick_h = out_w, out_h
    print(f"brick w/h = {brick_w}/{brick_h}")

    if int(CFG["BRICK"]["ar_warping"]):
        ar_mean = float(CFG["BRICK"]["ar_mean"])
        ar_std = float(CFG["BRICK"]["ar_std"])
        ar = r.normalvariate(mu=ar_mean, sigma=ar_std)
        print(ar)
        if ar >= 1.0:
            brick_w = out_w
            brick_h = int(out_h / ar)
        else:
            brick_w = int(out_w / (2 - ar))
            brick_h = out_w
    print(f"post ar w/h = {brick_w}/{brick_h}")

    scl_min = float(CFG["BRICK"]["scale_min"])
    scl_max = float(CFG["BRICK"]["scale_max"])
    scl = r.uniform(scl_min, scl_max)
    
    brick_w = int(brick_w * scl)
    brick_h = int(brick_h * scl)
    print(f"post scale w/h = {brick_w}/{brick_h}")
    img = cv2.resize(img, (brick_w, brick_h))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = colour_filter(img, (70, 70, 70, 255), (72, 72, 72, 255))
    return img

def load_background():
    bg_dir = CFG["PATHS"]["backgrounds"]
    bg_filename = r.choice(os.listdir(bg_dir))
    bg_path = os.path.join(bg_dir, bg_filename)
    print(f"Background @ {bg_path}")

    out_w = int(CFG["GENERAL"]["output_width"])
    out_h = int(CFG["GENERAL"]["output_height"])

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
            img = cv2.blur(img, (k, k))
            
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
        freq = float(CFG["POSTPROCESS"]["sp_bw_frequency"])
        img = salt_pepper_noise(img, freq=freq, b_w=True)
        
    if int(CFG["POSTPROCESS"]["sp_rgb_noise"]):
        freq = float(CFG["POSTPROCESS"]["sp_rgb_frequency"])
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
    
    print(f"background @ {bg.shape} : brick @ {img.shape}")
    img = random_insert_image(bg, img)
    img = post_processing(img)
    save_img(out_path, img)
            
def dir_search(in_dir, out_dir):
    global DIR_COUNT, FILE_COUNT
    if not os.path.exists(in_dir):
        print(f"ERROR: Can't find input bricks @ {in_dir} -- exiting")
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
    global CFG
    CFG = configparser.ConfigParser()
    if len(sys.argv) == 1:
        if not os.path.exists("sample.ini"):
            save_cfg()
        CFG.read("sample.ini")
    else:
        CFG.read(sys.argv[1])

def save_cfg():
    CFG["GENERAL"] = {
        "gens_per_base": "2",
        # output resolution (ai expects: 224 x 224 x 3)
        "output_width": "224",
        "output_height": "224",
    }
    CFG["PATHS"] = {
        "bricks": "data/bricks/10_bricks",
        "backgrounds": "data/backgound",
        "unsplash": "data/unsplash",
        "output": "data/augmented", 
    }
    CFG["BACKGROUND"] = {
        "blurring": "1",
        "blur_min": "0",
        "blur_max": "11",
    }
    CFG["BRICK"] = {
        "ar_warping": "1",
        "ar_mean": "1.0",
        "ar_std": "0.05",
        
        "scaling": "1",
        "scale_min": "0.5",
        "scale_max": "1.0",
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
    
    dir_search(CFG["PATHS"]["bricks"], CFG["PATHS"]["output"])
    
    end_app = time.time()
    dt = end_app-start_app
    
    try:
        print(f"Program Duration = {dt:.2f}s @ {dt/FILE_COUNT:.5f}")
        print(f"Data = {FILE_COUNT} files in {DIR_COUNT} dirs")
    except ZeroDivisionError:
        print(f"Time = {dt:.5f}")


if __name__ == "__main__":
    main()
