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

# import ocve
OCVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocve"))
sys.path.append(OCVE_DIR)
import ocve

CONFIG = configparser.ConfigParser()
PROFILE = CONFIG["DEFAULT"]
PWD = os.getcwd()

DIR_COUNT = 0
FILE_COUNT = 0

BACKGROUND_DF = pd.read_csv("data/unsplash/photos.tsv000", delimiter="\t")

IMG_URLS = BACKGROUND_DF["photo_image_url"]
OUTPUT_RES = (600, 600)

def apply_transform(in_path, out_path):
    print(f"DEBUG: Generating @ {out_path}")

    # convert to RGBA
    img = ocve.read_img(in_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = ocve.colour_filter(img, (70, 70, 70, 255), (72, 72, 72, 255))
    # plt.imshow(img)
    # plt.show()
    
    # background image, try 3 times
    for i in range(5):
        print(f"attempt {i}")
        t = time.time()
        try:
            # TODO:
            # add bg img size check to prevent large images wasting time
            bg_url = IMG_URLS[int(random.random() * len(IMG_URLS))]
            with urllib.request.urlopen(bg_url) as url_response:
                img_array = np.asarray(bytearray(url_response.read()), dtype=np.uint8)
            bg = cv2.imdecode(img_array, -1)
            print(f"Image streamed @ {time.time() - t}")
            t = time.time()
            
            w, h = OUTPUT_RES
            x = int(random.random() * (bg.shape[0] - w))
            y = int(random.random() * (bg.shape[1] - h))
            bg = bg[y:y+h, x:x+w]
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
            
            blur_k_min = int(PROFILE["bg_blur_min"])
            blur_k_max = int(PROFILE["bg_blur_max"])
            k = random.randint(blur_k_min, blur_k_max)
            if (k <= 1):
                if k % 2 == 0:
                    k += 1
                bg = cv2.blur(bg, (k, k))
            
            # mix
            x_sub = int(random.random() * (w - img.shape[0]))
            y_sub = int(random.random() * (h - img.shape[1]))
            img = ocve.insert_image(bg, img, x_sub, y_sub)
            print(f"Insert img @ {time.time() - t}")
            t = time.time()
            
            break
        except Exception as e:
            print(f"ERROR: Background image url stream broke ({i}) @ {e}")
     
    # add noise
    img = ocve.gaussian_noise(
        img,
        strength=0.8,
        mean=0,
        variance=800)
    img = ocve.poisson_noise(img)
    img = ocve.salt_pepper_noise(img, freq=0.02, b_w=True)
    print(f"Noise added @ {time.time() - t}")
    t = time.time()
    
    # flip on x
    if (random.randint(0, 1)):
        img = cv2.flip(img, 0)
    # flip on y
    if (random.randint(0, 1)):
        img = cv2.flip(img, 1)
    
    ocve.save_img(out_path, img)
    print(f"Save img @ {time.time() - t}")
    t = time.time()
    print("\n\n")
            
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
            rel_path = os.path.join(os.path.basename(root), file)
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_dir, rel_path)
            apply_transform(in_path, out_path)

def load_cfg():
    CONFIG.read("sample.ini")

def save_cfg():
    CONFIG["DEFAULT"] = {
        "brick_data_dir": os.path.join(PWD, "data/bricks/v6"),
        "output_data_dir": os.path.join(PWD, "data/output"),
        "out_width": "300",
        "out_height": "300",
        "scale_min": "0.5",
        "scale_max": "0.8",
        "scale_ratio_min": "0.7",
        "scale_ratio_max": "1.5",
        "s&p_strength": "0.05",
        "guassian_mean": "0.5",
        "guassian_std": "0.05",
        "bg_blur_min": "0",
        "bg_blur_max": "5"
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