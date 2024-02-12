import configparser
import os
import sys

# import ocve
OCVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocve"))
sys.path.append(OCVE_DIR)
print("OCVE_DIR:", OCVE_DIR)
import ocve

CONFIG = configparser.ConfigParser()
PWD = os.getcwd()

def apply_randomisation():
    pass

def recurisve_dir_search(path):
    for (root, dirs, files) in os.walk(path):
        for dir in dirs:
            recurisve_dir_search(os.path.join(root, dir))
        for file in files:
            print(os.path.join(root, file))

def load_cfg():
    CONFIG.read("sample.ini")

def save_cfg():
    CONFIG["DEFAULT"] = {
        "brick_data_dir": os.path.join(PWD, "data/bricks/joosthazelzet"),
        "out_width": "600",
        "out_height": "600",
        "scale_min": "0.5",
        "scale_max": "1.5",
        "scale_ratio_min": "0.5",
        "scale_ratio_max": "2",
        "s&p_strength": "0.05",
        "guassian_mean": "",
        "guassian_std": "",
        "": "",
    }
    with open("sample.ini", "w") as configfile:
        CONFIG.write(configfile)

def main():
    save_cfg()
    load_cfg()
    recurisve_dir_search(CONFIG["DEFAULT"]["brick_data_dir"])
    # save_cfg()

if __name__ == "__main__":
    main()