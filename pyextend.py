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

def apply_transform(in_path, out_path):
    pass
    # print("IN:", in_path)
    # print("OUT:", out_path)
            
def dir_search(in_dir, out_dir):
    os.mkdir(out_dir)
    for (root, dirs, files) in os.walk(in_dir):
        for dir in dirs:
            os.mkdir(os.path.join(out_dir, dir))
        for file in files:
            rel_path = os.path.join(os.path.basename(root), file)
            in_path = os.path.join(root, file)
            out_path = os.path.join(out_dir, rel_path)
            apply_transform(in_path, out_path)

def load_cfg():
    CONFIG.read("sample.ini")

def save_cfg():
    CONFIG["DEFAULT"] = {
        "brick_data_dir": os.path.join(PWD, "data/bricks/joosthazelzet"),
        "output_data_dir": os.path.join(PWD, "data/output"),
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
    profile = CONFIG["DEFAULT"]
    dir_search(profile["brick_data_dir"], profile["output_data_dir"])
    # save_cfg()

if __name__ == "__main__":
    main()