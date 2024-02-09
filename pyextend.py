import configparser
import os
import sys

# import ocve
OCVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocve"))
sys.path.append(OCVE_DIR)
print("OCVE_DIR:", OCVE_DIR)
import ocve

def recurisve_dir_search(path):
    pass

if __name__ == "__main__":
    PWD = os.getcwd()
    config = configparser.ConfigParser()

    config["DEFAULT"] = {
        "root_data": os.path.join(PWD, "data"),
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
        config.write(configfile)
