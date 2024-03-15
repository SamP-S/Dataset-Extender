import requests
import zipfile
import os
import pandas as pd
import urllib
import numpy as np
import cv2

def setup_unsplash(path_bg, path_csv, num_photos=1000):
      
    # unsplash db paths
    zip_filename = "unsplash_lite.zip"
    zip_path = os.path.join(path_csv, )
    df_path = os.path.join(path_csv, "photos.tsv000")
       
    if not os.path.exists(path_csv):
        print(f"DEBUG: {path_csv} directory not found - creating")
        os.mkdir(path_csv)
        
    if not os.path.exists(path_bg):
        print(f"DEBUG: {path_bg} directory not found - creating")
        os.mkdir(path_bg)

    if not os.path.exists(zip_path):
        print(f"DEBUG: {zip_filename} not found - downloading")
        url = "https://unsplash.com/data/lite/latest"
        r = requests.get(url)
        open(zip_path , 'wb').write(r.content)

    with zipfile.ZipFile(zip_path) as zf:
        for z_path in zf.namelist():
            out_path = os.path.join(path_csv, z_path)
            if not os.path.exists(out_path):
                print(f"DEBUG: {z_path} not found - extracting")
                zf.extract(z_path, path_csv)
    
    if len(os.listdir(path_bg)) == 0:
        print("DEBUG: images not found locally - downloading")
        timeout = 20
        u_df = pd.read_csv(df_path, delimiter="\t")
        print(f"INFO: unsplash dataset contains {u_df.shape[0]} images")
        print(f"INFO: attepting to download subset of {num_photos}")
        for idx, row in u_df.iterrows():
            w = row["photo_width"]
            h = row["photo_height"]
            url = row["photo_image_url"]
            
            try:
                with urllib.request.urlopen(url, timeout=timeout) as url_response:
                    img_path = os.path.join(path_bg, f"photo_{idx}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(url_response.read())
                
            except urllib.error.URLError as e:
                print(f"DEBUG: {idx}/{num_photos} server timeout ({w},{h}) @ {url}")
            except Exception as e:
                print(f"WARNING: {idx}/{num_photos} unexpected error - {e}")
                
            print(f"DEBUG: {idx}/{num_photos} downloaded ({w},{h}) @ {url}")
            
            # stop once limit reached, save memory
            if idx == num_photos:
                break
    else:
        print("DEBUG: files found locally - skipping download")
        
if __name__ == "__main__":
    setup_unsplash()