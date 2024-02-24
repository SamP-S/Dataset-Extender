import requests
import zipfile
import os
import pandas as pd
import urllib
import numpy as np
import cv2

def setup_unsplash():
    
    # img data paths
    data_path = "data"
    imgs_path = os.path.join(data_path, "unsplash")
    
    # unsplash db paths
    extract_path = "unsplash"
    zip_filename = "unsplash_lite.zip"
    zip_path = os.path.join(extract_path, zip_filename)
    df_path = os.path.join(extract_path, "photos.tsv000")

    if not os.path.exists(data_path):
        print(f"DEBUG: {data_path} directory not found - creating")
        os.mkdir(data_path)
        
    if not os.path.exists(imgs_path):
        print(f"DEBUG: {imgs_path} directory not found - creating")
        os.mkdir(imgs_path)
        
    if not os.path.exists(extract_path):
        print(f"DEBUG: {extract_path} directory not found - creating")
        os.mkdir(extract_path)

    if not os.path.exists(zip_path):
        print(f"DEBUG: {zip_filename} not found - downloading")
        url = "https://unsplash.com/data/lite/latest"
        r = requests.get(url)
        open(zip_path , 'wb').write(r.content)

    with zipfile.ZipFile(zip_path) as zf:
        for z_path in zf.namelist():
            out_path = os.path.join(extract_path, z_path)
            if not os.path.exists(out_path):
                print(f"DEBUG: {z_path} not found - extracting")
                zf.extract(z_path, extract_path)
    
    if len(os.listdir(imgs_path)) == 0:
        print("DEBUG: images not found locally - downloading")
        timeout = 5
        num_photos = 1000
        u_df = pd.read_csv(df_path, delimiter="\t")
        print(f"INFO: unsplash dataset contains {u_df.shape[0]} images")
        print(f"INFO: attepting to download subset of {num_photos}")
        for idx, row in u_df.iterrows():
            w = row["photo_width"]
            h = row["photo_height"]
            url = row["photo_image_url"]
            
            try:
                with urllib.request.urlopen(url, timeout=timeout) as url_response:
                    img_path = os.path.join(imgs_path, f"photo_{idx}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(url_response.read())
                
            except urllib.error.URLError as e:
                print(f"DEBUG: server timeout ({w},{h}) @ {url}")
            except Exception as e:
                print(f"WARNING: unexpected error - {e}")
                
            print(f"DEBUG: downloaded ({w},{h}) @ {url}")
            
            # safe quit for testing
            if idx == 100:
                break
    else:
        print("DEBUG: images found locally - skipping download")
        
if __name__ == "__main__":
    setup_unsplash()