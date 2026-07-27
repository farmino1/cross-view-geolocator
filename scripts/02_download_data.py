import subprocess, os, zipfile
from huggingface_hub import hf_hub_download

CITIES = ["seattle", "london", "tokyo", "sydney"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

for city in CITIES:
    city_dir = f"{DATA_DIR}/{city}"
    sat_dir = f"{city_dir}/satellite"
    street_dir = f"{city_dir}/streetview"
    
    if os.path.exists(sat_dir) and os.path.exists(street_dir) and len(os.listdir(sat_dir)) > 0:
        print(f"{city}: already downloaded, skipping.")
        continue
    
    # Clean partial extract
    if os.path.exists(city_dir):
        subprocess.run(f"rm -rf {city_dir}", shell=True)
    
    print(f"Downloading {city}...")
    zip_path = hf_hub_download(
        repo_id="gaoshuang98/CV-Cities",
        filename=f"{city}.zip",
        repo_type="dataset",
    )
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print(f"  Done.")

print("\nDataset ready.")
for city in CITIES:
    sat_dir = f"{DATA_DIR}/{city}/satellite"
    street_dir = f"{DATA_DIR}/{city}/streetview"
    if os.path.exists(sat_dir):
        sat = len(os.listdir(sat_dir))
        street = len(os.listdir(street_dir))
        print(f"  {city}: {sat} satellite, {street} streetview")
    else:
        print(f"  {city}: NOT DOWNLOADED")
