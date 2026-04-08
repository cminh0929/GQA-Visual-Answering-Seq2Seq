import kagglehub
import shutil
import os

# Download images
print("Starting download of gqa-images-subset...")
cache_path = kagglehub.dataset_download("minhngcng3/gqa-images-subset")
print(f"Download complete! Saved at: {cache_path}")

# Destination folder
dest_dir = r"C:\Users\cminh\Downloads\gqa-images-subset"
os.makedirs(dest_dir, exist_ok=True)

# Move entire contents of cache_path to dest_dir
print(f"Moving files to {dest_dir}...")
for filename in os.listdir(cache_path):
    src = os.path.join(cache_path, filename)
    dst = os.path.join(dest_dir, filename)
    if os.path.isdir(src):
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
        shutil.rmtree(src)
    else:
        shutil.move(src, dst)

print(f"Dataset successfully moved to: {dest_dir}")
