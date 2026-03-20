import json
import random
import os

# Paths
train_path = r'd:\Deeplearning\gqa_data\questions\train_balanced_questions.json'
val_path = r'd:\Deeplearning\gqa_data\questions\val_balanced_questions.json'
output_dir = r'd:\Deeplearning\gqa_data\subset'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def filter_json(input_file, output_filename, max_images):
    print(f"Filtering {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all unique image IDs
    img_ids = list(set(v['imageId'] for v in data.values()))
    
    # Select max_images image IDs
    selected_img_ids = set(random.sample(img_ids, min(max_images, len(img_ids))))
    
    # Filter questions
    filtered_data = {k: v for k, v in data.items() if v['imageId'] in selected_img_ids}
    
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
        
    return len(selected_img_ids), len(filtered_data)

if __name__ == "__main__":
    random.seed(42)
    train_imgs, train_qs = filter_json(train_path, 'train_subset_25k.json', 25000)
    val_imgs, val_qs = filter_json(val_path, 'val_subset_5k.json', 5000)
    
    print(f"\nTraining subset: {train_imgs} images, {train_qs} questions.")
    print(f"Validation subset: {val_imgs} images, {val_qs} questions.")
