import os
import shutil

# Root directory for GQA data
base_dir = r'd:\Deeplearning\gqa_data'
questions_dir = os.path.join(base_dir, 'questions')
scenegraphs_dir = os.path.join(base_dir, 'sceneGraphs')
train_all_dir = os.path.join(questions_dir, 'train_all_questions')

# Files/Folders to KEEP
keep_files = [
    'testdev_balanced_questions.json',
    'images.zip',
    'questions1.2.zip',
    'sceneGraphs.zip'
]

def cleanup_folder(folder_path, recursive=False):
    if not os.path.exists(folder_path):
        return
    
    print(f"Cleaning up: {folder_path}")
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Skip forbidden items
        if item in keep_files or item == 'subset' or item == 'images':
            continue
            
        try:
            if os.path.isfile(item_path):
                # Check if it's a zip file or a keep_file
                if item.endswith('.zip') or item in keep_files:
                    continue
                os.remove(item_path)
                print(f"Deleted file: {item}")
            elif os.path.isdir(item_path):
                # Delete subdirectories (like train_all_questions) if not special
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item}")
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")

if __name__ == "__main__":
    # Clean questions folder
    cleanup_folder(questions_dir)
    # Clean sceneGraphs folder
    cleanup_folder(scenegraphs_dir)
    
    print("\n--- DỌN DẸP HOÀN TẤT ---")
    print("Đã giữ lại: Toàn bộ ảnh, tệp ZIP, và các tệp trong thư mục 'subset'.")
