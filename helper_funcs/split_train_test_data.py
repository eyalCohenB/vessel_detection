import os
import numpy as np
"this script is for splitting the data into train and test sets, and copying selected files to the respective directory"

def main():
    source_dir_audio = "C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/model_training_demon_mat/data/talmon_data/mixed"
    source_dir_anotations = "C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/model_training_demon_mat/data/talmon_data/mixed"
    train_dir = "C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/model_training_demon_mat/data/talmon_data/train_data"
    test_dir = "C:/Users/ecohen/Documents/eyal_work/vessle_detection_new/model_training_demon_mat/data/talmon_data/test_data"

    files = [f for f in os.listdir(source_dir_audio) if f.endswith(".wav") or f.endswith(".flac")]
    np.random.shuffle(files)
    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for f in train_files:
        os.system(f"move {os.path.join(source_dir_audio, f)} {os.path.join(train_dir, f)}")
        txt_file = f.split(".")[0] + ".txt"
        os.system(f"move {os.path.join(source_dir_anotations, txt_file)} {os.path.join(train_dir, txt_file)}")

    for f in test_files:
        os.system(f"move {os.path.join(source_dir_audio, f)} {os.path.join(test_dir, f)}")
        txt_file = f.split(".")[0] + ".txt"
        os.system(f"move {os.path.join(source_dir_anotations, txt_file)} {os.path.join(test_dir, txt_file)}")

    print(f"Copied {len(train_files)} files to {train_dir} and {len(test_files)} files to {test_dir}")

    pass

def complete_splitting():
    source_dir_audio = "../data/"
    source_dir_anotations = "../anotations/"
    train_dir = "../train_data/"
    test_dir = "../test_data/"

    files = os.listdir(source_dir_audio)
    train_files = os.listdir(train_dir)

    missing_test = [f for f in files if f.endswith(".wav") or f.endswith(".flac") and f not in train_files]
    print(f"Found {len(missing_test)} missing files for test set.")
    
    for f in missing_test:
        os.system(f"move {os.path.join(source_dir_audio, f)} {os.path.join(test_dir, f)}")
        txt_file = f.split(".")[0] + ".txt"
        os.system(f"move {os.path.join(source_dir_anotations, txt_file)} {os.path.join(test_dir, txt_file)}")

    print(f"Copied {len(missing_test)} files to {test_dir}")


if __name__ == "__main__":
    main()
    # complete_splitting()