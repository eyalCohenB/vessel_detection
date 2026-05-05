"""
this file is for testing the wav files in the data directory,
to make sure they are all readable and not corrupted.
It will print out the names of any files that are not readable.
"""
import os
import soundfile as sf

def main():
    data_dir1 = "/data/users/ecohen/vessle_detection_new/model_training_demon_mat/train_data/" 
    data_dir2 = "/data/users/ecohen/vessle_detection_new/model_training_demon_mat/test_data/" 

    for data_dir in [data_dir1, data_dir2]:
        files = os.listdir(data_dir)
        audio_files = [f for f in files if f.endswith(".wav") or f.endswith(".flac")]
        print(f"Checking {len(audio_files)} audio files in {data_dir}...")
        unreadable_files = []
        for f in audio_files:
            file_path = os.path.join(data_dir, f)
            try:
                data, fs = sf.read(file_path)
            except Exception as e:
                print(f"Unreadable file: {f}, Directory: {data_dir}, error: {e}")
                unreadable_files.append(f)

        if not unreadable_files:
            print("All audio files are readable.")
        else:
            print(f"Found {len(unreadable_files)} unreadable files in {data_dir}.")

if __name__ == "__main__":
    main()

