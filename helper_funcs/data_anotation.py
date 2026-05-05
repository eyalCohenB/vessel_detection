import soundfile as sf
import os, numpy as np
import sys
from viterbi import RunViterbi
# sys.path.append('..')
from custom_demon import buildDemonMat, plot_demon_matrix


# data_path = "/home/rbenjano/Boat_tagging_data/boat_detected_Eilat_dolphins/" # POSITIVE
data_path = "/home/rbenjano/Boat_tagging_data/No_Boat_Eilat_dolphins/" # NEGATIVE
# data_path = "../" # for local testing

anotations_path = "../anotations/"
boat_presence = 0 if "No_Boat" in data_path else 1

os.makedirs(anotations_path, exist_ok=True)

write_anotations = True
manual_anotations = False

if boat_presence:
    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".wav") and not filename.endswith(".flac"):
            continue
        
        if write_anotations:
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(anotations_path, txt_filename)
            txt_file = open(txt_path, "w")

        data, fs = sf.read(os.path.join(data_path, filename))
        sig = data[:, 0] if data.ndim > 1 else data

        demonMat, snrVec = buildDemonMat(
            sig=sig,
            fs=fs,
            BlockLen=1,
            fpass=100,
            fstop=1500,
            fpassDemon=300,
        )
        print(f"Processed {filename}")
        print("DemonMat shape:", demonMat.shape)
        print("SNRVec shape:", snrVec.shape)
        
        DetectFlag, tracker = RunViterbi(demonMat, Th=0)
        print(f"median tracker: {int(np.median(tracker) - 1)}")

        if write_anotations:
            if manual_anotations:
                plot_demon_matrix(demonMat)
                txt_file.write(f"{boat_presence} {input('Enter column: ')}")
            else:
                txt_file.write(f"{boat_presence} {int(np.median(tracker) - 1)}")
            txt_file.close()

else:
    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".wav") and not filename.endswith(".flac"):
            continue
        
        if write_anotations:
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(anotations_path, txt_filename)
            txt_file = open(txt_path, "w")

            txt_file.write(f"{boat_presence} -1")
            txt_file.close()


print(f"Done processing all files in {data_path}")

