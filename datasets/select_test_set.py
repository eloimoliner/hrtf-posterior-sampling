"""Builds the 600-sample test set of the paper (6 test subjects x 100 DoAs, each with a random VCTK utterance)."""
import os
from os.path import join, exists, basename
from glob import glob
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--path_v3p1", required=True, help="sofamyroom_v3p1/hutubs directory containing valid/ and test/")
parser.add_argument("--path_speech", required=True, help="VCTK directory containing p226/ and p287/")
parser.add_argument("--path_new_test", required=True)
args = parser.parse_args()

N_per_subject = 100

path_valid = join(args.path_v3p1, "valid")
path_test = join(args.path_v3p1, "test")
path_speech = args.path_speech
path_new_test = args.path_new_test

subjects = ["pp1", "pp27", "pp30", "pp53", "pp65", "pp67"]
folders = ["anthropometry", "brir", "doa", "hrir"]
file_types = ["np", "wav", "flac"]
speech_files = glob(join(path_speech, "p226", "*.flac")) + glob(join(path_speech, "p287", "*.flac"))
assert len(speech_files) > 0, f"no speech files found in {path_speech}"

if exists(join(path_new_test)):
    decision = input("Are you sure you want to delete the folder?")
    if decision.lower() in ["yes", "y"]:
        shutil.rmtree(join(path_new_test))
        os.makedirs(join(path_new_test))
    else:
        sys.exit(1)

for folder in tqdm(folders):

    if not exists(join(path_new_test, folder)):
        os.makedirs(join(path_new_test, folder))
    files = []

    for s in subjects:

        s_files = []
        for file_type in file_types:
            s_files += sorted(glob(join(path_valid, folder, f"{s}_*.{file_type}")))
            s_files += sorted(glob(join(path_test, folder, f"{s}_*.{file_type}")))
        files += s_files[: N_per_subject]

    for file in tqdm(files):
        shutil.copy(file, join(path_new_test, folder, basename(file)))


if not exists(join(path_new_test, "speech_48k")):
    os.makedirs(join(path_new_test, "speech_48k"))

data = {"hrir": [], "speech": []}

for file in tqdm(glob(join(path_new_test, "brir", "*.wav"))):
    speech_file = np.random.choice(speech_files)

    shutil.copy(speech_file, join(path_new_test, "speech_48k", basename(file).replace("wav", "flac")))

    data["hrir"].append(os.path.basename(file))
    data["speech"].append(os.path.basename(speech_file))

# Save results as DataFrame
df = pd.DataFrame(data)
df.to_csv(join(path_new_test, "speech_files.csv"), index=False)
