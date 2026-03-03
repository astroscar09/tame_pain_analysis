import os
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from natsort import natsorted
import numpy as np
from tqdm import tqdm

model_name = "superb/wav2vec2-base-superb-er"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model.eval()

def grab_audio_files():

    main_folder = "data/mic1_trim_v2"

    # First, get the subfolders in a natural-sorted list
    # (in case your subfolders also have numeric components in their names)
    subfolders = [
        f for f in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, f))
    ]
    subfolders = natsorted(subfolders)

    all_wavs = []  # to collect (filepath, sr, data) or similar

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)

        # List .wav files in subfolder
        wav_files = [
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(".wav")
        ]
        # Sort them in natural order
        wav_files = natsorted(wav_files)

        # Process each .wav file
        for wav_file in wav_files:
            wav_path = os.path.join(subfolder_path, wav_file)

            # for example, using librosa (just as a placeholder)
            # import librosa
            # data, sr = librosa.load(wav_path, sr=None)

            # or using scipy
            data,sr = librosa.load(wav_path,sr=16000)

            # collect or do something with the data
            all_wavs.append((wav_path, sr, data))

    return all_wavs


def extract_embedding(all_wavs):
    
    embeddings = []
    
    for i in tqdm(range(len(all_wavs))):
        waveform = all_wavs[i][2]
        try:
            inputs = feature_extractor(waveform, return_tensors="pt", sampling_rate=16000, padding = True)

            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states[-1]
            mean_embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
            embeddings.append(mean_embedding)
        except Exception as e:
            print(f"Error processing File: {e}")

    return np.array(embeddings)


if __name__ == "__main__":
    all_wavs = grab_audio_files()
    embeddings = extract_embedding(all_wavs)
    # print(embeddings.shape)
    # np.save("Audio_Embeddings.npy", embeddings)
