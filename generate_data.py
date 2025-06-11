import numpy as np
import librosa
import torch
import laion_clap
import os
from natsort import natsorted
from scipy.io import wavfile
import pandas as pd
from tqdm import tqdm

audio_embedding_file = 'Audio_Embeddings.npy'
text_embedding_file = 'Text_Embeddings.npy'

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def load_clap_model():

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    model.load_ckpt('/Users/oac466/Downloads/music_speech_audioset_epoch_15_esc_89.98.pt') # download the default pretrained checkpoint.

    return model

def get_single_audio_embeddings(model):

    # Get audio embeddings from audio data
    audio_data, _ = librosa.load('data/mic1_trim_v2/p10085/p10085.LC.1.161.wav', sr=48000) # sample rate should be 48000
    audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)

    audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
    return audio_embed


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
            data,sr = librosa.load(wav_path,sr=48000)

            # collect or do something with the data
            all_wavs.append((wav_path, sr, data))

    return all_wavs

def get_meta_participant():
    meta_participant = pd.read_csv('data/meta_participant.csv')
    return meta_participant

def get_meta_audio():
    meta_audio = pd.read_csv('data/meta_audio.csv')
    return meta_audio

def grab_good_data_indices(meta_audio):

    # Filter indices based on the 'ACTION LABEL' column
    filtered_indices = meta_audio[meta_audio['ACTION LABEL'].isin([0, 1, 2])].index.values

    return filtered_indices

def get_audio_embeddings(model, all_wavs):

    if os.path.exists(audio_embedding_file):
        print(f"{audio_embedding_file} already exists. Loading embeddings...")
        embeddings = np.load(audio_embedding_file)
    else:
        print(f"{audio_embedding_file} not found. Generating audio embeddings...")
        embeddings = []
        
        for i in tqdm(range(len(all_wavs))):        # Now, you're calling the tqdm function
            audio_data = all_wavs[i][2]             #this 2 index gets the data since we get (path, sr, data))
            audio_data = audio_data.reshape(1, -1)
            audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            embeddings.append(audio_embed)
        
        embeddings = np.array(embeddings).squeeze() #making all embeddings into a single array

    return embeddings

def get_text_embeddings(model, meta_audio):

    if os.path.exists(text_embedding_file):
        print(f"{text_embedding_file} already exists. Loading embeddings...")
        embeddings = np.load(text_embedding_file)
    
    else:
        text_embeddings = []
        
        for i in tqdm(range(len(meta_audio))): # Now, you're calling the tqdm function
            text_data = meta_audio['NOTES'][i]
            if not isinstance(text_data, str):
                text_data = ' '
            text_embed = model.get_text_embedding(text_data)
            text_embeddings.append(text_embed)

        embeddings = np.array(text_embeddings).squeeze()

        return embeddings

def get_embeddings(audio_embeddings, filtered_indices, text_embeddings = None):

    # Filter embeddings based on the indices
    if text_embeddings is not None:
        filtered_audio_embeddings = audio_embeddings[filtered_indices]
        filtered_text_embeddings = text_embeddings[filtered_indices]
        embeddings = np.concatenate((filtered_audio_embeddings, filtered_text_embeddings), axis=1)
    else:
        embeddings = audio_embeddings[filtered_indices]
    
    df = pd.DataFrame(embeddings)
    return df


def get_tabular_data(meta_audio, meta_participant, filtered_indices):

    meta_participant = meta_participant.loc[meta_participant.index.repeat(meta_participant["NUMBER OF FILES"])]
    meta_participant.reset_index(drop=True, inplace=True)
    merged_df = pd.merge(meta_audio, meta_participant, left_index=True, right_index=True)

    labels = merged_df.loc[:,'REVISED PAIN']
    labels= np.array([0 if x < 4 else 1 for x in labels]) # Labels: 0 or 1 for "No Pain"/"Pain"
    merged_df['Pain'] = labels

    tabular_data = merged_df.loc[:, ['GENDER', 'AGE', 'RACE/ETHNICITY', 'TOTAL DURATION (SEC)', 'Pain']]
    tabular_data = pd.get_dummies(tabular_data, columns=['GENDER', 'RACE/ETHNICITY'], dtype=np.float32)

    return tabular_data.iloc[filtered_indices, :].reset_index(drop=True)




def merge_data(df, tabular_data):

    merged_features = pd.concat([df, tabular_data], axis=1) # Concatenate along columns (axis=1)

    return merged_features


def main(text=True):
    
    if os.path.exists(audio_embedding_file):
        pass
    else:
        print(f"{text_embedding_file} already exists. Loading embeddings...")
        print('Loading CLAP model...')
        model = load_clap_model()
        print('Loading Audio files...')
        audio_wavs =  grab_audio_files()
        print(f'Found {len(audio_wavs)} audio files.')

        print('Extracting audio embeddings...')
        audio_embeddings = get_audio_embeddings(model, audio_wavs)

    print('Loading metadata...')
    meta_participant = get_meta_participant()
    meta_audio = get_meta_audio()
    filtered_indices = grab_good_data_indices(meta_audio)
    print(f'Filtered {len(filtered_indices)} audio files based on action labels.')

    if text:

        print('Extracting text embeddings...')
        text_embeddings = get_text_embeddings(model, meta_audio)
        print('Text embeddings shape:', text_embeddings.shape)
    else:
        text_embeddings = None
        print('Skipping text embeddings extraction.')

    if text:
        embeddings = get_embeddings(audio_embeddings, filtered_indices, text_embeddings)
    else:
        embeddings = get_embeddings(audio_embeddings, filtered_indices)
    
    print('Embeddings shape:', embeddings.shape)

    print('Merging metadata with audio embeddings...')
    merged_df = merge_data(meta_audio, meta_participant, filtered_indices)
    print('Merged DataFrame shape:', merged_df.shape)
    print('Data generation complete.')
    print('Embeddings and merged data ready for saving.')

    return merged_df, embeddings
    

if __name__ == "__main__":

    AUDIO = True
    TEXT = False

    merged_df, embeddings = main(text_embeddings=TEXT)

    # Save the embeddings and merged DataFrame
    #np.save('Audio_Embeddings.npy', embeddings)
    #merged_df.to_csv('merged_data.csv', index=False)

    #print("Data generation complete. Embeddings and merged data saved.")