import numpy as np
import librosa
import torch
import laion_clap
import os
from natsort import natsorted
from scipy.io import wavfile
import pandas as pd
from tqdm import tqdm # Import the tqdm function from the tqdm module
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

meta_participant = pd.read_csv('data/meta_participant.csv')
meta_audio = pd.read_csv('data/meta_audio.csv')
filtered_indices = meta_audio[meta_audio['ACTION LABEL'].isin([0, 1, 2])].index
# filtered_wavs = [all_wavs[i] for i in filtered_indices]

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
model.load_ckpt('/Users/oac466/Downloads/music_speech_audioset_epoch_15_esc_89.98.pt') # download the default pretrained checkpoint.


# Path to main folder containing all subfolders
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


# Get audio embeddings from audio data
audio_data, _ = librosa.load('data/mic1_trim_v2/p10085/p10085.LC.1.161.wav', sr=48000) # sample rate should be 48000
audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
print(audio_embed[:,-20:])
print(audio_embed.shape)


embeddings = []
for i in tqdm(range(len(all_wavs))): # Now, you're calling the tqdm function
  audio_data = all_wavs[i][2]
  audio_data = audio_data.reshape(1, -1)
  audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
  embeddings.append(audio_embed)

print(embeddings[0][:,-20:])
embeddings[0].shape

text_embeddings_array = np.array(text_embeddings).squeeze()
text_embeddings_array.shape


# prompt: load Audio Embeddings.npy

import numpy as np

# Load the saved embeddings
audio_embeddings = np.load('Audio Embeddings.npy')

# Now you can work with the loaded embeddings
print(audio_embeddings.shape)

embeddings_combined = np.concatenate((audio_embeddings, text_embeddings_array), axis=1)
embeddings_combined.shape

participant_data= meta_participant.loc[meta_participant.index.repeat(meta_participant["NUMBER OF FILES"])]

# 3. Reset the index (optional, but usually helpful)
participant_data.reset_index(drop=True, inplace=True)
# Merge meta_audio and participant_data DataFrames
merged_df = pd.merge(meta_audio, participant_data, left_index=True, right_index=True)
# merged_df = merged_df[merged_df['ACTION LABEL'].isin([0, 1, 2])]

tabular_data = merged_df.loc[:, ['GENDER', 'AGE', 'RACE/ETHNICITY', 'TOTAL DURATION (SEC)']]
tabular_data = pd.get_dummies(tabular_data, columns=['GENDER', 'RACE/ETHNICITY'], dtype=np.float32)
num_tab_features = tabular_data.shape[1]

labels = merged_df.loc[:,'REVISED PAIN']
labels= np.array([0 if x < 4 else 1 for x in labels]) # Labels: 0 or 1 for "No Pain"/"Pain"

df = pd.DataFrame(embeddings_combined)
tabular_data_reset = tabular_data.reset_index(drop=False)  # Reset index of tabular_data
merged_features = pd.concat([df, tabular_data_reset], axis=1) # Concatenate along columns (axis=1)
merged_features.index = merged_df.index
merged_features = merged_features.drop(columns=['index'])
merged_features.head()

unique_pids = np.array(merged_df['PID_x'].unique())
random_pid = np.random.choice(unique_pids, size=10, replace=False)
# X_train, y_train = merged_features[~merged_df['PID_x'].isin(random_pid)], labels[~merged_df['PID_x'].isin(random_pid)]
# X_test, y_test = merged_features[merged_df['PID_x'].isin(random_pid)], labels[merged_df['PID_x'].isin(random_pid)]

X_train, X_test, y_train, y_test = train_test_split(merged_features, labels, test_size=0.3, random_state=2)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Define the neural network architecture
class ComplexClassifier(nn.Module):
    def __init__(self, input_size):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Dropout for regularization
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 2)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Get the number of input features
model = ComplexClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


# Training loop
num_epochs = 2500 # Adjust as needed
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
       print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

y_pred_tensor = model(X_test_tensor)
y_pred = torch.argmax(y_pred_tensor, dim=1)

for pid in random_pid:
    index = merged_df[merged_df['PID_x'] == pid].index
    X_test_pid = merged_features.iloc[index,:]
    X_test_scaled_pid = scaler.transform(X_test_pid)
    X_test_tensor_pid = torch.tensor(X_test_scaled_pid, dtype=torch.float32)
    y_pred_tensor_pid = model(X_test_tensor_pid)
    y_pred_pid = torch.argmax(y_pred_tensor_pid, dim=1).numpy()
    y_test_pid = labels[index]
    acc = accuracy_score(y_test_pid, y_pred_pid)
    auc = roc_auc_score(y_test_pid, y_pred_pid)
    # f1 = f1_score(y_test_pid, y_pred_pid,average='micro')

    # Print the results
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUC: {auc * 100:.2f}%")
    # print(f"Test F1 Score: {f1 * 100:.2f}%")

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# all_preds = xgb_classifier.predict(X_test)
all_preds = y_pred.numpy()
# Calculate accuracy
acc = accuracy_score(y_test, all_preds)
auc = roc_auc_score(y_test, all_preds)
f1_score = f1_score(y_test, all_preds,average='micro')

# Print the results
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test AUC: {auc * 100:.2f}%")
print(f"Test F1 Score: {f1_score * 100:.2f}%")

# Compute confusion matrix
cm = confusion_matrix(y_test, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()