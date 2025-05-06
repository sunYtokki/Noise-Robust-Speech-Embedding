# fork from https://github.com/msplabresearch/MSP-Podcast_Challenge

# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy
import csv
from time import perf_counter
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from transformers import AutoModel, WavLMModel

# Self-Written Modules
sys.path.append("/proj/speech/users/syk2145/baseline/MSP-Podcast_Challenge")
import net
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--model_path", type=str, default="./model/wavlm-large")
parser.add_argument("--pooling_type", type=str, default="MeanPooling")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument('--audio_dir', type=str, help="Path to directory containing audio files")
parser.add_argument('--audio_list', type=str, help="Path to file with list of audio files and ground truth")
parser.add_argument('--store_path')
args = parser.parse_args()

SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

import json
from collections import defaultdict
config_path = "config_cat.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"] if not args.audio_dir else args.audio_dir
label_path = config["label_path"]

def load_audio_list(list_path, audio_dir):
    """Load list of audio files to process from a text file."""
    print(f"Loading audio list from {list_path}")
    audio_files = []
    labels = []
    
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line: Audios/MSP-PODCAST_0408_0129.wav; H; A:5.4; V:4.0; D:5.6;
            parts = line.split(';')
            if len(parts) < 2:
                print(f"Skipping invalid line: {line}")
                continue
                
            # Get file path
            file_path = parts[0].strip()
            # Make path absolute if it's relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(audio_dir, file_path)
            
            # Extract label information
            label_info = {}
            
            # Extract emotion class label
            if len(parts) > 1:
                emotion = parts[1].strip()
                # Map emotion labels to indices
                label_map = {
                    'A': 0,  # Anger
                    'S': 1,  # Sadness
                    'H': 2,  # Happiness
                    'U': 3,  # Surprise
                    'F': 4,  # Fear
                    'D': 5,  # Disgust
                    'C': 6,  # Contempt
                    'N': 7,  # Neutral
                }
                label_info['emotion'] = label_map.get(emotion, -1)
            
            # Extract dimensional values if present
            for i in range(2, len(parts)):
                part = parts[i].strip()
                if not part:
                    continue
                    
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        value = float(value)
                        label_info[key] = value
                    except ValueError:
                        label_info[key] = value
            
            audio_files.append(file_path)
            labels.append(label_info)
    
    print(f"Loaded {len(audio_files)} files from list")
    return audio_files, labels


# Load the CSV file
df = pd.read_csv(label_path)

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()

# Total number of samples
total_samples = len(train_df)

# Calculate class weights
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}

print(class_weights)

# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]

# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)

# Print or return the tensor
print(class_weights_tensor)

# Load SSL model and other models
print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained("microsoft/wavlm-large")

ssl_model.freeze_feature_encoder()
ssl_checkpoint = torch.load(MODEL_PATH+"/final_ssl.pt")
if "model_state_dict" in ssl_checkpoint:
    # First, extract the model weights from the checkpoint
    wavlm_weights = {k.replace("online_encoder.model.", ""): v 
                for k, v in ssl_checkpoint["model_state_dict"].items() 
                if k.startswith("online_encoder.model.")}

    # Convert the unexpected keys to the expected format
    new_weights = {}
    for key, value in wavlm_weights.items():
        if "parametrizations.weight.original0" in key:
            new_key = key.replace("parametrizations.weight.original0", "weight_g")
            new_weights[new_key] = value
        elif "parametrizations.weight.original1" in key:
            new_key = key.replace("parametrizations.weight.original1", "weight_v")
            new_weights[new_key] = value
        else:
            new_weights[key] = value

    # Load the modified weights
    ssl_model.load_state_dict(new_weights, strict=False)    
else:
    ssl_model.load_state_dict(ssl_checkpoint)
ssl_model.eval(); ssl_model.cuda()
########## Implement pooling method ##########
feat_dim = ssl_model.config.hidden_size

pool_net = getattr(net, args.pooling_type)
attention_pool_type_list = ["AttentiveStatisticsPooling"]
if args.pooling_type in attention_pool_type_list:
    is_attentive_pooling = True
    pool_model = pool_net(feat_dim)
    pool_model.load_state_dict(torch.load(MODEL_PATH+"/final_pool.pt"))
else:
    is_attentive_pooling = False
    pool_model = pool_net()
print(pool_model)

pool_model.eval()
pool_model.cuda()
concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if args.pooling_type in concat_pool_type_list \
    else feat_dim

ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, 8, dropout=0.5)
##############################################
ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()

lm = utils.LogManager()
lm.alloc_stat_type_list(["inference_loss"])

min_epoch=0
min_loss=1e10

lm.init_stat()

# Create results directory if it doesn't exist
if not os.path.exists(MODEL_PATH + '/results'):
    os.mkdir(MODEL_PATH + '/results')

# Function to process a single audio file
def process_audio_file(file_path, ssl_model, pool_model, ser_model, label_idx=None):
    try:
        # Load audio file
        wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        waveform, sr = librosa.load(file_path, sr=16000)
        
        # Normalize waveform
        waveform = (waveform - wav_mean) / (wav_std + 1e-10)
        
        # Convert to tensor
        x = torch.tensor(waveform).unsqueeze(0).float().cuda()
        mask = torch.ones(1, len(waveform)).cuda()
        
        # Process with model
        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state
            ssl = pool_model(ssl, mask)
            emo_pred = ser_model(ssl)
            
        # Process prediction and label
        if label_idx is not None:
            y = torch.tensor([label_idx]).long().cuda()
            return emo_pred, y, os.path.basename(file_path)
        else:
            return emo_pred, None, os.path.basename(file_path)
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, os.path.basename(file_path)

# Main inference function
def run_inference():
    INFERENCE_TIME = 0
    FRAME_SEC = 0
    
    total_pred = []
    total_y = []
    total_utt = []
    
    # Decide how to process audio files
    if args.audio_list:
        # Process files from list
        audio_files, labels = load_audio_list(args.audio_list, args.audio_dir)
        
        for i, file_path in enumerate(tqdm(audio_files, desc="Processing files")):
            label_idx = labels[i].get('emotion', None) if i < len(labels) else None
            stime = perf_counter()
            emo_pred, y, fname = process_audio_file(file_path, ssl_model, pool_model, ser_model, label_idx)
            etime = perf_counter()
            INFERENCE_TIME += (etime - stime)
            
            if emo_pred is not None:
                total_pred.append(emo_pred)
                if y is not None:
                    total_y.append(y)
                total_utt.append(fname)
    else:
        # Process from dataset
        total_dataset = dict()
        total_dataloader = dict()
        for dtype in ["dev"]:
            cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
            cur_wavs = utils.load_audio(audio_path, cur_utts)
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
            cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
            cur_emo_set = utils.CAT_EmoSet(cur_labs)

            total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
            total_dataloader[dtype] = DataLoader(
                total_dataset[dtype], batch_size=1, shuffle=False, 
                pin_memory=True, num_workers=4,
                collate_fn=utils.collate_fn_wav_lab_mask
            )
            
        for dtype in ["dev"]:
            for xy_pair in tqdm(total_dataloader[dtype]):
                x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
                y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.cuda(non_blocking=True).long()
                mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
                fname = xy_pair[3]
                
                FRAME_SEC += (mask.sum()/16000)
                stime = perf_counter()
                with torch.no_grad():
                    ssl = ssl_model(x, attention_mask=mask).last_hidden_state
                    ssl = pool_model(ssl, mask)
                    emo_pred = ser_model(ssl)

                    total_pred.append(emo_pred)
                    total_y.append(y)
                    total_utt.append(fname)

                etime = perf_counter()
                INFERENCE_TIME += (etime-stime)
    
    # Save results to CSV
    def label_to_one_hot(label, num_classes=8):
        if label is None or (hasattr(label, 'item') and label.item() == -1):
            return ','.join(['0.0'] * num_classes)
        one_hot = ['0.0'] * num_classes
        one_hot[label.item()] = '1.0'
        return ','.join(one_hot)

    data = []
    for i, (pred, utt) in enumerate(zip(total_pred, total_utt)):
        y = total_y[i] if i < len(total_y) else None
        one_hot_label = label_to_one_hot(y)
        pred_values = ', '.join([f'{val:.4f}' for val in pred.cpu().numpy().flatten()])
        data.append([utt, one_hot_label, pred_values])

    # Writing to CSV file
    csv_filename = MODEL_PATH + '/results/inference_results.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Label', 'Prediction'])
        writer.writerows(data)

    # Calculate metrics if ground truth is available
    if total_y:
        # Load the CSV file
        df = pd.read_csv(csv_filename)

        # Function to convert string representation of one-hot vectors to numpy arrays
        def string_to_array(s):
            return np.array([float(i) for i in s.strip('\"').split(',')])

        # Convert the string representations to numpy arrays
        df['Label'] = df['Label'].apply(string_to_array)
        df['Prediction'] = df['Prediction'].apply(string_to_array)

        # Use argmax to determine the class with the highest probability
        y_true = np.argmax(np.stack(df['Label'].values), axis=1)
        y_pred = np.argmax(np.stack(df['Prediction'].values), axis=1)

        # Compute metrics
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        # Print results
        print(f"F1-Micro: {f1_micro}")
        print(f"F1-Macro: {f1_macro}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # Save the results in a text file
        with open(MODEL_PATH + '/results/metrics.txt', 'w') as f:
            f.write(f"F1-Micro: {f1_micro}\n")
            f.write(f"F1-Macro: {f1_macro}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")

        # CCC calculation for weighted loss
        total_pred_tensor = torch.cat(total_pred, 0)
        total_y_tensor = torch.cat(total_y, 0)
        loss = utils.CE_weight_category(total_pred_tensor, total_y_tensor, class_weights_tensor)
        
        # Logging
        lm.add_torch_stat("inference_loss", loss)
        lm.print_stat()

    print("Duration of whole inference set", FRAME_SEC, "sec")
    print("Inference time", INFERENCE_TIME, "sec")
    print("Inference time per sec", INFERENCE_TIME/FRAME_SEC if FRAME_SEC > 0 else "N/A", "sec")

    # Save metrics to specified path if provided
    if args.store_path:
        os.makedirs(os.path.dirname(args.store_path), exist_ok=True)
        with open(args.store_path, 'w') as f:
            loss = str(lm.get_stat("inference_loss")) if total_y else "N/A"
            f.write(loss+"\n")

# Run the inference
run_inference()