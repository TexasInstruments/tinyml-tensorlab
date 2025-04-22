# torch imports
import torch
import torch.backends.cudnn as cudnn
from torch.ao.quantization import quantize_fx
import torch.utils
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, Subset
import os
import shutil
import re
import random
import torchaudio
from scipy.io import wavfile
from pydub import AudioSegment
from typing import Tuple, List

from tinyml_torchmodelopt.quantization import \
    TINPUTinyMLQATFxModule, TINPUTinyMLPTQFxModule, GenericTinyMLQATFxModule, GenericTinyMLPTQFxModule
import onnx
import onnxruntime as ort
# other imports
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


from torchmetrics.classification import Accuracy
from tqdm import tqdm
import tensorflow as tf
import torch.optim as optim
import hashlib
import requests
import tarfile

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE='cpu'
def process_audio_file(file_path, label, output_dir, SAMPLE_RATE=16000):
    """
    Process a .wav audio file by splitting it into 1-second segments and saving them as .wav files.
    
    Args:
    - file_path: Path to the input .wav audio file
    - example_id: Base identifier for the audio segments
    - label: Label to associate with the audio segments
    - output_dir: Directory to save the generated .wav files
    - SAMPLE_RATE: Sampling rate of the audio (default 16000 Hz)
    
    Yields:
    - Tuple of (segment_id, segment_metadata)
    """
    relpath, wavname = os.path.split(file_path)
    _, word = os.path.split(relpath)
    example_id = '{}_{}'.format(word, wavname)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Convert to numpy array of samples
    audio_samples = np.array(audio.get_array_of_samples())
    cc = 0
    # Split audio into 1-second segments with 50% overlap
    for start in range(0, len(audio_samples) - SAMPLE_RATE, SAMPLE_RATE // 2):
        # Extract audio segment
        audio_segment = audio_samples[start : start + SAMPLE_RATE]
        
        # Create unique identifier for the segment
        cur_id = f'{example_id}_{start}'
        
        # Prepare segment metadata
        example = {
            'audio': audio_segment, 
            'label': label
        }
        
        # Generate full path for the output .wav file
        output_path = os.path.join(output_dir, f'{cur_id}.wav')
        
        # Save the audio segment as a .wav file
        wavfile.write(output_path, SAMPLE_RATE, audio_segment.astype(np.int16))
        
        # Yield the segment ID and metadata
        cc += 1
    return cc

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    import hashlib
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                        (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


# Fetch Background noise data
def prepare_background_data(bg_path, BACKGROUND_NOISE_DIR_NAME):
    """Searches a folder for background noise audio, and loads it into memory.
    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.
    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.
    Returns:
    List of raw PCM-encoded audio samples of background noise.
    Raises:
    Exception: If files aren't found in the folder.
    """
    background_data = []
    background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
        return background_data
    for wav in os.listdir(background_dir):
        wav_path = os.path.join(background_dir, wav)
        if not wav_path.endswith((".wav", ".WAV")):
            continue
        audio, _ = torchaudio.load(wav_path)
        background_data.append(audio.squeeze())
    if not background_data:
        raise Exception('No background wav files were found in ' + background_dir)
    return background_data

# Audio Preprocessor
class AudioPreprocessor(object):
    def __init__(self, flag_trainer: bool, flag_background_noise: bool=False, background_data: list = [], audio_features: str='tfMFCC', sample_rate: int = 16000, audio_duration: int = 1000, n_mfcc: int=10):
        self.flag_trainer = flag_trainer
        self.flag_background_noise = flag_background_noise
        if audio_features in ["torchMFCC", "tfMFCC", "td_samples"]:
            self.audio_features = audio_features
        else:
            raise NotImplementedError(f"Audio Features of Type : {audio_features} are Not Supported")
        self.background_data = background_data
        self.sr = sample_rate
        self.audio_duration = audio_duration
        self.n_mfcc = n_mfcc
        self.n_audio = int(self.sr * self.audio_duration / 1000)
    
    def __call__(self, audio_tensor):
        audio_tensor = torch.squeeze(audio_tensor)
        audio_tensor = audio_tensor.to(torch.float32)
        # Normalize
        if self.audio_features in ["tfMFCC", "torchMFCC"]:
            audio_tensor = audio_tensor / torch.max(audio_tensor).item()
        else:
            audio_tensor = audio_tensor / 2 ** 15
        # Pad
        audio_tensor = F.pad(audio_tensor, (0, self.n_audio - audio_tensor.shape[-1]))
        
        
        # bg Noise augmentation
        if self.flag_trainer and self.flag_background_noise:
            audio_tensor = self.add_bgNoise(audio_tensor)
        
        # Feature Extraction
        if self.audio_features == "torchMFCC":
            audio_tensor = self.compute_torchMFCC(audio_tensor)
        elif self.audio_features == "tfMFCC":
            audio_tensor = self.compute_tfMFCC(audio_tensor)
        else:
            audio_tensor = audio_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        
        return audio_tensor
    
    def compute_tfMFCC(self, audio_tensor):
        # Convert to TF Tensor
        spectrogram_length = 1 + int ((self.n_audio - int(self.sr * 30 / 1000)) / int(self.sr * 20 / 1000))
        
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        audio_tensor_tf = tf.convert_to_tensor(audio_tensor.numpy(), dtype=tf.float32)
        stfts = tf.signal.stft(audio_tensor_tf, frame_length=int(self.sr * 30 / 1000), 
                    frame_step=int(self.sr * 20 / 1000), fft_length=None,
                    window_fn=tf.signal.hann_window
                    )
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        # default values used by contrib_audio.mfcc as shown here
        # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins,
                                                                            self.sr,
                                                                            lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs_tf = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self.n_mfcc]
        mfccs_tf = tf.reshape(mfccs_tf, [1, spectrogram_length, self.n_mfcc])
        mfccs = torch.tensor(mfccs_tf.numpy(), dtype=torch.float32)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        return mfccs
    
    def compute_torchMFCC(self, audio_tensor):
        __MFCC = T.MFCC(
            sample_rate=self.sr,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": int(self.sr * 40 / 1000),
                "hop_length": int(self.sr * 20 / 1000),
                "n_mels": 128,
                "mel_scale": "htk",
                "center": False 
            }
        )
        mfccs = __MFCC(audio_tensor).T.unsqueeze(dim=0)
        return mfccs
    
    def add_bgNoise(self, audio_tensor):
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        background_offset = np.random.randint(0, len(background_samples) - self.n_audio)
        background_clipped = background_samples[background_offset : (background_offset + self.n_audio)]
        background_clipped = torch.squeeze(background_clipped)
        background_reshaped = F.pad(background_clipped, (0, self.n_audio - audio_tensor.shape[-1]))
        background_reshaped = background_reshaped.to(torch.float32)
        if np.random.uniform(0, 1) < 0.8:
            background_volume = np.random.uniform(0, 0.1)
        else:
            background_volume = 0
        background_mul = background_reshaped * background_volume
        background_add = background_mul + audio_tensor
        noisy_audio_tensor = torch.clamp(background_add, -1.0, 1.0)
        return noisy_audio_tensor
    
        

                    
class GoogleSpeechDatasetGenerator(object):
    def __init__(self, dataset_dir: str, save_dir: str, flag_trainer: bool, flag_background_noise: bool=False, background_data: list = [], audio_features: str="tfMFCC"):
        super().__init__()
        self.num_classes = 12
        self.labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', '_silence_', '_unknown_']
        self.dataset_dir = dataset_dir
        self.sample_count = sum(len(samples) for _, _, samples in os.walk(dataset_dir))
        self.audio_preprocessor = AudioPreprocessor(flag_trainer=flag_trainer, flag_background_noise=flag_background_noise, background_data=background_data, audio_features=audio_features)
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            raise IsADirectoryError("Error! This directory shouldn't exist; Flaw in code pipeline")
        else:
            os.makedirs(self.save_dir)


    def __call__(self):
        with tqdm(total=self.sample_count, unit='samples', desc="Saving Tensors...") as pbar:
            for label in os.listdir(self.dataset_dir):
                if label not in self.labels:
                    continue
                label_dir = os.path.join(self.dataset_dir, label)
                if os.path.isdir(label_dir):
                    for audio_file in os.listdir(label_dir):
                        if audio_file.endswith(('.wav', '.WAV')):
                            raw_audio_wave, _ = torchaudio.load(os.path.join(label_dir, audio_file))
                            audio_wave = self.audio_preprocessor(raw_audio_wave)
                            save_label_dir = os.path.join(self.save_dir, label)
                            if not os.path.isdir(save_label_dir):
                                os.mkdir(save_label_dir)
                            torch.save(audio_wave, os.path.join(save_label_dir, f"{audio_file}_tensor.pt"))
                            pbar.update(1)
        pass

def prepare_dataset(root=".", force=False, seed=1, audio_features="tfMFCC"):


    # Download Dataset
    if os.path.exists(os.path.join(root, "SpeechCommands")) and not force:
        print("Dataset already downloaded.")
        return
    elif os.path.exists(os.path.join(root, "SpeechCommands")) and force:
        shutil.rmtree(os.path.join(root, "SpeechCommands"))
    print("Downloading dataset......")
    full_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=root,      # Specify your download directory
        download=True   # This will download the entire dataset
    )
    
    # Separate into train/test/val
    print("Rearranging/Filtering dataset......")
    raw_dataset_path = os.path.join(root, "SpeechCommands", "speech_commands_v0.02")
    filtered_dataset_path = os.path.join(root, "SpeechCommands", "google_vcdataset")
    if not os.path.isdir: 
        os.mkdir(filtered_dataset_path)
    train_dir = os.path.join(filtered_dataset_path, "train")
    test_dir = os.path.join(filtered_dataset_path, "test")
    val_dir = os.path.join(filtered_dataset_path, "val")
    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    if os.path.isdir(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)
    os.makedirs(os.path.join(train_dir, "_unknown_"))
    os.makedirs(os.path.join(test_dir, "_unknown_"))
    os.makedirs(os.path.join(val_dir, "_unknown_"))
    os.makedirs(os.path.join(train_dir, "_silence_"))
    os.makedirs(os.path.join(test_dir, "_silence_"))
    os.makedirs(os.path.join(val_dir, "_silence_"))
    train_samples, test_samples, val_samples = 0, 0, 0
    train_samples_, test_samples_, val_samples_ = 0, 0, 0
    val_per, test_per = 10, 10
    word_labels = ["down", "go", "left", "no", "off", "on", "right",
               "stop", "up", "yes", "silence", "unknown"]
    for label in os.listdir(raw_dataset_path):
        if label == "_background_noise_":
            continue
        label_dir = os.path.join(raw_dataset_path, label)
        if os.path.isdir(label_dir):
            if label not in word_labels:
                for filename in os.listdir(label_dir):
                    filepath = os.path.join(label_dir, filename)
                    res_set = which_set(filename, val_per, test_per)
                    if res_set == "training":
                        shutil.copy(filepath, os.path.join(train_dir, "_unknown_", f"{label}_{filename}"))
                        train_samples += 1
                    elif res_set == "validation":
                        shutil.copy(filepath, os.path.join(val_dir, "_unknown_", f"{label}_{filename}"))
                        val_samples += 1
                    else:
                        shutil.copy(filepath, os.path.join(test_dir, "_unknown_", f"{label}_{filename}"))
                        test_samples += 1     
                continue            
            os.makedirs(os.path.join(train_dir, label))
            os.makedirs(os.path.join(test_dir, label))
            os.makedirs(os.path.join(val_dir, label))
            for filename in os.listdir(label_dir):
                filepath = os.path.join(label_dir, filename)
                res_set = which_set(filename, val_per, test_per)
                if res_set == "training":
                    shutil.copy(filepath, os.path.join(train_dir, label))
                    train_samples += 1
                elif res_set == "testing":
                    shutil.copy(filepath, os.path.join(test_dir, label))
                    test_samples += 1
                else:
                    shutil.copy(filepath, os.path.join(val_dir, label))
                    val_samples += 1

    label = "_background_noise_"
    label_dir = os.path.join(raw_dataset_path, label)
    for filename in os.listdir(label_dir):
        filepath = os.path.join(label_dir, filename)
        if filename.endswith("md"):
            continue
        elif filename == "running_tap.wav":
            val_samples += process_audio_file(filepath, "silence", os.path.join(val_dir, "_silence_"))
        else:
            train_samples += process_audio_file(filepath, "silence", os.path.join(train_dir, "_silence_"))
    print(f"Train dataset samples = {train_samples}")
   # print(f"Test dataset samples = {test_samples}")
    print(f"Val dataset samples = {val_samples}")

    shutil.rmtree(test_dir)
    os.environ["http_proxy"] = "http://wwwinproxy.itg.ti.com:80" #for donwloading dataset over our Internal TI Network, customer can comment this 
    balanced_mlperf_test_url= "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"

    download_path=os.path.join(root,"SpeechCommands","Google_vcdataset.tar.gz")
    extract_path=os.path.join(root,"SpeechCommands","test") 
    destination_path = os.path.join(root,"SpeechCommands", "google_vcdataset") 


    # Step 1: Download the tar.gz file
    print("Downloading file...")
    response = requests.get(balanced_mlperf_test_url, stream=True)
    if response.status_code == 200:
     with open(download_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
     print("Download complete.")
    else:
     print("Failed to download the file.")
     exit(1)

# Step 2: Extract the tar.gz file
    print("Extracting the file...")
    with tarfile.open(download_path, "r:gz") as tar:
     tar.extractall(extract_path)
    print("Extraction complete.")

# Step 3: Move the entire 'test/' folder into 'google_vcdataset/'
    final_destination = os.path.join(destination_path, "test")

# Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

# Move the entire 'test/' folder into 'google_vcdataset/' instead of just its contents   
    if not os.path.exists(final_destination):
     shutil.move(extract_path, final_destination)
     print(f"Successfully moved 'test/' folder to '{final_destination}'")
    else: 
     print(f"'test/' folder already exists in '{destination_path}'. Skipping move.")

# Cleanup: Remove the downloaded .tar.gz file   
    os.remove(download_path)

    print("Process completed successfully.")
    # Preprocess and save as .pt tensors
    print("Preparing Saved TensorDataset....")
    dataset_dir = os.path.join(root, "SpeechCommands", "google_vcdataset")
    save_dir    = os.path.join(root, "SpeechCommands", "tensor_vcdataset")
   
    os.mkdir(save_dir)
    # Prepare bg noise data
    BACKGROUND_NOISE_DIR_NAME='_background_noise_'
    background_data = prepare_background_data(os.path.join(root, "SpeechCommands", "speech_commands_v0.02"), BACKGROUND_NOISE_DIR_NAME)
    print("print the background data before trainset_generator", background_data)
    # Create dataset generator handles
    trainset_generator = GoogleSpeechDatasetGenerator(
        dataset_dir=os.path.join(dataset_dir, "train"),
        save_dir=os.path.join(save_dir, "train"),
        flag_trainer=True, flag_background_noise=True,
        background_data=background_data, audio_features=audio_features
    )
    valset_generator = GoogleSpeechDatasetGenerator(
        dataset_dir=os.path.join(dataset_dir, "val"),
        save_dir=os.path.join(save_dir, "val"),
        flag_trainer= False, audio_features=audio_features
    )
    testset_generator = GoogleSpeechDatasetGenerator(
        dataset_dir=os.path.join(dataset_dir, "test"),
        save_dir=os.path.join(save_dir, "test"),
        flag_trainer=False, audio_features=audio_features
    )
    
    # Populate the save_dir
    trainset_generator()
    valset_generator()
    testset_generator()
    print("Done Preparing Saved TensorDataset.")

class DSCNN(nn.Module):
    def __init__(self):
        super(DSCNN, self).__init__()
        self.spectrogram_length = 49
        self.dct_coefficient_count = 10
        self.label_count = 12
        filters = 64

        # Calculate initial padding for the first layer
        pads = (4 + self.spectrogram_length % 2, 1 + self.dct_coefficient_count % 2)

        # First Conv Layer
        self.conv1 = nn.Conv2d(1, filters, kernel_size=(10, 4), stride=(2, 2), padding=pads)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Depthwise and Pointwise Convolutions with dynamic padding
        self.depthwise2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn21 = nn.BatchNorm2d(filters)
        self.relu21 = nn.ReLU()
        self.pointwise2 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn22 = nn.BatchNorm2d(filters)
        self.relu22 = nn.ReLU()

        self.depthwise3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn31 = nn.BatchNorm2d(filters)
        self.relu31 = nn.ReLU()
        self.pointwise3 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn32 = nn.BatchNorm2d(filters)
        self.relu32 = nn.ReLU()

        self.depthwise4 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn41 = nn.BatchNorm2d(filters)
        self.relu41 = nn.ReLU()
        self.pointwise4 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn42 = nn.BatchNorm2d(filters)
        self.relu42 = nn.ReLU()

        self.depthwise5 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn51 = nn.BatchNorm2d(filters)
        self.relu51 = nn.ReLU()
        self.pointwise5 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn52 = nn.BatchNorm2d(filters)
        self.relu52 = nn.ReLU()

        # Dropout
        self.dropout2 = nn.Dropout(p=0.4)

        # Final Pooling, Flattening, and Fully Connected Layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten6 = nn.Flatten(start_dim=1)
        self.fc6 = nn.Linear(filters, self.label_count)

    def forward(self, x):
        # Input Conv2D Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Depthwise separable convolutions
        x = self.depthwise2(x)
        x = self.bn21(x)
        x = self.relu21(x)
        x = self.pointwise2(x)
        x = self.bn22(x)
        x = self.relu22(x)

        x = self.depthwise3(x)
        x = self.bn31(x)
        x = self.relu31(x)
        x = self.pointwise3(x)
        x = self.bn32(x)
        x = self.relu32(x)

        x = self.depthwise4(x)
        x = self.bn41(x)
        x = self.relu41(x)
        x = self.pointwise4(x)
        x = self.bn42(x)
        x = self.relu42(x)

        x = self.depthwise5(x)
        x = self.bn51(x)
        x = self.relu51(x)
        x = self.pointwise5(x)
        x = self.bn52(x)
        x = self.relu52(x)

        # Dropout
        x = self.dropout2(x)

        # Pooling and Flattening
        x = self.avgpool(x)
        x = self.flatten6(x)

        # Fully Connected Layer
        x = self.fc6(x)
        return x
    
class SavedTensorDataset(Dataset):
    def __init__(self, dataset_dir):
        self.encode_labels = {
            "down": 0,
            "go": 1,
            "left": 2,
            "no": 3,
            "off": 4,
            "on": 5,
            "right": 6,
            "stop": 7,
            "up": 8,
            "yes": 9,
            "_silence_": 10,
            "_unknown_": 11,
        }
        self.max = 85511 if "train" in dataset_dir else 4890 if "test" in dataset_dir else 10102
        self.tensors = []
        self.labels = []
        self.filenames = []
        self.class_count=torch.zeros(12)
        with tqdm(total=self.max, desc="loading Tensor...", unit='tensor') as pbar:
            for label in os.listdir(dataset_dir):
                if label not in self.encode_labels.keys():
                    continue
                label_dir = os.path.join(dataset_dir, label)
                if os.path.isdir(label_dir):
                    for tensor_file in os.listdir(label_dir):
                        audio_tensor = torch.load(os.path.join(label_dir, tensor_file), weights_only=True)
                        self.tensors.append(audio_tensor)
                        self.labels.append(label)
                        self.class_count[self.encode_labels[label]] += 1
                        pbar.update(1)
                        self.filenames.append(os.path.join(label_dir, tensor_file))  # Store full path to file
        self.labels = [self.encode_labels[label] for label in self.labels]
        
        self.class_weights = self.class_count.sum().item() / (12 * self.class_count)

    def __len__(self):
        return len(self.labels)
    
    def _get_imbalance(self):
        return self.class_weights
    
    def __getitem__(self, index):
        audio_tensor = self.tensors[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)  # Convert label to tensor
        filename = self.filenames[index]  # Include filename
        return {"audio" : audio_tensor, "label" : label}

def train(dataloader, model, loss_fn, optimizer, scheduler):
    """
    Train the model for one epoch and print running accuracy.
    """
    model.train()
    
    first_batch = next(iter(dataloader), None)
    if first_batch is None:
        print("🚨 Error: No data in dataloader! Exiting training loop.")
        return 0, 0, model, loss_fn, optimizer
    
    print(" First batch keys:", first_batch.keys())  
    print(" First batch shapes:", first_batch["audio"].shape, first_batch["label"].shape)  

    total_loss = 0
    running_corrects = 0
    total_samples = 0

    accuracy_metric = Accuracy(task="multiclass", num_classes=12).to(DEVICE)  # Adjust `num_classes` as needed

    for batch_idx, data in enumerate(dataloader):
        inputs, targets = data['audio'].to(DEVICE), data['label'].to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        batch_acc = accuracy_metric(preds, targets)
        
        # Running statistics
        total_loss += loss.item()
        running_corrects += torch.sum(preds == targets).item()
        total_samples += targets.size(0)

        # Print running accuracy every 10 batches
        if batch_idx % 10 == 0:
            print(f"🟢 Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}, Running Train Accuracy: {batch_acc:.4f}")

    # Compute final epoch loss & accuracy
    avg_loss = total_loss / len(dataloader)
    avg_acc = running_corrects / total_samples  # Overall epoch accuracy
    print(f" Epoch Finished - Avg Loss: {avg_loss:.4f}, Avg Train Accuracy: {avg_acc:.4f}")

    return avg_loss, avg_acc, model, loss_fn, optimizer

def train_model(model, train_loader, total_epochs, learning_rate):
    """
    Train the model for multiple epochs and display accuracy.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{total_epochs}")

        loss, acc, model, loss_fn, optimizer = train(train_loader, model, loss_fn, optimizer, scheduler)

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        print(f"📌 Epoch {epoch+1} - Loss: {loss:.5f} - Train Accuracy: {acc:.4f} - LR: {last_lr:.4f}")

    return model
# Validation & Testing Function
def test(model, test_loader, loss_fn):
    """
    Evaluate the model on test or validation data.
    
    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test/validation dataset.
        loss_fn: Loss function.
        device: Device to evaluate on.
    
    Returns:
        test_loss: Average test loss.
        test_acc: Average test accuracy.
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=12)

    with torch.inference_mode():
        for batch in test_loader:
            inputs, targets = batch['audio'].to(DEVICE), batch['label'].to(DEVICE)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            test_loss += loss.item()
            test_acc += accuracy(preds, targets.squeeze()).item()

    return test_loss / len(test_loader), test_acc / len(test_loader)

def calibrate(dataloader: DataLoader, model: nn.Module, loss_fn):
    """
    Calibrate the model using the provided dataloader.
    Compute the loss for the purpose of information.
    Returns the average loss.
    """
    model.train()  # Ensure the model is in evaluation mode
    avg_loss = 0.0
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
            inputs = batch["audio"].to(DEVICE)
            labels = batch["label"].clone().to(torch.long).to(DEVICE)
          #  labels = torch.tensor(batch["label"], dtype=torch.long).to(DEVICE)  # Convert to tensor and move to device
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.flatten(start_dim=1)

            # Compute the loss
            loss = loss_fn(outputs, labels)
            avg_loss += loss.item()

            print(f"Calibration Batch {batch_idx}: Loss: {loss.item()}")

    avg_loss /= total_batches
    print(f"Calibration complete. Average Loss: {avg_loss}")
    return avg_loss, model, loss_fn, None

def get_quant_model(nn_model: nn.Module, example_input: torch.Tensor, total_epochs: int, weight_bitwidth: int,
        activation_bitwidth: int, quantization_method: str, quantization_device_type: str) -> nn.Module:
    """
    Convert the torch model to quant wrapped torch model. The function requires
    an example input to convert the model.
    """

    '''
    The QAT wrapper module does the preparation like in:
    quant_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
    It also uses an appropriate qconfig that imposes the constraints of the hardware.

    The api being called doesn't actually pass qconfig_type - so it will be defined inside. 
    But if you need to pass, it can be defined.
    '''
    if weight_bitwidth is None or activation_bitwidth is None:
        '''
        # 8bit weight / activation is default - no need to specify inside.
        qconfig_type = {
            'weight': {
                'bitwidth': 8,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': 8,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            }
        }
        '''
        qconfig_type = None
    elif weight_bitwidth == 8:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            }
        }
    elif weight_bitwidth == 4:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False
            }
        }
    elif weight_bitwidth == 2:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False,
                'quant_min': -1,
                'quant_max': 1,
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False
            }
        }
    else:
        raise RuntimeError("unsupported quantization parameters")
 
    if quantization_device_type == 'TINPU':
        if quantization_method == 'QAT':
            quant_model = TINPUTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        elif quantization_method == 'PTQ':
            quant_model = TINPUTinyMLPTQFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        else:
            raise RuntimeError(f"Unknown Quantization method: {quantization_method}")
        #
    elif quantization_device_type == 'GENERIC':
        if quantization_method == 'QAT':
            quant_model = GenericTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        elif quantization_method == 'PTQ':
            quant_model = GenericTinyMLPTQFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        else:
            raise RuntimeError(f"Unknown Quantization method: {quantization_method}")
        
    else:
        raise RuntimeError(f"Unknown Quantization device type: {quantization_device_type}")

    
    return quant_model

def calibrate_model(model: nn.Module, dataloader: DataLoader, total_epochs: int) -> nn.Module:
    """
    Calibrate the model for PTQ - (torch model or qat wrapped torch model) with the given train dataloader,
    learning_rate and loss are not needed for PTQ / calibration as backward / back propagation is not performed.
    loss_fn is used here only for the purpose of information - to know how good is the calibration.
    """
    # loss_fn for multi class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    
   # with torch.no_grad():
    for epoch in range(total_epochs):
        # train the model for an epoch
        loss, model, loss_fn, opti = calibrate(dataloader, model, loss_fn)
        last_lr = 0
        print(f"Epoch: {epoch+1}\t LR: {round(last_lr,5)}\t Loss: {round(loss, 5)}")

    return model

def rename_input_node_for_onnx_model(onnx_model, input_node_name: str):
    """Rename the node of an ONNX model"""
    # Update graph input name.
    onnx_model.graph.input[0].name = input_node_name
    # Update input of the first node also to correspond.
    onnx_model.graph.node[0].input[0] = input_node_name
    # Check and write out the updated model
    onnx.checker.check_model(onnx_model)
    return onnx_model


def export_model(quant_model, example_input: torch.Tensor, model_name: str, with_quant: bool = False):
    """
    Export the quantized model and print its layer-wise quantization parameters.
    """

    quant_model.to(DEVICE)

    # Convert model using FX Graph-based quantization if needed
    if with_quant:
        if hasattr(quant_model, "convert"):
         
         print(" Running `convert()` on quant_model...")
         quant_model = quant_model.convert()

        else:
         
         quant_model = quantize_fx.convert_fx(quant_model.module)
   
    #  Export to ONNX
    if hasattr(quant_model, "export"):
        print(" Exporting to ONNX...")
        quant_model.export(example_input, model_name, input_names=['input'])
    else:
        torch.onnx.export(quant_model, example_input, model_name, input_names=['input'])

    print("Model exported successfully")
    return quant_model


def validate_model(model: nn.Module, test_loader: DataLoader, num_categories: int, categories_name: List[str]) -> float:
    """
    The function takes the model (torch model or qat wrapped torch model), torch dataloader
    and the num_categories to give the confusion matrix and accuracy of the model.
    """
    model.eval()
    y_target = []
    y_pred = []

    for batch_idx, batch in enumerate(test_loader):
  
        X = batch["audio"].clone().to(DEVICE)
        y = batch["label"].clone().to(torch.long).to(DEVICE)

        # make prediction for the current batch
        pred = model(X)
        pred = pred.flatten(start_dim=1)
        # take the max probability among the classes predicted
        _, pred = torch.max(pred, 1)
        y_pred.append(pred.cpu().numpy())
        y_target.append(y.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_target = np.concatenate(y_target)
    categories_idx = np.arange(0, num_categories, 1)
    # create a confusion matrix
    cf_matrix = confusion_matrix(y_target, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[categories_name[i] for i in categories_idx],
                         columns=[categories_name[i] for i in categories_idx])
    print()
    print("Confusion Matrix")
    print(df_cm)

    # Accuracy of the model
    accuracy = np.diag(df_cm).sum()/np.array(df_cm).sum()
    return accuracy


def validate_saved_model(model_name: str, dataloader: DataLoader) -> float:
    """
    Validate the saved ONNX model using the test DataLoader.
    """
    correct_predictions = 0
    total_predictions = 0

    ort_session_options = ort.SessionOptions()
    ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort_session = ort.InferenceSession(model_name, ort_session_options)

    for batch in dataloader:
        X = batch["audio"].numpy()  
        Y = batch["label"].numpy()

        
        outputs = ort_session.run(None, {'input': X})  

        # Convert predictions to class indices
        preds = np.argmax(outputs[0], axis=1)

        # Count correct predictions
        correct_predictions += np.sum(preds == Y)
        total_predictions += Y.shape[0]

    accuracy = round(correct_predictions / total_predictions, 5)
    return accuracy

def load_calibration_indices(file_path):
    """Load indices from a calibration indices file."""
    with open(file_path, "r") as f:
        indices = [int(line.strip()) for line in f if line.strip().isdigit()]
    return indices

def create_calibration_dataset(dataset, indices):
    """Create a calibration dataset using specified indices."""
    from torch.utils.data import Subset
    return Subset(dataset, indices)

def set_seed(SEED):
        # set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    #https://pytorch.org/docs/stable/notes/randomness.html
    #these flags are for reproducibility
    cudnn.deterministic = True  
    cudnn.benchmark = False    
    os.environ['PYTHONHASHSEED'] = str(SEED)

if __name__ == '__main__':

    MODEL_NAME = "kws.onnx"
    CATEGORIES_NAME = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_EPOCHS = 36
    LEARNING_RATE = 0.5
    QUANTIZATION_METHOD = 'QAT' #'PTQ' #'QAT' #None
    WEIGHT_BITWIDTH = 8 #2 #4 #8
    ACTIVATION_BITWIDTH = 8 #8 #4 #2
    QUANTIZATION_DEVICE_TYPE = 'TINPU' #'TINPU', 'GENERIC'
    NORMALIZE_INPUT = False #True, #False
    NUM_CATEGORIES = 12  
    BATCH_SIZE = 489
    SEED = 42
    assert QUANTIZATION_DEVICE_TYPE != 'GENERIC' or (not NORMALIZE_INPUT), \
        'normalizing input with BatchNorm is not supported for the export format used for Generic Quantization. Please set NORMALIZE_INPUT to False.'
    
    root = "."
    set_seed(SEED)


    #Downloading and preparing the dataset
    prepare_dataset(root) 

    set_seed(SEED)

    #Define the dataloaders
    save_dir = os.path.join(root, "SpeechCommands", "tensor_vcdataset")
    ds_train = SavedTensorDataset(dataset_dir= os.path.join(save_dir,"train"))
    ds_test = SavedTensorDataset(dataset_dir = os.path.join(save_dir,"test"))
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    calibration_indices_file = r"quant_cal_indices_mlperf.txt"  # Path to calibration indices

   # Load calibration indices
    calibration_indices = load_calibration_indices(calibration_indices_file) 
    print(f"Loaded {len(calibration_indices)} indices for calibration.")
    validation_dataset = SavedTensorDataset(dataset_dir = os.path.join(save_dir,"val"))
    calibration_dataset = create_calibration_dataset(validation_dataset, calibration_indices)
    calibration_loader = DataLoader(dataset=calibration_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    example_batch = next(iter(test_loader))
    example_input = example_batch["audio"].float().to(DEVICE)  # Add channel dimension
 
    #Import model structure
    nn_model = DSCNN()
    # nn_model = torch.load(os.path.join('trained_models', 'pb2pth_model.pth'))
  
    nn_model = nn_model.to(DEVICE)
    
    #Train and Validate fp32 model
    nn_model = train_model(nn_model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    accuracy = validate_model(nn_model, test_loader, NUM_CATEGORIES , CATEGORIES_NAME)
    print("OG model accuracy is", accuracy)
    
  
    if QUANTIZATION_METHOD in ('QAT', 'PTQ'):

        MODEL_NAME = 'quant_' + MODEL_NAME
        quant_epochs = (NUM_EPOCHS*10) if ((WEIGHT_BITWIDTH<8) or (ACTIVATION_BITWIDTH<8)) else max(NUM_EPOCHS//2, 5)
        quant_model = get_quant_model(nn_model, example_input=example_input, total_epochs=quant_epochs, weight_bitwidth=WEIGHT_BITWIDTH, activation_bitwidth=ACTIVATION_BITWIDTH, quantization_method=QUANTIZATION_METHOD,quantization_device_type=QUANTIZATION_DEVICE_TYPE)
      
        if QUANTIZATION_METHOD == 'QAT':
            quant_learning_rate = (LEARNING_RATE/100) #if ((WEIGHT_BITWIDTH<8) or (ACTIVATION_BITWIDTH<8)) else (LEARNING_RATE/10)
         
            quant_model = train_model(quant_model, train_loader, quant_epochs, quant_learning_rate)
       
        elif QUANTIZATION_METHOD == 'PTQ':
            quant_model = calibrate_model(quant_model, calibration_loader, quant_epochs)
        
        accuracy = validate_model(quant_model, test_loader, NUM_CATEGORIES, CATEGORIES_NAME)
        print(f"{QUANTIZATION_METHOD} Model Accuracy: {round(accuracy, 5)}\n")

        quant_model = export_model(quant_model, example_input, MODEL_NAME, with_quant=True)

        
    else:
        print("No Quantization method is specified. Will not do quantization.")
    
    accuracy = validate_saved_model(MODEL_NAME, test_loader)
    print(f"Exported ONNX Quant Model Accuracy: {round(accuracy, 5)}")

    random_indices = random.sample(range(len(ds_test)), 1000)
    ds_test_subset = Subset(ds_test, random_indices)
    ds_test_loader = DataLoader(dataset=ds_test_subset, batch_size=100, shuffle=False, drop_last=False)
    accuracy = validate_saved_model(MODEL_NAME, ds_test_loader)
    print(f"Exported ONNX Quant Model Accuracy on 1000 samples: {accuracy}")
  