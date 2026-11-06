 # Google Speech Commands 12-Class Dataset

  A TensorLab-compatible 12-class variant of the Google Speech Commands dataset for keyword spotting and audio            classification experiments using TinyML models such as DSCNN.
                                                                                                                          ---

  ## Dataset Source

  Downloaded automatically via TorchAudio:

  ```python
  torchaudio.datasets.SPEECHCOMMANDS(
      root=root,
      url="speech_commands_v0.02",
      folder_in_archive="SpeechCommands",
      download=True,
  )
 ```
  ## Classes

```
  ┌────────────────┬──────────────────────────────────────────────────────────┐
  │      Type      │                          Labels                          │
  ├────────────────┼──────────────────────────────────────────────────────────┤
  │ Known keywords │ down, go, left, no, off, on, right, stop, up, yes        │
  ├────────────────┼──────────────────────────────────────────────────────────┤
  │ Unknown        │ _unknown_ - all non-keyword words (e.g. bird, cat, tree) │
  ├────────────────┼──────────────────────────────────────────────────────────┤
  │ Silence        │ _silence_ - 1-second clips from _background_noise_/      │
  └────────────────┴──────────────────────────────────────────────────────────┘
```

  ## Quick Start

  - Install dependencies
  
  ```python
  python -m pip install torch torchaudio scipy pydub numpy tqdm
```

  - Generate the dataset

  ```python
  python generate_dataset.py
  ```

  ## Output Structure

```
  SpeechCommands/
  ├── speech_commands_v0.02/       # Original downloaded dataset
  │   ├── down/
  │   ├── go/
  │   └── _background_noise_/
  └── classes/                     # TensorLab-ready dataset
      ├── down/
      ├── go/
      ├── left/
      ├── no/
      ├── off/
      ├── on/
      ├── right/
      ├── stop/
      ├── up/
      ├── yes/
      ├── _silence_/
      └── _unknown_/
```

  What generate_dataset.py Does:

  1. Downloads Google Speech Commands v0.02 via TorchAudio
  2. Copies the 10 keyword classes into their own folders
  3. Maps all other word classes into _unknown_ (prefixing the original label to avoid filename collisions)
  4. Splits _background_noise_/ audio into 1-second clips for _silence_
  5. Saves everything under SpeechCommands/classes/



  ## How _silence_ Samples Are Generated
                                                Silence samples are not recorded speech - they are synthetic clips cut from the background noise audio files in_background_noise_/.                                                                                                
  **Source files:** all `.wav` files inside `SpeechCommands/speech_commands_v0.02/_background_noise_/`

  **Process (`create_silence_samples`):**

  1. Each background noise file is loaded in full via `pydub.AudioSegment`
  2. Raw PCM samples are extracted as a NumPy array
  3. The array is sliced into 1-second windows using a sliding loop with 50% overlap.
  4. Each segment is written to classes/_silence_/ as a 16-bit .wav file

  ## Recommended preset

```
  GoogleSpeechCommands_MFCC_Default = dict(
      data_processing_feature_extraction=dict(
          sampling_rate=16000,
          audio_duration_ms=1000,
          audio_feature="MFCC",
          n_mfcc=10,
          n_mels=40,
          frame_length_ms=30,
          frame_step_ms=20,
          normalize_audio=True,
          mono=True,
          variables=1,
          feat_ext_transform=["MFCC"],
          data_proc_transforms=[],
      ),
      common=dict(
          task_type=TASK_TYPE_AUDIO_CLASSIFICATION,
      ),
  )
```


  ## MFCC Feature Extraction

  MFCCs (Mel Frequency Cepstral Coefficients) compactly represent the frequency characteristics of speech, making them
  well-suited for keyword spotting.

```
  ┌───────────────────┬──────────┐
  │     Parameter     │  Value   │
  ├───────────────────┼──────────┤
  │ Sampling rate     │ 16000 Hz │
  ├───────────────────┼──────────┤
  │ Audio duration    │ 1000 ms  │
  ├───────────────────┼──────────┤
  │ Frame length      │ 30 ms    │
  ├───────────────────┼──────────┤
  │ Frame step        │ 20 ms    │
  ├───────────────────┼──────────┤
  │ MFCC coefficients │ 10       │
  ├───────────────────┼──────────┤
  │ Mel bins          │ 40       │
  └───────────────────┴──────────┘
```
  Output feature shape: [N, 1, 49, 10]
  (batch size × 1 channel × 49 time frames × 10 MFCC coefficients)

 
  ## DSCNN Model

  The recommended model is DSCNN (Depthwise Separable Convolutional Neural Network), designed for efficient TinyML
  inference on our NPU

  Architecture

  Conv10x4 / stride 2
  Dropout
  Depthwise3x3 + Pointwise1x1  ×4
  Dropout
  AdaptiveAvgPool
  Fully Connected (→ 12 classes)

  Filters: 64 | Output classes: 12

  ### Why DSCNN

  A standard convolution performs spatial filtering and channel mixing in one operation. DSCNN splits this into:

  - Depthwise conv - spatial filtering independently per channel
  - Pointwise conv - 1×1 convolution for channel mixing

  This reduces computation and model size while maintaining strong keyword spotting accuracy, making it suitable for
  embedded deployment.

