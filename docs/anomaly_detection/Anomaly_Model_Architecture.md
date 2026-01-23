# **Anomaly Detection Model Architecture & Theory**

## **Table of Contents**
[1. Introduction](#1-introduction-to-anomaly-detection)<br>
[2. Autoencoder Fundamentals](#2-autoencoder-fundamentals)<br>
[3. Reconstruction Error](#3-reconstruction-error-the-anomaly-score)<br>
[4. Threshold Calculation](#4-threshold-calculation-statistical-method)<br>


## **1. Introduction to Anomaly Detection**

Anomaly detection is a machine learning technique used to identify patterns in data that deviate significantly from expected "normal" behavior. Unlike classification, where the model learns to distinguish between multiple known classes, anomaly detection focuses on learning what "normal" looks like and flagging anything that doesn't fit that pattern.


### **1.1 Definition**

**Anomaly detection** is the process of identifying rare events, observations, or data points that differ significantly from the majority of the data. These deviations are called **anomalies**, **outliers**, or **faults**.

**Key characteristics of anomalies:**
- **Rare:** Occur infrequently compared to normal data
- **Different:** Have distinct patterns or features from normal samples
- **Unpredictable:** May represent previously unseen failure modes

**Real-world applications:**

| Domain | Normal Behavior | Anomalies |
|-------|----------------|-----------|
| **Predictive Maintenance** | Healthy motor vibration | Bearing wear, imbalance, misalignment |
| **Manufacturing** | Defect-free products | Cracks, scratches, dimensional errors |
| **Medical Diagnosis** | Healthy vital signs | Arrhythmia, abnormal glucose levels |
| **IoT Sensors** | Expected sensor readings | Sensor drift, hardware failure |


### **1.2 Supervised vs Semi-Supervised vs Unsupervised**

Anomaly detection can be approached using different learning paradigms, depending on the availability of labeled data:


#### **Supervised Anomaly Detection (Classification)**

**Training data required:**
-  Labeled normal samples
-  Labeled anomaly samples (multiple types)

**How it works:**
- Train a classifier (e.g., CNN, SVM) to distinguish between normal and anomaly classes
- At inference: Classify input as one of the known classes

**Example:**
```
Training data:
├── Class 0: Normal bearing (1000 samples)
├── Class 1: Inner race fault (200 samples)
├── Class 2: Outer race fault (200 samples)
└── Class 3: Ball fault (200 samples)

Model learns: "Which fault type is this?"
```

**Advantages:**
-  Can identify specific anomaly types
-  High accuracy when anomaly types are known

**Disadvantages:**
-  Requires labeled anomaly samples for training
-  **Cannot detect new/unseen anomaly types**
-  High labeling effort (need examples of all fault types)


#### **Semi-Supervised Anomaly Detection (Autoencoder - Our Approach)**

**Training data required:**
-  Labeled normal samples only
-  No anomaly samples needed for training

**How it works:**
- Train an autoencoder to reconstruct normal data with low error
- At inference: High reconstruction error → Anomaly

**Example:**
```
Training data:
└── Normal bearing (1000 samples)

Model learns: "What does normal look like?"

At test time:
├── Normal bearing → Low reconstruction error → Normal 
├── Inner race fault → High reconstruction error → Anomaly 
├── Outer race fault → High reconstruction error → Anomaly 
└── New fault type (never seen) → High reconstruction error → Anomaly 
```

**Advantages:**
-  **Only needs normal data for training** (easy to collect)
-  **Detects unseen anomaly types** (generalizes)
-  Lower labeling effort

**Disadvantages:**
-  Cannot distinguish between anomaly types (just "normal" vs "not normal")
-  Requires good coverage of normal operating conditions


#### **Unsupervised Anomaly Detection**

**Training data required:**
-  Unlabeled data (mix of normal and anomaly)

**How it works:**
- Clustering-based methods (e.g., DBSCAN, Isolation Forest)
- Identify samples that are "far" from dense regions

**Example:**
```
Training data:
└── Mixed data (unlabeled, mostly normal with some anomalies)

Model learns: "Which samples are outliers?"
```

**Advantages:**
-  No labels needed at all

**Disadvantages:**
-  Assumes anomalies are rare (may not hold)
-  Less accurate than supervised/semi-supervised
-  Difficult to tune (what is "outlier"?)


#### **Comparison Table**

| Aspect | Supervised | **Semi-Supervised (Our Approach)** | Unsupervised |
|--------|------------|-------------------------------------|--------------|
| **Training data** | Normal + Anomaly (labeled) | **Normal only** | Mixed (unlabeled) |
| **Labeling effort** | High | **Low** | None |
| **Detects unseen anomalies** |  No | ** Yes** |  Yes |
| **Accuracy** | High (known types) | **High (if good normal data)** | Medium |
| **Use case** | Known, fixed fault types | **Unknown or evolving faults** | Exploratory analysis |

### **1.3 Why Semi-Supervised (Autoencoder Approach)?**

For industrial anomaly detection (motors, fans, manufacturing), the **semi-supervised autoencoder approach** is ideal because:

#### **Reason 1: Anomalies are Rare and Diverse**

**Challenge:**
- In production, systems operate normally 99%+ of the time
- Failures are rare events
- New failure modes emerge over time

**Supervised approach problem:**
```
Need 100+ samples of each fault type:
├── Bearing wear (hard to collect)
├── Imbalance (need to induce fault)
├── Misalignment (requires setup)
└── New fault type (unknown!) 
```

**Semi-supervised solution:**
```
Only need normal data:
└── Healthy operation (easy to collect)

Detects all faults automatically, including new types
```

#### **Reason 2: Collecting Anomaly Data is Expensive**

**Cost considerations:**

| Approach | Data Collection Cost |
|----------|---------------------|
| **Supervised** | High (need to induce faults, run-to-failure tests) |
| **Semi-supervised** | **Low (just collect normal operation data)** |
| **Unsupervised** | Medium (need diverse data, may include anomalies) |

**Example: Motor bearing monitoring**
- Normal data: Run motor under various loads/speeds (easy)
- Fault data: Run motor until bearing fails (expensive, time-consuming)


#### **Reason 3: Generalization to Unseen Faults**

**Key advantage:** The model learns "what normal looks like" rather than "what specific faults look like."

**Example scenario:**
```
Training: Only normal bearing vibration

Deployment encounters:
├── Erosion (never seen) → High error → Detected 
├── Contamination (never seen) → High error → Detected 
├── Flaking (never seen) → High error → Detected 
└── New failure mode (completely unknown) → High error → Detected 
```

**Why this works:** Any deviation from learned "normal" manifold triggers high reconstruction error.


#### **Reason 4: Practical Deployment Advantages**

 **Faster time-to-deployment**
- Don't need to wait for faults to occur
- Can deploy model with just normal data

 **Lower maintenance burden**
- Don't need to retrain when new fault types emerge
- Model already generalizes to unseen anomalies

 **Simpler data pipeline**
- Just collect normal operation data
- No need for complex fault labeling


#### **When NOT to Use Semi-Supervised Approach**

**Use supervised classification instead when:**

 You need to **identify specific fault types** (not just detect "something wrong")
- Example: "Is this inner race fault or outer race fault?"

 You have **abundant labeled anomaly data** already available
- Example: Historical database with 1000+ samples per fault type

 Anomalies are **not rare** (50%+ of data)
- Semi-supervised assumes normal data dominates

# **2. Autoencoder Fundamentals**

This section explains what autoencoders are, how they work, and why they are particularly well-suited for anomaly detection.



## **2.1 What is an Autoencoder?**

An **autoencoder** is a type of neural network designed to learn an efficient compressed representation of input data. It consists of two main parts:

1. **Encoder:** Compresses the input into a lower-dimensional representation (latent space)
2. **Decoder:** Reconstructs the original input from the compressed representation

**Training objective:** Make the output as similar to the input as possible.

```
Input → Encoder → Latent Space → Decoder → Output
  X   →    f    →      z       →    g    →   X̂

Goal: Minimize ||X - X̂||² (reconstruction error)
```

**Key insight:** The autoencoder learns to capture the **essential patterns** in the data while discarding noise and irrelevant details.



## **2.2 Architecture Overview**

### **High-Level Structure**

![Diagram showing encoder-decoder structure](autoencoder_architecture.png)

```

┌─────────────────────────────────────────────────────┐
│                    INPUT                            │
│          (e.g., 256 samples × 3 channels)           │
│                  = 768 features                     │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│                   ENCODER                           │
│              (Dimensionality Reduction)             │
│                                                     │
│  Layer 1: 768 → 512 features  (Linear + ReLU)       │
│  Layer 2: 512 → 256 features  (Linear + ReLU)       │
│  Layer 3: 256 → 128 features  (Linear + ReLU)       │
│  Layer 4: 128 → 64  features  (Linear + ReLU)       │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│               LATENT SPACE                          │
│         (Compressed representation: 64 features)    │
│                                                     │
│     Contains essential patterns from input          │
│     Compression ratio: 768/64 = 12×                 │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│                   DECODER                           │
│            (Dimensionality Expansion)               │
│                                                     │
│  Layer 1: 64  → 128 features  (Linear + ReLU)      │
│  Layer 2: 128 → 256 features  (Linear + ReLU)      │
│  Layer 3: 256 → 512 features  (Linear + ReLU)      │
│  Layer 4: 512 → 768 features  (Linear)             │
└─────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────┐
│              RECONSTRUCTED OUTPUT                   │
│                  768 features                       │
│          (reshaped to 256 samples × 3 channels)     │
└─────────────────────────────────────────────────────┘
```



### **Dimensionality Flow**

The encoder progressively **compresses** the input:

```
Input:           768 features  (256 samples × 3 channels)
                  ↓
Encoder Layer 1: 512 features
                  ↓
Encoder Layer 2: 256 features
                  ↓
Encoder Layer 3: 128 features
                  ↓
Encoder Layer 4:  64 features
                  ↓
LATENT SPACE:     64 features  (8.3% of original size)
```

The decoder then **expands** back to original dimensions:

```
LATENT SPACE:     64 features
                  ↓
Decoder Layer 1: 128 features
                  ↓
Decoder Layer 2: 256 features
                  ↓
Decoder Layer 3: 512 features
                  ↓
Decoder Layer 4: 768 features
                  ↓
Output:          768 features  (same as input)
```

**Key observation:** The bottleneck at the latent space (64 features) forces the network to learn compressed representations. The network cannot simply memorize the input—it must extract the most important patterns.



## **2.3 Key Components**

### **Linear (Fully Connected) Layers**

Transform features from one dimensional space to another:

```
y = W × x + b

Where:
  x = Input features
  W = Weight matrix (learned during training)
  b = Bias vector (learned during training)
  y = Output features
```

Each output feature is a **weighted combination** of all input features, allowing the network to learn complex relationships.

**Note:** We can also use **convolutional layers** to achieve the same effect. Convolution layers capture the local structure of the input and learns to extract features across channels. 



### **ReLU Activation**

Introduces non-linearity (enables learning of complex patterns):

```
ReLU(x) = max(0, x)

If x > 0: output = x
If x ≤ 0: output = 0
```

**Why ReLU?**
- Simple and fast to compute
- Avoids vanishing gradient problem
- Without activation functions, the network would be purely linear (just matrix multiplications)

**Note:** The final decoder layer has **no activation**, allowing the output to have negative values.



### **Encoder-Decoder Symmetry**

Notice the **mirror structure**:

```
ENCODER (Compression)          DECODER (Expansion)
─────────────────────          ───────────────────
768 → 512                      64  → 128
512 → 256                      128 → 256
256 → 128                      256 → 512
128 → 64                       512 → 768
```

This symmetry ensures:
- Balanced architecture 
- Smooth gradient flow during training
- Each encoder layer has a corresponding decoder layer to "undo" the compression



## **2.4 Why Autoencoders for Anomaly Detection?**

### **The Core Principle**

**Training phase:**
- Autoencoder is trained **only on normal data**
- It learns to compress and reconstruct normal patterns efficiently
- Reconstruction error on normal samples is **low**

**Inference phase:**
- **Normal samples:** Model has seen similar patterns → Low reconstruction error 
- **Anomaly samples:** Model has never seen these patterns → **High reconstruction error** 

```
┌─────────────────────────────────────────────────────┐
│              TRAINING (Normal Data Only)            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Model learns: "What do normal patterns look like?" │
│                                                     │
│  Normal vibration → Encode → Decode → ≈ Input       │
│  Reconstruction Error: LOW                          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              INFERENCE (Test Data)                  │
└─────────────────────────────────────────────────────┘
                        ↓
         ┌──────────────────────────────┐
         │                              │
    Normal sample               Anomaly sample
         │                              │
         ↓                              ↓
  Encode → Decode              Encode → Decode
         ↓                              ↓
  Output ≈ Input              Output ≠ Input
         ↓                              ↓
  Error: LOW                   Error: HIGH
         ↓                              ↓
  Prediction: Normal          Prediction: Anomaly
```

### **What the Encoder Learns**

The encoder learns to:
- **Extract essential features** from input (frequency content, amplitude patterns, temporal relationships)
- **Discard noise and irrelevant details**
- **Create compressed representation** that captures "normality"

**For normal data:**
```
Input:  Normal vibration pattern
Encoder: Recognizes familiar patterns
Latent: [0.8, -0.3, 1.2, 0.5, ..., 0.9]  ← Fits learned space
```

**For anomaly data:**
```
Input:  Bearing fault vibration
Encoder: Tries to encode unfamiliar patterns
Latent: [2.5, -1.8, 0.1, 3.2, ..., -0.4]  ← Doesn't fit learned space
```



### **What the Decoder Learns**

The decoder learns to:
- **"Unpack" latent representations** back to original dimensions
- **Generate normal-looking patterns** from the latent space
- **Reconstruct accurately** when latent representation is within learned manifold

**Key insight:** The decoder is a "normal pattern generator"

**For normal data:**
```
Latent: Within learned space
Decoder: Recognizes pattern, reconstructs accurately
Output: ≈ Input
Error: Low
```

**For anomaly data:**
```
Latent: Outside learned space
Decoder: Tries to "force fit" using normal patterns it knows
Output: Looks more "normal" than input (distorted)
Error: High (input ≠ output)
```

## **2.5 Advantages of Autoencoder Approach**

 **Unsupervised for anomalies**
- No need to label anomaly types
- Only need normal data

 **Generalizes to unseen anomalies**
- Any deviation from normal → High error
- Works on faults never seen during training

 **Learns complex patterns**
- Can capture non-linear relationships
- Handles multi-dimensional data (multi-channel sensors)

 **Scalable**
- Once trained, fast inference
- Can deploy on edge devices (MCUs)



## **2.6 Limitations**

 **Requires good normal data**
- Must cover full range of normal operating conditions
- If training data is narrow, valid conditions may be flagged as anomalies

 **Threshold selection is critical**
- Need to tune threshold (mean + k×std) based on application
- Trade-off between false positives and false negatives

 **Cannot distinguish anomaly types**
- Only outputs "normal" vs "anomaly"
- Cannot tell you "this is bearing fault vs imbalance"
- (For classification, use supervised approach)

 **Subtle anomalies may be missed**
- If anomaly is very similar to normal, error may not exceed threshold
- Model may "learn" to reconstruct slight deviations if training data has noise


# **3. Reconstruction Error: The Anomaly Score**

Reconstruction error is the core metric used to detect anomalies. This section explains what it is, how it's calculated, and why it serves as an effective anomaly indicator.


## **3.1 Definition**

**Reconstruction error** measures how well the autoencoder reconstructs the input. It quantifies the difference between the original input and the model's output.

**Formula (Mean Squared Error - MSE):**

```
MSE = (1/N) × Σ(input[i] - output[i])²

Where:
  N = Total number of elements in the input
  input[i] = Original value at position i
  output[i] = Reconstructed value at position i
```

**Example:**

For a 256-sample, 3-channel input:
```
N = 256 samples × 3 channels = 768 elements

MSE = (1/768) × Σ(input[i] - output[i])²
```



## **3.2 Interpretation**

The reconstruction error tells us how "different" the output is from the input:

### **Low MSE (e.g., 0.05)**
```
Input:  [1.2, -0.8, 2.1, -1.5, ...]
Output: [1.1, -0.9, 2.0, -1.4, ...]
        ↑     ↑     ↑     ↑
      Close match → Low error
```

**Interpretation:** Model reconstructed the input accurately
**Prediction:** Normal sample



### **High MSE (e.g., 10.0 )**
```
Input:  [1.2, -0.8, 2.1, -1.5, 3.5, ...]
Output: [0.5, -0.2, 1.1, -0.8, 1.2, ...]
        ↑     ↑     ↑     ↑     ↑
      Poor match → High error
```

**Interpretation:** Model failed to reconstruct the input
**Prediction:** Anomaly 



## **3.3 Decision Rule**

The reconstruction error is compared to a **threshold** to make the anomaly decision:

```python
if reconstruction_error > threshold:
    prediction = "Anomaly"
else:
    prediction = "Normal"
```

- Threshold is set based on normal training data statistics
- It represents the boundary between "normal" and "anomaly"



## **3.4 Why High Reconstruction Error Indicates Anomalies**

### **For Normal Data**

**What happens:**
1. Input passes through encoder
2. Latent representation **fits the learned normal space**
3. Decoder recognizes the pattern
4. Output closely matches input
5. **Low reconstruction error**

**Example:**
```
Input:  Healthy motor vibration (50 Hz, 1.2 m/s²)
Encode: [0.8, -0.3, 1.2, 0.5, ..., 0.9]  ← Within learned space
Decode: Smooth 50 Hz pattern
Output: ≈ Input
MSE:    0.05 (low) → Normal 
```



### **For Anomaly Data**

**What happens:**
1. Input passes through encoder
2. Latent representation **doesn't fit the learned normal space**
3. Decoder tries to "force fit" using normal patterns it knows
4. Output looks more "normal" than input (distorted)
5. **High reconstruction error**

**Example:**
```
Input:  Bearing fault vibration (irregular, high-frequency spikes)
Encode: [2.5, -1.8, 0.1, 3.2, ..., -0.4]  ← Outside learned space
Decode: Attempts to reconstruct as "normal-looking" pattern
Output: ≠ Input (forced to look normal)
MSE:    15.8 (high) → Anomaly 
```

## **3.5 Reconstruction Error Distribution**

After training, ModelMaker analyzes reconstruction error distributions:

### **Normal Training Data**
```
Ex:
Errors typically cluster around low values:
Mean: 1.66
Std:  1.97
Most samples: 0.5 - 5.0
```

### **Normal Test Data**
```
Ex:
Similar to training (good generalization):
Mean: 2.85
Std:  1.34
Most samples: 1.0 - 6.0
```

### **Anomaly Test Data**
```
Ex:
Errors are much higher:
Mean: 141.99
Std:  112.76
Most samples: 50.0 - 300.0
```

**Good separation:** `mean_anomaly >> mean_normal` indicates the model can distinguish anomalies effectively.



# **4. Threshold Calculation: Statistical Method**

The threshold is the critical value that separates "normal" from "anomaly" predictions. This section explains how ModelMaker calculates thresholds and the intuition behind the statistical approach.


## **4.1 Formula**

ModelMaker uses a **statistical threshold** based on the reconstruction errors from **normal training data**:

```
threshold = mean_train + k × std_train

Where:
  mean_train = Average reconstruction error on normal training data
  std_train  = Standard deviation of reconstruction errors on normal training data
  k          = Multiplier (ranges from 0 to 4.5)
```

## **4.2 Intuition**

### **Why Mean + k×Std?**

The formula is based on the assumption that reconstruction errors on normal data follow a **Gaussian (normal) distribution**:

![](normal_distribution.webp)

**k represents how many standard deviations away from the mean:**

| k | Threshold | Interpretation (if Gaussian) |
|---|-----------|------------------------------|
| 1 | mean + 1σ | ~84% of normal samples below threshold |
| 2 | mean + 2σ | ~97.5% of normal samples below threshold |
| 3 | mean + 3σ | ~99.7% of normal samples below threshold |
| 4 | mean + 4σ | ~99.99% of normal samples below threshold |

**Key insight:** Higher k means stricter threshold (fewer normal samples flagged as anomalies).


## **4.3 k Value Impact**

The choice of k determines the **trade-off between false positives and false negatives**:

### **Lower k (e.g., k = 1)**

```
threshold = mean_train + 1 × std_train
```

**Effect:**
-  **High recall:** Catches almost all anomalies
-  **Low precision:** Many false positives (normal samples flagged)
-  **High false positive rate:** Frequent false alarms

**When to use:**
- Safety-critical applications (aircraft, medical devices)
- Missing an anomaly is very costly
- Can tolerate false alarms

### **Higher k (e.g., k = 4)**

```
threshold = mean_train + 4 × std_train
```

**Effect:**
-  **High precision:** Few false positives
-  **Low false positive rate:** Rare false alarms
-  **Lower recall:** May miss subtle anomalies

**When to use:**
- Cost-sensitive applications (avoid unnecessary maintenance)
- False alarms are expensive (production shutdowns)
- Can tolerate missing some subtle faults

### **Balanced k (e.g., k = 2.5 - 3.0)**

```
threshold = mean_train + 2.5 × std_train
```

**Effect:**
-  **Balanced recall and precision**
-  **Moderate false positive rate**
-  **Good F1 score**

**When to use:**
- Most general applications
- Need to optimize both detection and false alarm rate

## **4.4 Assumptions and Limitations**

### **Assumption: Gaussian Distribution**

The k-value interpretation assumes reconstruction errors follow a **normal (Gaussian) distribution**.

**In practice:**
- Errors may not be perfectly Gaussian
- Distribution may be skewed or have heavy tails
- k values still work, identifies best threshold based on actual errors

**ModelMaker's approach:**
- Tests multiple k values (0 to 4.5)
- User selects based on actual metrics (precision/recall)
- No strict reliance on Gaussian assumption


### **Limitation: Outliers in Training Data**

If training data contains **mislabeled anomalies**, the threshold may be too high. In this case, the model may not be able to distinguish between normal and anomaly samples.

---
