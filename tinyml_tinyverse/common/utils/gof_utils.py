import numpy as np
import pandas as pd
import os
import pywt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from logging import getLogger
import textwrap
import math
import pickle

# Function that computes Fourier transform + Absolute value of positive frequencies + Logarithmic scaling
def compute_fourier_abs_log(frame_size, classes_dir, frame_skip,class_names): 
    logger = getLogger("root.utils.compute_fourier_abs_log")
    logger.info("Starting Fourier transform + Abs + Log computation...")
    fft_features = []
    frame_labels = []

    class_to_label = {name: idx for idx, name in enumerate(class_names)} #Assign numeric label to each class
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    for class_name in class_names:

        logger.info(f"Processing class: {class_name}")
        class_dir = os.path.join(classes_dir, class_name)

        if not os.path.isdir(class_dir):
            logger.warning(f"Skipping non-directory: {class_dir}")
            continue

        class_label = class_to_label[class_name]
        
        for file in os.listdir(class_dir):
            
            file_path = os.path.join(class_dir, file)

            # Reading the first value from the file
            with open(file_path, 'r') as f:
                first_line = f.readline().strip().split(',')
                first_value = first_line[0].strip()

            # Case when header is not present
            if first_value.replace('.','',1).isnumeric():  # If the first value is an int or float, assume no header
                logger.warning(f"File '{file_path}' starts with a number. Assuming no header.")
                df = pd.read_csv(file_path, header=None).to_numpy()  # Read without a header
                
            # Case when header is present
            else:
                header_row = pd.read_csv(file_path, nrows=0).columns  # Read header row
                non_time_columns = [col for col in header_row if 'time' not in col.lower()]

                if not non_time_columns:
                    logger.warning(f"No usable columns after excluding 'time' column in: {file_path}")
                    continue

                # Read data excluding "time" columns
                df = pd.read_csv(file_path, usecols=non_time_columns).to_numpy()


            frame_num = df.shape[0] // frame_size # Number of frames

            for i in range(0, frame_num, frame_skip):  # Skip frames based on the 'skip' parameter
                frame_labels.append(class_label)
                fft_log_complete = []

                # Apply FFT to all columns
                for col_num in range(df.shape[1]):
                    
                    # Extract frame
                    frame = df[i * frame_size:(i + 1) * frame_size, col_num] 
                    # Perform FFT 
                    fft = np.fft.fft(frame, n=frame_size) 
                    # Magnitude of positive frequencies 
                    fft_positive = np.abs(fft[:frame_size // 2])  
                    # Logarithmic scaling (log(1+x) instead of log x is done to prevent issues with taking the log of 0)
                    fft_log = np.log1p(fft_positive)  
                    fft_log_complete.append(fft_log)

                fft_features.append(np.concatenate(fft_log_complete).flatten())

    logger.info("Fourier transform + Abs + Log computation completed.")
    fft_features = np.array(fft_features, dtype=np.float32) 
    title = "FFT+Abs+Log"
    return fft_features, frame_labels, title, class_names


# Function that computes Wavelet Transform and statistical features
def compute_wavelet(frame_size, classes_dir, frame_skip,class_names):
    logger = getLogger("root.utils.compute_wavelet")
    logger.info("Starting Wavelet Transform computation...")
    wavelet_features = []
    frame_labels = []
    wavelet = 'db4' #Using Daubechies Wavelet which has 4 coefficients

    class_to_label = {name: idx for idx, name in enumerate(class_names)} #Assign numeric label to each class
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        logger.info(f"Processing class: {class_name}")
        class_dir = os.path.join(classes_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Skipping non-directory: {class_dir}")
            continue
        class_label = class_to_label[class_name]
        

        for file in os.listdir(class_dir):

            file_path = os.path.join(class_dir, file)

            # Reading the first value from the file
            with open(file_path, 'r') as f:
                first_line = f.readline().strip().split(',')
                first_value = first_line[0].strip()

            # Case when header is not present
            if first_value.replace('.','',1).isnumeric():  # If the first value is a int or float, assume no header
                logger.warning(f"File '{file_path}' starts with a number. Assuming no header.")
                df = pd.read_csv(file_path, header=None).to_numpy()  # Read without a header
                
            # Case when header is present
            else:
                header_row = pd.read_csv(file_path, nrows=0).columns  # Read header row
                non_time_columns = [col for col in header_row if 'time' not in col.lower()]

                if not non_time_columns:
                    logger.warning(f"No usable columns after excluding 'time' column in: {file_path}")
                    continue

                # Read data excluding "time" columns
                df = pd.read_csv(file_path, usecols=non_time_columns).to_numpy()
                
            #logging.info(f"Processing file: {file_path}")
            frame_num = df.shape[0] // frame_size  # Number of frames

            for i in range(0, frame_num, frame_skip):  # Skip frames based on the 'skip' parameter
                frame_labels.append(class_label)
                features = []

                # Apply Wavelet Transform to all columns
                for col_num in range(df.shape[1]):

                    #Extract frame
                    frame = df[i * frame_size:(i + 1) * frame_size, col_num]
                    # Perform Wavelet transform
                    coeffs = pywt.wavedec(frame, wavelet, level=3) 
                    # Extract statistical features from wavelet coefficients
                    for coeff in coeffs:
                        coeff = np.array(coeff)
                        features.extend([
                            np.mean(coeff),        # Mean
                            np.std(coeff),         # Standard deviation
                            np.max(coeff),         # Maximum
                            np.min(coeff),         # Minimum
                            np.sum(np.abs(coeff))  # Energy
                        ])

                wavelet_features.append(np.array(features, dtype=np.float32)) 
    logger.info("Wavelet Transform computation completed.")
    wavelet_features = np.array(wavelet_features, dtype=np.float32) 
    title = "WT"
    return wavelet_features, frame_labels, title, class_names

# Scales the features to a range of [0, 1] using MinMax scaling
def scale_minmax(features, title):
    logger = getLogger("root.utils.scale_minmax")
    logger.info("Starting MinMax scaling...")
    scaler = MinMaxScaler()  
    scaled_features = scaler.fit_transform(features)
    title = title + "+MinMax Scaler"
    logger.info("MinMax scaling completed. Features are now scaled to [0, 1].")
    return scaled_features, title

# Standardizing features by removing the mean and scaling to unit variance
def scale_std(features, title):
    logger = getLogger("root.utils.scale_std")
    logger.info("Starting standardization (StandardScaler)...")
    scaler = StandardScaler() 
    scaled_features = scaler.fit_transform(features) 
    title = title + "+Standard Scaler"  
    logger.info("Standardization completed. Features are now scaled to have zero mean and unit variance.")
    return scaled_features, title

# Perform Dimensionality Reduction (to 3 components) using PCA
def dim_redn_pca(scaled_features, title):
    logger = getLogger("root.utils.dim_redn_pca")
    logger.info("Starting dimensionality reduction using PCA...")
    pca = PCA(n_components=3, random_state=42) 
    reduced_features = pca.fit_transform(scaled_features)
    title = title + "+PCA"  
    logger.info("PCA completed. Features reduced to 3 dimensions.")
    return reduced_features, title

# Perform Dimensionality Reduction (to 3 components) using TSNE
def dim_redn_tsne(scaled_features, title):
    logger = getLogger("root.utils.dim_redn_tsne")
    logger.info("Starting dimensionality reduction using TSNE...")
    tsne = TSNE(n_components=3, random_state=42)  
    reduced_features = tsne.fit_transform(scaled_features)
    title = title + "+TSNE" 
    logger.info("TSNE completed. Features reduced to 3 dimensions.")
    return reduced_features, title

#Plots the 3D graph of the reduced features.
def plot_gof_graph(current_plot, row_num, col_num, reduced_features, frame_labels, fig, title, class_names):
    logger = getLogger("root.utils.plot_gof_graph")
    logger.info("Starting to plot the 3D graph...")
    ax = fig.add_subplot(row_num, col_num, current_plot, projection='3d')
    ax.set_title(title, fontsize=11)
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=frame_labels, cmap='viridis')
    legend_patches = [
        mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=class_names[i])
        for i in range(len(class_names))
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1, 1))
    logger.info(f"Added 3D subplot at position ({row_num}, {col_num}, {current_plot}).")
    logger.info("3D graph plotting completed.")

# Perform Goodness of Fit Test
def goodness_of_fit_test(frame_size, classes_dir, output_dir,class_names):

    logger = getLogger("root.utils.goodness_of_fit_test")
    logger.info("Starting Goodness of Fit test...")
   
    num_of_frames = 0
    for class_name in class_names:
        class_dir = os.path.join(classes_dir, class_name)
        files = os.listdir(class_dir)
       
        for file_name in files:
            file_path = os.path.join(class_dir, file_name)

            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
                num_of_rows = len(df)

            elif file_name.endswith('.npy'):
                data = np.load(file_path)
                num_of_rows = data.shape[0]

            elif file_name.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    num_of_rows = data.shape[0]

            elif file_name.endswith('.txt'):
                with open(file_path, 'r') as f:
                    num_of_rows = sum(1 for _ in f)
            
            num_of_frames += num_of_rows // frame_size
    
    frame_skip=math.ceil(num_of_frames/3000)
    logger.info(f"Total number of frames: {num_of_frames}")
    logger.info(f"Frame skip: {frame_skip}")
    logger.info(f"Total number of frames after frame skip: {num_of_frames//frame_skip}")
    # List of transformation, scaling, and dimensionality reduction functions
    transforms = [compute_fourier_abs_log,compute_wavelet]
    scalers = [scale_std,scale_minmax]
    dim_redn = [dim_redn_pca,dim_redn_tsne]

    # Create figure
    figw, figh = 18.0, 40.0
    fig = plt.figure(figsize=(figw, figh))
    fig.patch.set_facecolor('white')  # Set background color


    fig.suptitle("GOODNESS OF FIT ANALYSIS", fontsize=18, fontweight='bold', ha='center')

    fig.text(0.5, 0.966, f"Frame Size: {frame_size} | Frame Skip: {frame_skip}",
            ha='center', fontsize=12, fontstyle='italic', color='gray')

    # Text properties
    title_font = {'fontsize': 14, 'fontweight': 'bold'}
    disclaimer_title_font={'fontsize': 14, 'fontweight': 'bold', 'fontstyle':'italic'}
    content_font = {'fontsize': 13}

    
    max_width_title = 115   # Maximum width for wrapping text
    max_width_content=140
    notes = [
        ("Disclaimer: Not all eight plots need to have separable clusters. Each plot represents a different method to analyze a time-series classification dataset. If any of the plots show separable clusters, it’s a strong sign that the dataset is suitable for classification.",
         ""),

        (f"* Cluster purity matters more than shape or number of clusters",
        f"If a plot clearly shows the expected number of clusters corresponding to the true number of classes, users should focus on that plot for classification insights. Even if a class splits into multiple clusters in some plots, as long as each cluster maintains high purity (ie, almost all points within a cluster belong to the same class) and is separable from other classes, it is still classifiable and should be fine. The shape of clusters does not always matter—clusters may appear as annular or ring like structures. If they are separable, they are still valid."),
        
        (f"* Overlapping classes indicate potential issues",
        f"If two or more classes consistently overlap in most plots, it might indicate either a dataset issue (e.g., mislabeled data, inadequate features or noise in the data) or a natural similarity between those classes, leading to a higher risk of misclassification."),

        (f"* Possible reasons why a class splits into multiple clusters",
        f"One possible reason is different sampling frequencies within the dataset (e.g., points sampled at 10Hz, 20Hz, and 30Hz may result in three separate clusters per class). However, this is just one possibility—multiple clusters could arise due to other factors like different environment setups or data collection inconsistencies."),

        (f"* Outliers and Noise",
        f"Small, scattered points appearing outside main clusters might be outliers or noise in the data.")
    ]

    # y coordinate
    y_pos = 0.96

    for title, content in notes:

        wrapped_title = textwrap.fill(title, width=max_width_title)
        if "Disclaimer:" in wrapped_title.split(" "):
            fig.text(0.17, y_pos, wrapped_title, disclaimer_title_font, va='top')
            y_pos -= 0.02
            
            fig.text(0.17,y_pos,"Note:-",fontsize=14,fontweight='bold',va='top')
            y_pos -= 0.001
        else:
            fig.text(0.17, y_pos, wrapped_title, title_font, va='top')
            y_pos -= 0.007

        wrapped_content = textwrap.fill(content, width=max_width_content)
        for line in wrapped_content.split('\n'):
            x_pos = 0.17  # Reset x position for each line

            '''
            #Keeping this for loop here just in case if you want to bold certain segments of notes

            bold_segments = re.split(r"(\*\*.*?\*\*)", line)  # Split at bold markers
            
            for segment in bold_segments:
                if segment.startswith("**") and segment.endswith("**"):  # Bold text
                    text = segment.strip("**")
                    fig.text(x_pos, y_pos, text, {**content_font, 'fontweight': 'bold'}, va='top')
                    x_pos += 0.0053 * len(segment)
                else:  # Normal text
                    fig.text(x_pos, y_pos, segment, content_font, va='top')
                    x_pos += 0.0048 * len(segment)  # Adjust x position dynamically
            '''
            fig.text(x_pos, y_pos, line, content_font, va='top')
            y_pos -= 0.005  # Line spacing
        y_pos -= 0.005  # Line spacing
    # Adjust layout
    fig.subplots_adjust(wspace=0.001, top=0.8,left=0.15,right=0.85)

    current_plot = 1 
    # Iterate over all transformations
    for i, transform in enumerate(transforms):
        features, frame_labels, transformed_title, class_names = transform(frame_size, classes_dir, frame_skip, class_names)
        logger.info(f"Processing status: {transformed_title}")

        # Iterate over all scalers
        for j, scaler in enumerate(scalers):
            scaled_features, scaled_title = scaler(features, transformed_title)
            logger.info(f"Processing status: {scaled_title}")

            # Iterate over dimensionality reduction methods
            for k, reducer in enumerate(dim_redn):
                reduced_features, dim_reduced_title = reducer(scaled_features, scaled_title)
                logger.info(f"Processing status: {dim_reduced_title}")

                plot_gof_graph(
                    current_plot,
                    row_num=4,
                    col_num=2,
                    reduced_features=reduced_features,
                    frame_labels=frame_labels,
                    fig=fig,
                    title=dim_reduced_title,
                    class_names=class_names,
                )
                logger.info(f"Subplot {current_plot} created.")
                current_plot += 1 

    # Save the figure to the output directory
    plt.savefig(os.path.join(output_dir, f'GoF_frame_size_{frame_size}.png'))
    logger.info(f"Goodness of Fit graph is at: {os.path.join(output_dir, f'GoF_frame_size_{frame_size}.png')}")
    logger.info("Goodness of Fit test completed.")
    # plt.show()

