import os
import numpy as np
import pandas as pd
import pickle
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from UNet_model import UNet_simp  # Assuming you have the UNet model in a separate file

def load_pickled_data(pickle_file_path):
    """
    Load data from a pickle file.

    Parameters:
    - pickle_file_path: Path to the pickle file.

    Returns:
    - Loaded data.
    """
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_kmeans_model(model_filename):
    """
    Load a pre-trained KMeans model.

    Parameters:
    - model_filename: Filename of the KMeans model.

    Returns:
    - Loaded KMeans model.
    """
    from joblib import load
    kmeans = load(model_filename)
    return kmeans

def encode_image(img, model):
    """
    Encode an image using a pre-trained model.

    Parameters:
    - img: Input image.
    - model: Pre-trained model.

    Returns:
    - Encoded features.
    """
    if torch.cuda.is_available():
        inputs = img.cuda()
        model = model.cuda()
    else:
        inputs = img

    with torch.no_grad():  # Ensuring no gradients are computed
        outputs = model.encoder(inputs)

    vec = outputs[-1]  # This selects the last tensor from the tuple
    # Flatten the selected tensor to 1D and move it to CPU
    vec = vec.view(vec.size(0), -1).cpu().numpy()
    return vec

def get_top_similar_file(new_point, df_cluster, kmeans):
    """
    Get the top similar files based on a new point.

    Parameters:
    - new_point: New data point.
    - df_cluster: DataFrame containing clustered data.
    - kmeans: KMeans model for clustering.

    Returns:
    - Top similar filenames.
    """
    cluster_label = kmeans.predict(new_point)[0]
    same_cluster_df = df_cluster[df_cluster["Cluster"] == cluster_label]
    cluster_encoded_features = np.stack(same_cluster_df['Encoded Features'].values)
    distances = cdist(new_point.reshape(1, -1), cluster_encoded_features, metric='cosine').flatten()
    top_indices = distances.argsort()[0]
    top_info = same_cluster_df.iloc[top_indices]
    top_filenames = top_info["IMAGE_FILE"]
    return top_filenames, top_info

def get_similar_image(input_image, pickle_file_path, kmeans_model_path, unet_model_path,search_directory):
    """
    Find and display similar images to the input image.

    Parameters:
    - input_image: The input image as a tensor
    - pickle_file_path: Path to the pickle file containing encoded features and clustering information.
    - kmeans_model_path: Path to the KMeans model file.
    - unet_model_path: Path to the UNet model file.
    - search_directory: The root directory to search for the image file.

    Returns:
    - Top similar image 
    """
    # Load data
    data = load_pickled_data(pickle_file_path)
    
    # Load KMeans model
    kmeans = load_kmeans_model(kmeans_model_path)
    
    # Load UNet model
    model_unet = UNet_simp(kernel=3, num_filters=32, num_colours=3, num_in_channels=3)
    model_state_dict = torch.load(unet_model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model_unet.load_state_dict(model_state_dict)
    
    # Encode input image
    emb = encode_image(input_image, model_unet)
    
    # Convert encoded features to DataFrame
    df_cluster = pd.DataFrame(data)
    
    # Get top similar image filenames
    top_filename, top_info = get_top_similar_file(emb, df_cluster, kmeans)
    
    for root, dirs, files in os.walk(search_directory):
        if top_filename in files:
            image_path = os.path.join(root, top_filename)
            img = Image.open(image_path)
    return img, top_info['TITLE'], top_info['DESCRIPTION']

""" output = get_similar_image(input_image = input_tensor, 
                           pickle_file_path='./outputs/encoded_features_desc_wCluster.pkl', 
                           kmeans_model_path='./outputs/kmeans_model.joblib', 
                           unet_model_path='./outputs/UNet_SEMART/model_epoch_2.pth',
                           search_directory="E:\\SemArt\\Images") """

