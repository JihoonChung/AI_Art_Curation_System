# Art_Curation Module
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from joblib import load
import rec as rc #rec.py autoencoder
from style import * #style CNN

def preprocess_image(image):
    """
    Preprocesses an input image for use in the models.
    1. Adds padding to make aspect ratio 1:1
    2. Resizes image to 224x224 (ResNet)
    3. Normalizes to the mean/std of ResNet
    
    Parameters:
    image (PIL.Image): Input image to be preprocessed.
    
    Returns:
    torch.Tensor: Preprocessed image tensor suitable for model input.
    """
    # Open image using PIL
    # Add zero padding to make the aspect ratio 1:1
    width, height = image.size
    max_dim = max(width, height)
    new_size = (max_dim, max_dim)
    padded_image = Image.new("RGB", new_size)
    padded_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))

    # Resize the image to 224x224
    resized_image = padded_image.resize((224, 224))

    # Convert PIL image to PyTorch tensor
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(resized_image).unsqueeze(0)

    return input_tensor

# Get labels for style/genre from .txt file
def read_labels(filename):
    """
    Reads labels from a text file.
    
    Parameters:
    filename (str): The path to the text file containing labels.
    
    Returns:
    list: List of labels read from the text file.
    """
    with open(filename, 'r') as f:
        lst = [line.strip() for line in f]
    return lst

def get_model(path):
    """
    Loads a pre-trained model from the specified path.
    Specifically used only for the style model.
    
    Parameters:
    path (str): The path to the pre-trained model file.
    
    Returns:
    Any: The loaded pre-trained model.
    """
    best_model = load(path)
    return best_model

def get_genre_model(path):
    """
    Loads a pre-trained ResNet model with modified top layer for genre classification.
    
    Parameters:
    path (str): The path to the pre-trained ResNet model file.
    
    Returns:
    torch.nn.Module: The loaded pre-trained ResNet model for genre classification.
    """
    # Load pre-trained ResNet model
    resnet = models.resnet101(pretrained=True)
    resnet.name = 'ResNet101'

    # Freeze parameters in ResNet architecture
    for param in resnet.parameters():
        param.requires_grad = False

    # Modify the top layer
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 66),
        nn.ReLU(),
        nn.Linear(66, 11))
    state = torch.load(path)
    resnet.load_state_dict(state)
    return resnet

def captioner(image, style):
    """
    Generates a caption for an input image based on its style using Blip.
    
    Parameters:
    image (torch.Tensor): Input image tensor.
    style (str): Style of the image.
    
    Returns:
    str: Generated caption for the image.
    """
    # Load BLIP
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    # Conditional image captioning
    text = f"{style} artwork of "
    #image = transforms.ToPILImage()(image.squeeze()) #if input is pytorch tensor
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def predict(image, app = True):
    """
    Predicts the style and genre of an input image and provides recommendations based on similarity.
    
    Parameters:
    image ((PIL.Image | EditorValue): Input image to be analyzed.
    app (bool): Flag indicating whether the image is coming from an app environment.
    
    Returns:
    tuple: A tuple containing dictionaries of style and genre probabilities, 
           recommended image, title, and description.
    """
    # Preprocess the image
    if app == True:
        input_tensor = preprocess_image(image['composite']) # Access image editor object
    else:
        input_tensor = preprocess_image(image)
    #############################################################
    # Style and Genre CNN
    #############################################################
    # Get labels
    styles = read_labels('./labels/styles.txt')
    genres = read_labels('./labels/genres.txt')
    # Get models
    model_style = get_model('./models/model_50_2_batch_128_p2_epoch9.joblib')
    model_genre = get_genre_model('./models/Genre_classifier_model.001_epoch8')
    # Use GPU so it runs faster for demo, change to CPU for huggingface
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_style = model_style.to(device)
    model_genre = model_genre.to(device)
    input_tensor = input_tensor.to(device)
    # Get prediction from model
    with torch.no_grad():
        model_style.eval()
        model_genre.eval()
        outputs_style = model_style(input_tensor)
        outputs_genre = model_genre(input_tensor)
        # Get probabilities of each label
        probabilities_style = torch.softmax(outputs_style, dim=1)
        probabilities_genre = torch.softmax(outputs_genre, dim=1)
        #most_likely_style = max(probabilities_style, key=probabilities_style.get)
        # Map labels to probabilities
        style_probabilities = {styles[i]: probabilities_style[0][i].item() for i in range(len(styles))}
        genre_probabilities = {genres[i]: probabilities_genre[0][i].item() for i in range(len(genres))}
    #############################################################
    # Recommendation
    #############################################################
    sim_img, title, description = rc.get_similar_image(input_image = input_tensor, 
                           pickle_file_path='./models/encoded_features_desc_w_Cluster_unique.pkl', 
                           kmeans_model_path='./models/kmeans_model.joblib', 
                           unet_model_path='./models/UNet_SEMART/model_epoch_2.pth',
                           search_directory="./SemArt/Images") # Change to wherever the SemART dataset is saved
    
    #caption = captioner(input_tensor, most_likely_style) #BLIP Captions

    return style_probabilities, genre_probabilities, sim_img, title, description