# Webapp
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import joblib
import rec as rc

# Autoencoder for similar images



# Define a random model for testing purposes
class RandomModel(nn.Module):
    def __init__(self, num_classes):
        super(RandomModel, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(224 * 224 * 3, num_classes)  # Assuming input size is 224x224 and 3 channels

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define a list of genre labels for testing purposes
styles = ['Realism', 'Baroque', 'Post_Impressionism', 'Impressionism',
       'Romanticism', 'Art_Nouveau', 'Northern_Renaissance', 'Symbolism',
       'Naive_Art_Primitivism', 'Expressionism', 'Cubism', 'Fauvism',
       'Analytical_Cubism', 'Abstract_Expressionism', 'Synthetic_Cubism',
       'Pointillism', 'Early_Renaissance', 'Color_Field_Painting',
       'New_Realism', 'Ukiyo_e', 'Rococo', 'High_Renaissance',
       'Mannerism_Late_Renaissance', 'Pop_Art', 'Contemporary_Realism',
       'Minimalism', 'Action_painting']
genres = ['landscape', 'religious_painting', 'portrait', 'genre_painting',
       'Unknown Genre', 'still_life', 'sketch_and_study', 'illustration',
       'cityscape', 'nude_painting', 'abstract_painting']

# Define the custom function to preprocess the image and get predictions
def preprocess_image(image):
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

def get_model(path):
    best_model = joblib.load('model\\resnet50_wo_l2_epoch3.joblib')
    return best_model

def captioner(image, style):
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

def predict(image):
    # Preprocess the image
    input_tensor = preprocess_image(image['composite'])

    # Initialize the random model
    model_style = RandomModel(num_classes=len(styles))
    model_genre = RandomModel(num_classes=len(genres))
    # Get prediction from model
    with torch.no_grad():
        model_style.eval()
        model_genre.eval()
        outputs_style = model_style(input_tensor)
        outputs_genre = model_style(input_tensor)
        probabilities_style = F.softmax(outputs_style, dim=1)
        probabilities_genre = F.softmax(outputs_genre, dim=1)
        #_, predicted = torch.max(outputs, 1)
         # Create a dictionary to store class probabilities
        style_probabilities = {styles[i]: probabilities_style[0][i].item() for i in range(len(styles))}
        genre_probabilities = {genres[i]: probabilities_genre[0][i].item() for i in range(len(genres))}
        #most_likely_style = max(style_probabilities, key=style_probabilities.get)
        #most_likely_genre = max(genre_probabilities, key=genre_probabilities.get)
    sim_img, title = rc.get_similar_image(input_image = input_tensor, 
                           pickle_file_path='./outputs/encoded_features_desc_w_Cluster_unique.pkl', 
                           kmeans_model_path='./outputs/kmeans_model.joblib', 
                           unet_model_path='./outputs/UNet_SEMART/model_epoch_2.pth',
                           search_directory="E:\\SemArt\\Images")
    
    return style_probabilities, genre_probabilities, sim_img, title

# Flagging (Feedback)
callback = gr.CSVLogger()

# Add feedback for style and genre and allow user to correct (dropdown)
# Add rating for the captions between 1-10
# Export all to .csv
# In saved inputs, just take the background.png for making a new dataset based on user feedback

with gr.Blocks(theme=gr.themes.Soft(), title="AI Art Curation System") as demo:
    gr.Markdown("""
                # AI Art Curation System

                Upload an image (and select ROI to begin)
                """)
    with gr.Tab("Model"): # Main tab
        with gr.Row():
            with gr.Column():
                input_image = gr.ImageEditor(type='pil', image_mode='RGB', transforms='crop', eraser=False, brush=False,scale=2, label="Input")
                btn = gr.Button("Generate")
                reset_btn = gr.Button("Reset")
            with gr.Column():
                output_style = gr.Label(label="Predicted Style",num_top_classes=3,scale=0)
                output_genre = gr.Label(label="Predicted Genre",num_top_classes=3,scale=0)
                output_text = gr.Textbox(label="Suggested Artwork Title",scale=0,show_copy_button=True,max_lines=2)
            with gr.Column():
                output_image = gr.Image(label="Recommended Artwork",scale=0, type="pil", interactive=False,height=512, width=512)
                #output_caption = gr.Text(label="Generated Caption",scale=0,show_copy_button=True)
        btn.click(predict, inputs=input_image, outputs=[output_style, output_genre, output_image, output_text])
        reset_btn.click(inputs=[])
   
    with gr.Tab("Feedback"): # Feedback tab
        with gr.Column():
            with gr.Row():
                style_rating = gr.Dropdown(styles+['None'], label="Corrected Style")
                genre_rating = gr.Dropdown(genres+['None'], label="Corrected Genre")
            image_rating = gr.Slider(1, 10, step = 1, value=5, label="Recommendation Rating", info="Rate the Recommended Artwork Between 1 to 10 (Best)")
            #caption_rating = gr.Slider(1, 10, step = 1, value=5, label="Caption Rating", info="Rate the Generated Caption Between 1 to 10 (Best)")
            btn_feedback = gr.Button("Flag Feedback")
    flagged_content = [input_image, output_style, output_genre, output_image, style_rating, genre_rating, image_rating]
    callback.setup(flagged_content, flagging_dir="flagged_data") # This needs to be called at some point prior to the first call to callback.flag()
    btn_feedback.click(lambda *args: callback.flag(args), inputs=flagged_content, outputs=None, preprocess=False)


if __name__ == "__main__":
    demo.launch(share=False)
    #demo.launch(auth=('admin','MIE1517'),share=True)