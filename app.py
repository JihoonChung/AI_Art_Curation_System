# Webapp
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

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
classes = ['Realism', 'Baroque', 'Post_Impressionism', 'Impressionism',
       'Romanticism', 'Art_Nouveau', 'Northern_Renaissance', 'Symbolism',
       'Naive_Art_Primitivism', 'Expressionism', 'Cubism', 'Fauvism',
       'Analytical_Cubism', 'Abstract_Expressionism', 'Synthetic_Cubism',
       'Pointillism', 'Early_Renaissance', 'Color_Field_Painting',
       'New_Realism', 'Ukiyo_e', 'Rococo', 'High_Renaissance',
       'Mannerism_Late_Renaissance', 'Pop_Art', 'Contemporary_Realism',
       'Minimalism', 'Action_painting']

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

def captioner(image, style):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    # conditional image captioning
    text = f"{style} artwork of "
    image = transforms.ToPILImage()(image.squeeze())
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def predict(image):
    # Preprocess the image
    input_tensor = preprocess_image(image['layers'][0])

    # Initialize the random model
    model = RandomModel(num_classes=len(classes))

    # Get prediction from model
    with torch.no_grad():
        model.eval()
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        #_, predicted = torch.max(outputs, 1)
         # Create a dictionary to store class probabilities
        class_probabilities = {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
        most_likely_style = max(class_probabilities, key=class_probabilities.get)
        caption = captioner(input_tensor, most_likely_style)

    return class_probabilities, class_probabilities, caption

# Flagging (Feedback)
callback = gr.CSVLogger()

# Add feedback for style and genre and allow user to correct (dropdown)
# Add rating for the captions between 1-10
# Export all to .csv

with gr.Blocks(theme=gr.themes.Soft(), title="AI Art Curation System") as demo:
    gr.Markdown("""
                # AI Art Curation System

                Upload an image (and select ROI to begin)
                """)
    with gr.Tab("Model"): # Main tab
        with gr.Row():
            with gr.Column():
                input_image = gr.ImageEditor(type='pil', image_mode='RGB', transforms='crop', eraser=False, brush=False,scale=2)
                btn = gr.Button("Generate")
            with gr.Column():
                output_style = gr.Label(label="Predicted Style",num_top_classes=3,scale=0)
                output_genre = gr.Label(label="Predicted Genre",num_top_classes=3,scale=0)
            with gr.Column():
                output_caption = gr.Text(label="Generated Caption",scale=0)
        btn.click(predict, inputs=input_image, outputs=[output_style, output_genre, output_caption])
    # This needs to be called at some point prior to the first call to callback.flag()
    with gr.Tab("Feedback"): # Feedback tab
        with gr.Column():
            with gr.Row():
                style_rating = gr.Dropdown(["test1", "test2", "test3"], label="Corrected Style")
                genre_rating = gr.Dropdown(["test1", "test2", "test3"], label="Corrected Genre")
            caption_rating = gr.Slider(1, 10, step = 1, value=5, label="Caption Rating", info="Rate the Generated Caption Between 1 to 10 (Best)")
            btn_feedback = gr.Button("Flag Feedback")
    callback.setup([input_image, style_rating, genre_rating, caption_rating], "flagged_data_points")        
    btn_feedback.click(lambda *args: callback.flag(args), [input_image, style_rating, genre_rating, caption_rating], None, preprocess=False)

    

if __name__ == "__main__":
    demo.launch()