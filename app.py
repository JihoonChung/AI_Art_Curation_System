# Webapp
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

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

def predict_genre(image):
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
        caption = 'test caption'

    return class_probabilities, class_probabilities, caption


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Upload an image and select ROI")
    with gr.Row():
        with gr.Column():
            input_image = gr.ImageEditor(type='pil', image_mode='RGB', transforms='crop', eraser=False, brush=False,scale=2)
            btn = gr.Button("Generate")
        with gr.Column():
            output_style = gr.Label(label="Predicted Style",num_top_classes=3,scale=0)
            output_genre = gr.Label(label="Predicted Genre",num_top_classes=3,scale=0)
        with gr.Column():
            output_caption = gr.Text(label="Generated Caption",scale=0)
    btn.click(predict_genre, inputs=input_image, outputs=[output_style, output_genre, output_caption])
    

if __name__ == "__main__":
    demo.launch()