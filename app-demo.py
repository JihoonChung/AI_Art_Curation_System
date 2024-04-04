# Webapp for Art Curation Project

# Import gradio for webapp
import gradio as gr
# Import Art Curation Model
from art_curation import *

# Module Dependencies #
# art_curation.py
##   style.py
##   rec.py
###      UNet_model.py

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
            with gr.Column():
                output_style = gr.Label(label="Predicted Style",num_top_classes=3,scale=0)
                output_genre = gr.Label(label="Predicted Genre",num_top_classes=3,scale=0)
            with gr.Column():
                output_image = gr.Image(label="Recommended Artwork",scale=0, type="pil", interactive=False,height=512, width=512)
                output_text = gr.Textbox(label="Suggested Artwork Title",scale=0,show_copy_button=True,max_lines=2)
                output_description = gr.Textbox(label="Suggested Artwork Description",scale=0,show_copy_button=True)
                #output_caption = gr.Text(label="Generated Caption",scale=0,show_copy_button=True)
        btn.click(predict, inputs=input_image, outputs=[output_style, output_genre, output_image, output_text, output_description])
   
    with gr.Tab("Feedback"): # Feedback tab
        with gr.Column():
            with gr.Row():
                styles = read_labels('styles.txt')
                genres = read_labels('genres.txt')
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
    #demo.launch(share=True)