from PIL import Image
import os
import zipfile
import pandas as pd
import numpy as np

def calculate_average_color(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img_array = np.array(img)
    average_color = np.mean(img_array, axis=(0, 1))
    return tuple(map(int, average_color))

def process_images(zip_ref, style, df):
    data = []
    for file_info in zip_ref.infolist():
        # Extract the file name
        filename = file_info.filename

        # Check if it is a directory (artist folder)
        if '/' not in filename and filename.lower().endswith('.jpg'):
            with zip_ref.open(file_info) as file:
                # Extract the artist information from the file path
                artist = os.path.basename(filename)
                image_path = os.path.join(style, artist, filename)
                average_color = calculate_average_color(file)

                # Append the result to the data list
                data.append({'Style': style, 'Artist': artist, 'Image': image_path, 'AverageColor': average_color})

    # Concatenate the data list to the DataFrame
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    return df

def process_style_zip(style_zip_path, df):
    # Extract style information from the style zip file
    style = os.path.basename(style_zip_path).split('.')[0]

    with zipfile.ZipFile(style_zip_path, 'r') as style_zip_ref:
        # Process each artist folder in the style zip
        for file_info in style_zip_ref.infolist():
            if '/' in file_info.filename and file_info.filename.lower().endswith('.jpg'):
                with style_zip_ref.open(file_info) as file:
                    # Extract the artist information from the file path
                    artist = os.path.basename(file_info.filename)
                    image_path = os.path.join(style, artist, file_info.filename)
                    average_color = calculate_average_color(file)

                    # Append the result to the DataFrame
                    data = {'Style': style, 'Artist': artist, 'Image': image_path, 'AverageColor': average_color}
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    return df

# Specify the directory where your style zip archives are located
style_zip_directory = 'data\\'

# Create an empty DataFrame
result_df = pd.DataFrame(columns=['Style', 'Artist', 'Image', 'AverageColor'])

# Iterate through style zip archives in the specified directory
for style_zip_filename in os.listdir(style_zip_directory):
    # Join the directory and style zip file name to get the full path
    style_zip_full_path = os.path.join(style_zip_directory, style_zip_filename)

    # Check if it is a file and ends with ".zip"
    if os.path.isfile(style_zip_full_path) and style_zip_filename.lower().endswith('.zip'):
        # Process each style zip file
        result_df = process_style_zip(style_zip_full_path, result_df)

# Save the resulting DataFrame to a CSV file
result_df.to_csv('color-analysis.csv', index=False)