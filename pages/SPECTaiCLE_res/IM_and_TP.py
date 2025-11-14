import time
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
import cv2
import os
import re
from ultralytics import YOLO
import concurrent.futures
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model_v9i.pt')
model = YOLO(model_path)

"""
Image Maniplation and Text Preprocessing Functions for SPECTaiCLE
"""

def flip_black_white_fast(image):
    flip_black_white_faststart_time = time.time()
    image_array = np.array(image)
    
    # Get the total number of pixels in the image
    total_pixels = image_array.size  # Height * Width

    # Set a pixel count threshold (50% of total pixels)
    white_threshold = total_pixels // 2

    # Count white pixels and stop early if they exceed the threshold
    white_pixels = np.sum(image_array > 128)

    # If more than 50% of the pixels are white, invert the image
    if white_pixels > white_threshold:
        inverted_image = ImageOps.invert(image)
        flip_black_white_fastend_time = time.time()
        print(f"flip_black_white_fast took {flip_black_white_fastend_time - flip_black_white_faststart_time:.2f} seconds - INVERSION")
        return inverted_image
    else:
        # Return the original image if white pixels are not dominant
        flip_black_white_fastend_time = time.time()
        print(f"flip_black_white_fast took {flip_black_white_fastend_time - flip_black_white_faststart_time:.2f} seconds - INVERSION")
        return image
    

def rescaleImage(image_path = '', min_size=2000, max_size = 3000, upscale_factor=2, downscale_factor=0.75):
    # Load the image
    original_image = Image.open(image_path)

    # Get original image size
    original_width, original_height = original_image.size
    #print(f"Original size of {image_path}: {original_width}x{original_height}")

    # Calculate new dimensions while maintaining the aspect ratio
    new_width, new_height = original_width, original_height

    # Decide whether to upscale or downscale
    if min(original_width, original_height) < min_size:
        # If the image is too small, upscale
        scale_factor = upscale_factor
        while min(new_width, new_height) < min_size:
            new_width = int(new_width * scale_factor)
            new_height = int(new_height * scale_factor)
            
            # Avoid overly large upscaling
            if max(new_width, new_height) > max_size:
                #print(f"Image is too large after upscaling, adjusting max size.")
                new_width = min(new_width, max_size)
                new_height = min(new_height, max_size)
                break
    else:
        # If the image exceeds the max size, downscale it
        if max(new_width, new_height) > max_size:
            scale_factor = downscale_factor
            while max(new_width, new_height) > max_size:
                new_width = int(new_width * scale_factor)
                new_height = int(new_height * scale_factor)

    # Calculate the scaling factor to maintain aspect ratio
    aspect_ratio = original_width / original_height

    # Maintain the aspect ratio
    if new_width > new_height:
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = int(new_height * aspect_ratio)

    # Print new size after scaling
    #print(f"Rescaled size: {new_width}x{new_height}")

    # Rescale image using LANCZOS resampling for better quality
    image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def correctImage(images, save_dir=None):
    correctImagestart_time = time.time()
    '''
    Corrects the input images and saves rotated copies.
    Args:
    - images: List of image file paths (str)
    - save_dir: Directory to save rotated images. If None, saves in the same directory as the original images.
    '''
    corrected_images = []
    print(images)

    # Load the image
    for image_path in images:

        # Upscales the Image
        image = rescaleImage(image_path)

        # Enhance contrast
        image = ImageEnhance.Contrast(image)
        image = image.enhance(1.218)  # Increase contrast by a factor of 2

        # Denoise using median filter
        #image = image.filter(ImageFilter.MedianFilter(size=3))
    
        #Convert the Image to grayscale
        image = image.convert('L')

        # Invert if white predominates in the image
        image = flip_black_white_fast(image)

        #Black and White Image
        image = np.array(image)
        image = cv2.medianBlur(image, 3)




        image = Image.fromarray(image)

        # Ensure save_dir exists or use original image directory
        if save_dir is None:
            save_dir = os.path.dirname(image_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get save path
        corrected_image_path = os.path.join(save_dir, os.path.basename(image_path))
        # Save the corrected image (optional)
        image.save(corrected_image_path)



        # Images to return
        corrected_images.append(corrected_image_path)

    
    correctImageend_time = time.time()
    print(f"correctImage took {correctImageend_time - correctImagestart_time:.2f} seconds")
    return corrected_images

def process_folder(folder_path):
    process_folderstart_time = time.time()
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        process_folderend_time = time.time()
        print(f"process_folder took {process_folderend_time - process_folderstart_time:.2f} seconds")
        return []
    
    # List all files in the folder (excluding subdirectories)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if len(files) == 0:
        print(f"No files found in {folder_path}.")
    else:
        print(f"Number of files: {len(files)}")
    
    process_folderend_time = time.time()
    print(f"process_folder took {process_folderend_time - process_folderstart_time:.2f} seconds")
    return files

# Set the path to the Tesseract executable (if not in PATH)
#pytesseract.pytesseract.tesseract_cmd = r"c:\Users\arado\Desktop\Full_Stack_Final_Project\tesseract ocr\tesseract.exe"

def detect_text(image_path, oem=1, psm=3):
    detect_textstart_time = time.time()
    """
    Detects text in the image file using Tesseract's LSTM engine.
    Args:
    - image_path: Path to the image file.
    - oem: OCR Engine Mode (default: 1 for LSTM).
    - psm: Page Segmentation Mode (default: 3 for automatic segmentation).

    OEM Modes:
    0: Original Tesseract OCR Engine.
    1: Neural net LSTM engine (most efficient and accurate for most cases).
    2: Combined mode (use both).
    3: Default (lets Tesseract decide).

    PSM (Page Segmentation Mode): This determines how Tesseract splits text. For example:
    3: Fully automatic page segmentation (most common).
    6: Assume a single uniform block of text.
    7: Treat the image as a single text line (good for single spine).
    """
    
    # Open the image file
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        detect_textend_time = time.time()
        print(f"detect_text took {detect_textend_time - detect_textstart_time:.2f} seconds")   
        return None

    # Use pytesseract to extract text with specified engine mode and page segmentation mode
    custom_config = f'--oem {oem} --psm {psm}'
    detected_text = (image_path, pytesseract.image_to_string(image, config=custom_config))


    detect_textend_time = time.time()
    print(f"detect_text took {detect_textend_time - detect_textstart_time:.2f} seconds")   

    return detected_text

def rename_results_with_boundingBox(results, source, text_prediction_mode = False):

    # Regex to extract the file name
    pattern = r"[^/]+$"
    match_filename = re.search(pattern, source)
    file_name = None
    if match_filename:
        file_name = match_filename.group(0)
        print(file_name)  # Output: home-library-tour-v0-wk0mm30medka.jpg

    # Regex to extract the file type
    pattern = r"\.\w+$"
    match_filetype = re.search(pattern, file_name)
    file_type = None
    if match_filetype:
        file_type = match_filetype.group(0)
        print(file_type)  # Output: .jpg, .png, etc


    #remove the file type from the end of the filename string
    file_name = file_name.removesuffix(file_type)
    


    for i, box in enumerate(results[0].boxes.xyxy):  # Access bounding boxes
        x_min, y_min, x_max, y_max = map(int, box)  # Convert to 

        num_predictions_total = len(os.listdir("pages/SPECTaiCLE_res/Predict/"))

        cropped_images_dir = f"pages/SPECTaiCLE_res/Predict/predict{num_predictions_total}/crops/Book"

        # Construct the new file name
        if i == 0:
            original_file = os.path.join(cropped_images_dir, f"{file_name}{file_type}")  # Existing name
            new_file_name = f"{file_name}_0_{x_min}_{y_min}_{x_max}_{y_max}{file_type}"  # New name
            new_file_path = os.path.join(cropped_images_dir, new_file_name)
        else:
            original_file = os.path.join(cropped_images_dir, f"{file_name}{i+1}{file_type}")  # Existing name
            new_file_name = f"{file_name}_{i}_{x_min}_{y_min}_{x_max}_{y_max}{file_type}"  # New name
            new_file_path = os.path.join(cropped_images_dir, new_file_name)
        
        # Rename the file
        if os.path.exists(original_file):  # Ensure the file exists
            if text_prediction_mode == True:
                os.rename(original_file, new_file_path)
            else:
                os.rename(original_file, new_file_path)
                print(f"Renamed: {original_file} -> {new_file_path}")
        else:
            print(f"File not found: {original_file}")

def get_spines(source, save=False):
    get_spinesstart_time = time.time()
    results = model.predict(source = source, #Location of item to predict {'filepath: such as .mp4 videos, or .jpg photos', '0 for webcam', 'local hosting ip adresses'}
                    save=save, #Save to the local prediction folder ( runs/detect/predict  )
                    conf = 0.1, #Threshold for confidence, best value obtained from f1 curve is 0.456, higher conf better for webcams/video ~0.80+
                    line_width = 2, #Text and Line Thickness
                    save_crop=save, #If set to TRUE It will automaticlly crop out the bounding box of predictions and save them to ( runs/detect/predict/crops )
                    save_txt=save, #Saves annotations in YOLO format to ( runs/detect/predict/labels )
                    project = "pages/SPECTaiCLE_res/Predict/", #Project Directory
                )
    
    rename_results_with_boundingBox(results=results, source=source)

    get_spinesend_time = time.time()
    print(f"save_frame took {get_spinesend_time - get_spinesstart_time:.2f} seconds")   

def read_photos(clear_input_folder = True, correctImages=False, folder_path = "pages/SPECTaiCLE_res/Input/"):
    read_photosstart_time = time.time()
    # Specify your folder path here
    files = process_folder(folder_path)
    
    # Correct the images before OCR processing (if correctImages flag is set)
    if correctImages:
        corrected_images = correctImage([os.path.join(folder_path, picture) for picture in files])
    else:
        corrected_images = [os.path.join(folder_path, picture) for picture in files]

    # Tesseract OCR Call
    # Use ThreadPoolExecutor to parallelize the OCR process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results_tesseract = []
        results_tesseract = list(executor.map(lambda img_path: detect_text(image_path=img_path), corrected_images))


    if clear_input_folder:
        # Remove the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder {folder_path} and its contents have been deleted.")


    read_photosend_time = time.time()
    print(f"save_frame took {read_photosend_time - read_photosstart_time:.2f} seconds")   
    return results_tesseract



###
#  Book Bounding Box Extraction
###
all_predicted_book_spines = []
def get_books(correctImages=True, cullNullTextImages = False, image_file_paths = []):
    # YOLO Model Inference   
    for item in image_file_paths:
        # Perform inference
        get_spines(item, save=True)     

    # Process each image independently
    books = read_photos(clear_input_folder=False, correctImages=correctImages)

    if books:  # Make sure books is not empty
        if cullNullTextImages:
            #Keep only books where book[1] is not an empty string
            books = [book for book in books if book[1] != '']
           
        return books
    else:
        print('No Books Detected')
        return None

def compare_user_input(user_input, book_spines=[], heatmap=False, printEverything = True, showBook = False):
    # 1. Use TF-IDF Vectorizer to compute the embedding of the user input
    vectorizer = TfidfVectorizer()
    
    # Combine user_input with all predicted book spines for fitting the vectorizer
    texts = [user_input] + [pred_text[1] for book_spine in book_spines for pred_text in book_spine]

    # Fit and transform the texts to get the embeddings
    embeddings = vectorizer.fit_transform(texts).toarray()

    # User input embedding is the first embedding in the array
    user_input_embedding = embeddings[0]

    # 2. Process each bookshelf's predicted texts and compare with user input
    for idx, book_shelf in enumerate(book_spines):

        # 3. Extract the embeddings of the predicted bookshelf texts
        shelf_embeddings = embeddings[1 + sum(len(shelf) for shelf in book_spines[:idx]): 1 + sum(len(shelf) for shelf in book_spines[:idx + 1])]

        # 4. Calculate similarity between user input and bookshelf embeddings
        similarities = cosine_similarity([user_input_embedding], shelf_embeddings)
                

        # 5. Display the similarity results
        if printEverything:
            print(f"Comparing with Book Shelf {idx + 1}:")

            for i, text in enumerate(book_shelf):
                print(f'\n___________________________ \n{i + 1}')
                print(f"Predicted Text: {text}")
                print(f"Similarity with '{user_input}': {similarities[i]:.4f}\n")

        highest_score_index = similarities.argmax()
        print(f'Best Match for {user_input}: Score of : {similarities[0][highest_score_index]:.4f}, was - {book_spines[idx][highest_score_index]}')

        if heatmap:
            #visualize the similarities using a heatmap
            plt.figure(figsize=(5, 5))
            sns.heatmap(similarities.reshape(-1, 1), annot=False, cmap='rocket', robust=True)
            plt.title(f'Similarity Heatmap for Book Shelf {idx + 1}')
            plt.xticks([0], [user_input], rotation=45, ha='right')
            plt.show()


        if showBook:
            if similarities[0][highest_score_index] > 0:
                # Open the image
                filepath = book_spines[idx][highest_score_index][0]

                # Extract the file name and file type using a single regex
                match = re.search(r"([^\\]+)(\.\w+)$", filepath)
                if match:
                    file_name, file_type = match.groups()
                    print(file_name)  # e.g., home-library-tour-v0-wk0mm30medka.jpg
                    print(file_type)  # e.g., .jpg

                # Remove file type to extract the base name
                base_name = file_name.removesuffix(file_type)

                # Extract the main part of the name and coordinates in one step
                match = re.match(r"([^_]+)_(.+)", base_name)
                if match:
                    result, coordinates_str = match.groups()

                # Extract numbers and form the coordinates tuple
                coords = tuple(map(int, re.findall(r"\d+", coordinates_str)[1:]))
                print(coords)  # e.g., (1355, 1031, 1952, 1220)

                # Construct the image path
                image_path = f"pages/SPECTaiCLE_res/Predict/{result}{file_type}"

                # Open the image
                image = Image.open(image_path)

                # Draw the red box
                draw = ImageDraw.Draw(image)
                draw.rectangle(coords, outline='red', width = 2)
                

                # Define the text and its position
                text = user_input + ' - Score : 'f'{similarities[0][highest_score_index]:.4f}'

                try:
                    font = ImageFont.truetype("arial.ttf", size=20)  # Use a custom font
                except IOError:
                    font = ImageFont.load_default()  # Fallback to default font if not found

                # Calculate the text size
                text_bbox = font.getbbox(text)  # Get the bounding box of the text (left, top, right, bottom)
                text_width = text_bbox[2] - text_bbox[0]  # Calculate text width
                text_height = text_bbox[3] - text_bbox[1]  # Calculate text height

                # Define the rectangle coordinates
                padding = 5
                background_coords = (
                    coords[2]- text_width - padding, 
                    coords[3]- text_height - padding, 
                    coords[2],
                    coords[3] 
                )

                # Draw the solid background
                background_color = (255, 255, 255)  # White background
                draw.rectangle(background_coords, fill=background_color)

                # Draw the text over the rectangle
                text_color = (0, 0, 0)  # White text
                text_position = (coords[2]- text_width - padding, coords[3]- text_height - padding)  # Center text in the rectangle
                draw.text(text_position, text, fill=text_color, font=font)
                # Display the image
                roi = image.crop(coords)
                bw_image=image.convert('L').convert('RGB')
                # Paste the ROI back into the grayscale image
                bw_image.paste(roi, coords)
                
                bw_image.show()



### all_predicted_book_spines = [get_books(correctImages=True, cullNullTextImages=False, image_file_paths=image_file_paths)]
