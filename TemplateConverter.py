import cv2
import numpy as np
import os
import shutil

def extract_photos(input_image, aspect_ratio=(5.5, 8.5), min_percentage=0.1):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_photos = []
    total_pixels = input_image.shape[1] * input_image.shape[0]
    min_size = np.sqrt(min_percentage / 100.0 * total_pixels)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / h

        if (aspect_ratio[0] / aspect_ratio[1] * 0.8 <= ratio <= aspect_ratio[0] / aspect_ratio[1] * 1.2 
            and w >= min_size and h >= min_size):
            photo = input_image[y:y+h, x:x+w]
            extracted_photos.append(photo)

    return extracted_photos


def save_photos(photos, idol_name, base_filename='photocard'):
    script_directory = os.path.dirname(os.path.abspath(__file__))

    idols_folder = os.path.join(script_directory, 'Photocards')
    idol_directory = os.path.join(idols_folder, idol_name)

    if os.path.exists(idol_directory):
        shutil.rmtree(idol_directory)

    os.makedirs(idol_directory)

    for i, photo in enumerate(photos):
        filename = os.path.join(idol_directory, f'{base_filename}_{i+1}.jpg')
        cv2.imwrite(filename, photo, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

def process_templates():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    templates_folder = os.path.join(script_directory, 'Templates')

    for subdir, _, files in os.walk(templates_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)

                photocards = extract_photos(image)

                idol_name = os.path.splitext(file)[0]
                save_photos(photocards, idol_name)

                print(f"{idol_name}: {len(photocards)}")

process_templates()