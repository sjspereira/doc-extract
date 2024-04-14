import fitz
import pytesseract
from PIL import Image
import os
import io
import sys

args = sys.argv

textExtracted = ''

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text().strip()
            text += page_text + "\n\n"
    return text

def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image.save(f"{output_folder}/page{page_number + 1}_image{image_index + 1}.png")

def perform_ocr_on_images(image_folder):
    text = ''
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path)
            text += (pytesseract.image_to_string(img) + "/n")
    return text

# Usage
arguments = args[1:]

if len(arguments) >= 1:
    filename = arguments[0]
else:
    filename = "test.pdf"

pdf_path = filename
image_folder = "/images"
output_folder = "output"

extract_images_from_pdf(pdf_path, output_folder + image_folder)

textExtracted = extract_text_from_pdf(pdf_path)

textExtracted += perform_ocr_on_images(output_folder + image_folder)

with open(f"{output_folder}/texts/extracted_text.txt", "w") as txt_file:
    txt_file.write(textExtracted)
