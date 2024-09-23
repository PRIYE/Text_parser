import PyPDF2
import cv2
import pytesseract
import numpy as np
from PIL import Image
import json
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def perform_ocr_with_bounding_boxes(image):
    # Perform OCR to get data including bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return ocr_data

def detect_element_type(text, font_size, is_bold):
    # Simple heuristics to detect element type
    if font_size > 20 and is_bold:
        return "heading"
    elif font_size > 16 and is_bold:
        return "subheading"
    elif text.strip().startswith(('•', '-', '*')):
        return "list_item"
    else:
        return "paragraph"

def is_table(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    return lines is not None and len(lines) > 5  # Arbitrary threshold

def process_page(image):
    ocr_data = perform_ocr_with_bounding_boxes(image)
    elements = []
    current_element = {"type": None, "content": "", "bbox": None}

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i]
        if not text.strip():
            continue

        bbox = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
        font_size = ocr_data['height'][i]  # Approximating font size with height
        is_bold = ocr_data['font'][i] > 0  # Tesseract represents bold with positive font values

        element_type = detect_element_type(text, font_size, is_bold)

        if is_table(image, bbox):
            element_type = "table"

        if element_type != current_element["type"]:
            if current_element["type"]:
                elements.append(current_element)
            current_element = {"type": element_type, "content": text, "bbox": bbox}
        else:
            current_element["content"] += " " + text
            current_element["bbox"] = (
                min(current_element["bbox"][0], bbox[0]),
                min(current_element["bbox"][1], bbox[1]),
                max(current_element["bbox"][0] + current_element["bbox"][2], bbox[0] + bbox[2]) - min(current_element["bbox"][0], bbox[0]),
                max(current_element["bbox"][1] + current_element["bbox"][3], bbox[1] + bbox[3]) - min(current_element["bbox"][1], bbox[1])
            )

    if current_element["type"]:
        elements.append(current_element)

    # Sort elements by their vertical position (top coordinate of bbox)
    elements.sort(key=lambda x: x['bbox'][1])

    return elements

def detect_images(image):
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_elements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 100:  # Adjust these thresholds as needed
            image_elements.append({
                "type": "image",
                "content": "Image detected",
                "bbox": (x, y, w, h)
            })
    
    return image_elements

def process_pdf(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    all_elements = []

    for image in images:
        page_elements = process_page(image)
        image_elements = detect_images(image)
        all_elements.extend(page_elements + image_elements)

    # Sort all elements by their vertical position
    all_elements.sort(key=lambda x: x['bbox'][1])

    # Convert to the desired JSON structure
    structured_output = {}
    current_heading = None
    current_subheading = None

    for element in all_elements:
        if element['type'] == 'heading':
            current_heading = element['content']
            current_subheading = None
            structured_output[current_heading] = {}
        elif element['type'] == 'subheading' and current_heading:
            current_subheading = element['content']
            structured_output[current_heading][current_subheading] = {}
        elif current_heading:
            if current_subheading:
                if element['type'] not in structured_output[current_heading][current_subheading]:
                    structured_output[current_heading][current_subheading][element['type']] = []
                structured_output[current_heading][current_subheading][element['type']].append(element['content'])
            else:
                if element['type'] not in structured_output[current_heading]:
                    structured_output[current_heading][element['type']] = []
                structured_output[current_heading][element['type']].append(element['content'])

    return structured_output

# Usage
pdf_path = 'path/to/your/pdf/file.pdf'
results = process_pdf(pdf_path)

# Output as JSON
print(json.dumps(results, indent=2))
