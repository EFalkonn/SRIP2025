from flask import Flask, request, jsonify                # Import Flask web framework and helpers
from werkzeug.utils import secure_filename               # Import function to secure uploaded filenames
import os                                                # Import OS module for file operations
import cv2                                               # Import OpenCV for image processing
import pytesseract                                       # Import pytesseract for OCR
from ultralytics import YOLO                             # Import YOLO model from ultralytics for detection
import re                                                # Import regex module for text processing
import json                                              # Import JSON module

app = Flask(__name__)                                    # Create Flask app instance
UPLOAD_FOLDER = 'D:/ABC TOLL/uploads/'                   # Set upload folder path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER              # Add upload folder to Flask config
os.makedirs(UPLOAD_FOLDER, exist_ok=True)                # Create upload folder if it doesn't exist

CONFIDENCE_THRESHOLD = 0.3                               # Set confidence threshold for detections
logomod = "D:/ABC TOLL/LOGO177/weights/best.pt"          # Path to logo detection model weights
carcol = "D:/ABC TOLL/CARCOL/content/runs/detect/train2/weights/best.pt"  # Path to car color model
licol = "D:/ABC TOLL/LICOLI/classify/train2/weights/best.pt"              # Path to license plate color model

results_dict = {"objects_detected": [], "sticker_count": 0, "flag_count": 0}  # Initialize results dictionary

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)     # Remove all non-alphanumeric characters
    return cleaned_text                                  # Return cleaned text

def process_ocr_text(ocr_text):
    processed_text = clean_text(ocr_text.upper())         # Convert text to uppercase and clean it
    return processed_text                                # Return processed text


"""
This function `crop_and_display_license_plate` takes an image and a bounding box (`license_plate_box`) as input, crops the license plate from the image, uses a YOLO model (`licol`) to detect the license plate color, and appends the detection result to a global dictionary (`results_dict`). The result includes the class name ("license-plate"), bounding box coordinates, OCR text, and detected color.
"""
def crop_and_display_license_plate(image, license_plate_box):
    global results_dict                                  # Use global results_dict
    x1, y1, x2, y2 = license_plate_box                   # Unpack bounding box coordinates
    license_plate_image = image[int(y1):int(y2), int(x1):int(x2)]  # Crop license plate from image
    model = YOLO(licol)                                  # Load license plate color classification model
    print("Performing license plate detection...")        # Debug print
    results = model.predict(license_plate_image, conf=0.5) # Predict license plate color
    top1_confidence = results[0].probs.top1conf          # Get top-1 confidence
    top1_class_idx = results[0].probs.top1               # Get top-1 class index
    best_classification = results[0].names[top1_class_idx] # Get class name
    tesseract_text = pytesseract.image_to_string(license_plate_image, config='--psm 7') # OCR on plate
    tesseract_text = clean_text(tesseract_text)           # Clean OCR text
    ocr_text = process_ocr_text(tesseract_text)           # Process OCR text

    results_dict["objects_detected"].append({             # Append license plate info to results
        "class_name": "license-plate",
        "bounding_box": [x1, y1, x2, y2],
        "text": ocr_text,
        "color": best_classification
    })


"""
This code snippet defines a function `crop_car` that takes an `image` and a `car_box` (bounding box coordinates) as input. It:

1. Crops the car from the image using the bounding box coordinates.
2. Loads a car color detection model (YOLO) and predicts the car color.
3. If a detection is made, it extracts the class name (color) and appends the car information (class name, bounding box, and color) to the `results_dict`.
4. If no detection is made, it appends an "unknown" car color to the `results_dict`.

In summary, this function detects the color of a car in an image and updates the `results_dict` with the detection result.
"""
def crop_car(image, car_box):
    global results_dict                                  # Use global results_dict
    x1, y1, x2, y2 = car_box                             # Unpack bounding box coordinates
    car_image = image[int(y1):int(y2), int(x1):int(x2)]  # Crop car from image
    model = YOLO(carcol)                                 # Load car color detection model
    print("Performing car detection...")                 # Debug print
    results = model.predict(car_image, conf=0.5)         # Predict car color
    
    if results is not None and len(results) > 0:         # If results exist
        if isinstance(results, list):                    # If results is a list
            results = results[0]                         # Use first result
            names_dict = results.names                   # Get class names
            boxes = results.boxes.xyxy.tolist()          # Get bounding boxes
            class_labels = results.boxes.cls.tolist()    # Get class indices

            for i, box in enumerate(boxes):              # Iterate over detected boxes
                class_idx = class_labels[i]              # Get class index
                class_name = results.names[class_idx]    # Get class name

            results_dict["objects_detected"].append({    # Append car info to results
                "class_name": "car",
                "bounding_box": [x1, y1, x2, y2],
                "color": class_name
            })
    else:
        results_dict["objects_detected"].append({        # If no car detected, append unknown
            "class_name": "car",
            "color": "unknown"
        })

"""
This function `crop_logo` takes an image and a bounding box (`logo_box`) as input, crops the logo from the image, uses a YOLO model (`logomod`) to detect the logo, and appends the detection result to a global dictionary (`results_dict`). The result includes the class name ("logo"), bounding box coordinates, and the manufacturer of the logo.
"""
def crop_logo(image, logo_box):
    global results_dict                                  # Use global results_dict
    x1, y1, x2, y2 = logo_box                            # Unpack bounding box coordinates
    logo_image = image[int(y1):int(y2), int(x1):int(x2)] # Crop logo from image
    model = YOLO(logomod)                                # Load logo detection model
    print("Performing logo detection...")                # Debug print
    results = model.predict(logo_image, conf=0.5)        # Predict logo
    top1_confidence = results[0].probs.top1conf          # Get top-1 confidence
    top1_class_idx = results[0].probs.top1               # Get top-1 class index
    best_classification = results[0].names[top1_class_idx] # Get class name

    results_dict["objects_detected"].append({            # Append logo info to results
        "class_name": "logo",
        "bounding_box": [x1, y1, x2, y2],
        "manufacturer": best_classification
    })

"""
This is a function named `process_image` that takes an `image_path` as input and performs object detection on the image using a YOLO model. It then updates a global dictionary `results_dict` with the detection results, including counts of stickers and flags, and information about detected license plates, cars, and logos. If no logo or car is detected, it appends unknown values to the `results_dict`.
"""
def process_image(image_path):
    global results_dict                                  # Use global results_dict
    image = cv2.imread(image_path)                       # Read image from file
    model = YOLO('D:/ABC TOLL/runs_archive/segment/train3/weights/best.pt') # Load main detection model
    print("Processing image...")                         # Debug print
    results = model(image, conf=CONFIDENCE_THRESHOLD)    # Run detection on image
    if isinstance(results, list):                        # If results is a list
        results = results[0]                             # Use first result

    boxes = results.boxes.xyxy.tolist()                  # Get bounding boxes
    class_labels = results.boxes.cls.tolist()            # Get class indices

    sticker_count = 0                                    # Initialize sticker count
    flag_count = 0                                       # Initialize flag count

    for i, box in enumerate(boxes):                      # Iterate over detected objects
        class_idx = class_labels[i]                      # Get class index
        class_name = results.names[class_idx]            # Get class name

        if class_name == "sticker":                      # If sticker detected
            sticker_count += 1
        elif class_name == "flag":                       # If flag detected
            flag_count += 1

        if class_name == "license-plate":                # If license plate detected
            crop_and_display_license_plate(image, box)

        if class_name == "car":                          # If car detected
            crop_car(image, box)

        if class_name == "logo":                         # If logo detected
            crop_logo(image, box)

    results_dict["sticker_count"] = sticker_count        # Update sticker count in results
    results_dict["flag_count"] = flag_count              # Update flag count in results

    if not any(item["class_name"] == "logo" for item in results_dict["objects_detected"]): # If no logo detected
        results_dict["objects_detected"].append({
            "class_name": "logo",
            "manufacturer": "unknown"
        })

    if not any(item["class_name"] == "car" for item in results_dict["objects_detected"]):  # If no car detected
        results_dict["objects_detected"].append({
            "class_name": "car",
            "color": "unknown"
        })



"""
This code defines a Flask API endpoint (`/upload`) that accepts a file upload via HTTP POST. It checks if a file is present, if the filename is not empty, and if the file type is allowed. If all checks pass, it saves the file to disk, processes the image using the `process_image` function, and returns the results as JSON. If any checks fail, it returns an error message with a 400 status code.
"""
@app.route('/upload', methods=['POST'])                  # Define upload endpoint
def upload_file():
    file = request.files['file']                         # Get uploaded file
    if 'file' not in request.files:                      # If no file part in request
        return jsonify({"error": "No file part"}), 400
    
    if file.filename == '':                              # If filename is empty
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):             # If file is allowed type
        print("**")                                      # Debug print
        filename = secure_filename(file.filename)        # Secure the filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Build file path
        file.save(file_path)                             # Save file to disk
        print("File saved at:", file_path)               # Debug print
        process_image(file_path)                         # Process the uploaded image
        print("Image processed successfully.")           # Debug print
        os.remove(file_path)                             # Remove file after processing
        print("File removed from server.")               # Debug print
        return jsonify(results_dict), 200                # Return results as JSON
    else:
        return jsonify({"error": "File type not allowed"}), 400 # Invalid file type



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'} # Check allowed extensions

if __name__ == '__main__':                              # If script is run directly
    app.run(debug=True)                                 # Start Flask app in debug mode
