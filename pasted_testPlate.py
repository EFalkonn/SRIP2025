import re

log_file_path = r"C:\Users\afzal\Downloads\VehicleIVideos\plateInfo.log"

def get_valid_plate_Number(plate_text):
    return_value = re.search(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', plate_text)
    #print(return_value.group(0) if return_value else "No valid plate found")
    return return_value.group(0) if return_value else ""

with open(log_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 2025-06-21 12:24:41,047 - INFO - ## License Plate Detected: ITWASCKES11, Color: White, Bounding Box: [160, 569, 302, 611]
        # ITWASCKES11
        match = re.search(r'License Plate Detected: ([^,]*),', line)
        if match:
            plate_text = match.group(1).strip()
            ret_numbr = get_valid_plate_Number(plate_text)
            print(f"Before {plate_text} : {ret_numbr}")


def get_all_license_plate_texts(log_file_path):
    all_plate_texts=[]
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'License Plate Detected: ([^,]*),', line)
            if match:
                plate_text = match.group(1).strip()
                all_plate_texts.append(plate_text)
                ret_numbr = get_valid_plate_Number(plate_text)
                print(f"Before {plate_text} : {ret_numbr}")
    return all_plate_texts


def filter_valid_plates(plate_texts):
    valid_plates = []
    for plate in plate_texts:
        if re.fullmatch(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', plate):
            valid_plates.append(plate)
    return valid_plates

log_file_path = r"C:\Users\afzal\Downloads\VehicleIVideos\plateInfo.log"
all_plate_texts = get_all_license_plate_texts(log_file_path)
valid_plates = filter_valid_plates(all_plate_texts)