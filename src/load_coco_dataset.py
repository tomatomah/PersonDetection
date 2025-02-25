import glob
import os
import sys

import numpy as np
import simplejson as json
from tqdm import tqdm

category_dict = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def load_dataset(dataset_param, subset):
    image_paths, labels = format_data(dataset_param, subset)

    return image_paths, labels


def format_data(dataset_param, subset):
    format_image_paths = []
    format_labels = []

    image_paths = get_image_paths(dataset_param[f"{subset}_image_directory"])
    coco_labels = get_coco_labels(dataset_param[f"{subset}_label_path"], subset)

    assert len(image_paths) == len(coco_labels)

    oldid_to_newid = {k: i for i, k in enumerate(category_dict.keys())}

    for image_path in image_paths:
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])

        if image_id not in coco_labels.keys():
            continue

        # Box is empty
        if len(coco_labels[image_id]["objects"]) == 0:
            continue

        if subset == "train":
            # Skip crowd instances as they represent groups of objects with lower annotation quality
            if dataset_param["delete_iscrowd_image"] and check_iscrowd_object(coco_labels[image_id]["objects"]):
                continue

        if not os.path.isfile(image_path):
            print(f"The file '{image_path}' does not exist.")
            sys.exit(1)

        label = format_label(coco_labels[image_id]["objects"], oldid_to_newid)

        format_image_paths.append(image_path)
        format_labels.append(label)

    return format_image_paths, format_labels


def get_image_paths(dir_path):
    # collect paths of existing jpg images
    image_paths = [
        path
        for path in glob.glob(os.path.join(dir_path, "*"))
        if path.lower().endswith(".jpg") and os.path.isfile(path)
    ]

    return image_paths


def get_coco_labels(label_file_path, subset):
    if not os.path.isfile(label_file_path):
        print(f"'{label_file_path}' does not exist.")
        sys.exit(1)

    # coco_json: dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    coco_json = read_json_file(label_file_path, subset)
    coco_labels = create_labels(coco_json)

    return coco_labels


def read_json_file(label_file_path, subset):
    try:
        with open(label_file_path, "r", encoding="utf-8") as f:
            coco_json_data = json.load(f, object_hook=lambda obj: print_load_progress(obj, subset))
    except Exception as e:
        print(e)
        sys.exit(1)

    return coco_json_data


def print_load_progress(obj, subset):
    images_dict_info = obj.get("images")
    if images_dict_info:
        for _ in tqdm(images_dict_info, desc=f"[Loading MSCOCO Dataset: {subset} json file]"):
            pass

    return obj


def create_labels(coco_json):
    coco_labels = {
        image_info["id"]: {"file_name": image_info["file_name"], "objects": []} for image_info in coco_json["images"]
    }

    for annotation_info in coco_json["annotations"]:
        if annotation_info["image_id"] not in coco_labels:
            continue

        object_dict = {
            "bbox": annotation_info["bbox"],
            "iscrowd": annotation_info["iscrowd"],
            "category_id": annotation_info["category_id"],
        }

        coco_labels[annotation_info["image_id"]]["objects"].append(object_dict)

    return coco_labels


def check_iscrowd_object(objects):
    for object_info in objects:
        if object_info["iscrowd"] == 1:
            return True

    return False


def format_label(objects, oldid_to_newid):
    label_info = []
    for object_info in objects:
        bbox = object_info["bbox"]
        category_id = convert_category_id(object_info["category_id"], oldid_to_newid)

        # Calculate box coordinates
        bbox_ltx = bbox[0]
        bbox_lty = bbox[1]
        width = bbox[2]
        height = bbox[3]

        # Calculate center coordinates
        bbox_cx = bbox_ltx + (width * 0.5)
        bbox_cy = bbox_lty + (height * 0.5)

        # Calculate max coordinates
        bbox_rbx = bbox_ltx + width
        bbox_rby = bbox_lty + height

        # Validate all coordinates and dimensions
        invalid_coord = any(coord < 0.0 for coord in [bbox_ltx, bbox_lty, bbox_rbx, bbox_rby, bbox_cx, bbox_cy])
        invalid_wh = any(x < 0.0 for x in [width, height])
        if invalid_coord or invalid_wh:
            print(
                f"Error: Invalid negative or zero coordinates/dimensions detected:\n"
                f"Left-Top     : ({bbox_ltx:.2f}, {bbox_lty:.2f})\n"
                f"Right-Bottom : ({bbox_rbx:.2f}, {bbox_rby:.2f})\n"
                f"Center       : ({bbox_cx:.2f}, {bbox_cy:.2f})\n"
                f"Size         : width={width:.2f}, height={height:.2f}"
            )
            sys.exit(1)

        # [cx, cy, width, height, category_id]
        label_info.append([bbox_cx, bbox_cy, width, height, category_id])

    return np.array(label_info, dtype=np.float32)


def convert_category_id(category_id, oldid_to_newid):
    if category_id not in oldid_to_newid.keys():
        print(f"category_id={category_id} does not exist")
        sys.exit(1)

    return oldid_to_newid[category_id]
