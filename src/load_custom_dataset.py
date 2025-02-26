import os
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm


def load_dataset(dataset_param, class_to_id):
    image_paths, labels = format_data(dataset_param, class_to_id)

    return image_paths, labels


def format_data(dataset_param, class_to_id):
    format_image_paths = []
    format_labels = []

    images_directory = os.path.join(dataset_param["dataset_directory"], "images")
    annotations_directory = os.path.join(dataset_param["dataset_directory"], "annotations")

    image_anno_pairs = build_image_annotation_pairs(images_directory, annotations_directory)

    class_names = class_to_id.keys()
    for image_path, annotation_path in tqdm(image_anno_pairs, desc="[Loading Custom Dataset]"):
        objects = parse_voc_annotation(annotation_path)
        label = format_voc_label(objects, class_to_id, class_names)

        # Box is empty
        if len(label) == 0:
            continue

        format_image_paths.append(image_path)
        format_labels.append(label)

    return format_image_paths, format_labels


def build_image_annotation_pairs(images_directory, annotations_directory):
    subdirs = [
        item
        for item in os.listdir(images_directory)
        if os.path.isdir(os.path.join(images_directory, item))
        and os.path.isdir(os.path.join(annotations_directory, item))
    ]

    image_anno_pairs = []
    for subdir in sorted(subdirs):
        image_dir = os.path.join(images_directory, subdir)
        anno_dir = os.path.join(annotations_directory, subdir)

        image_files = {
            os.path.splitext(f)[0]: os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(image_dir, f))
        }
        anno_files = {
            os.path.splitext(f)[0]: os.path.join(anno_dir, f)
            for f in os.listdir(anno_dir)
            if f.lower().endswith(".xml") and os.path.isfile(os.path.join(anno_dir, f))
        }
        common_files = set(image_files.keys()) & set(anno_files.keys())

        for base_name in common_files:
            image_anno_pairs.append((image_files[base_name], anno_files[base_name]))

    return image_anno_pairs


def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = bbox.find("xmin")
        ymin = bbox.find("ymin")
        xmax = bbox.find("xmax")
        ymax = bbox.find("ymax")
        xmin = int(xmin.text)
        ymin = int(ymin.text)
        xmax = int(xmax.text)
        ymax = int(ymax.text)

        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})

    return objects


def format_voc_label(objects, class_to_id, class_names):
    label_info = []
    for obj in objects:
        name = obj["name"]
        if name not in class_names:
            continue

        class_id = class_to_id[name]
        bbox = obj["bbox"]

        bbox_ltx = bbox[0]  # left-top x
        bbox_lty = bbox[1]  # left-top y
        bbox_rbx = bbox[2]  # right-bottom x
        bbox_rby = bbox[3]  # right-bottom y

        width = bbox_rbx - bbox_ltx
        height = bbox_rby - bbox_lty
        bbox_cx = bbox_ltx + (width * 0.5)
        bbox_cy = bbox_lty + (height * 0.5)

        if (
            bbox_ltx < 0.0
            or bbox_lty < 0.0
            or bbox_rbx < 0.0
            or bbox_rby < 0.0
            or bbox_cx < 0.0
            or bbox_cy < 0.0
            or width <= 3.0
            or height <= 3.0
        ):
            continue

        # [cx, cy, width, height, class_id]
        label_info.append([bbox_cx, bbox_cy, width, height, class_id])

    # Box is empty
    if not label_info:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(label_info, dtype=np.float32)
