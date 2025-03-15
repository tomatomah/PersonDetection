import argparse
import os
import sys
import time

import cv2
import data_utils
import numpy as np
import onnxruntime
import utils
from tqdm import tqdm


class DetectionResult(object):
    def __init__(self):
        self.left_top_x = -1
        self.left_top_y = -1
        self.right_bottom_x = -1
        self.right_bottom_y = -1
        self.center_x = -1
        self.center_y = -1
        self.width = -1
        self.height = -1
        self.prob = -1.0
        self.objectness = -1.0
        self.class_prob = -1.0
        self.class_id = -1


class Detector(object):
    def __init__(self, args):
        self.config = utils.load_config(args.config_path)

        self.save_dir = os.path.join(self.config["detecting"]["save_dir"], "detect")
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_image_dir = os.path.join(self.save_dir, "images")
        self.save_movie_dir = os.path.join(self.save_dir, "movie")

        self.min_image_size = (240, 320)  # QVGA
        self.max_image_size = (2160, 3840)  # 4K

        self.model = None
        self.image_list = []
        self.cap = None
        self.frame_length = 0
        self.video_writer = None

        self._init_model()
        self._init_data()

        self.frame_index = 0
        self.inv_scales = []

    def _init_model(self):
        model_path = os.path.join(self.config["onnx"]["save_dir"], "onnx", self.config["detecting"]["model_file"])
        self.model = onnxruntime.InferenceSession(model_path)

    def _init_data(self):
        if self.config["detecting"]["input_type"] == "image":
            self._set_image()
        elif self.config["detecting"]["input_type"] == "movie":
            self._set_movie()
        else:
            print("Input file type is not supported")
            sys.exit(1)

    def _set_image(self):
        target_ext = (".jpg", ".jpeg", ".png", ".bmp")
        with open(self.config["detecting"]["image_list_file"], "r", encoding="utf-8") as f:
            for line in f.readlines():
                image_path = line.strip()
                ext = os.path.splitext(os.path.basename(image_path))[1]

                if not ext.lower().endswith(target_ext):
                    print("Image file type is not supported")
                    sys.exit(1)
                self.image_list.append(image_path)

        self.frame_length = len(self.image_list)

        os.makedirs(self.save_image_dir, exist_ok=True)

    def _set_movie(self):
        target_ext = ".mp4"
        file_name, ext = os.path.splitext(os.path.basename(self.config["detecting"]["movie_path"]))
        if not ext.lower().endswith(target_ext):
            print("Movie file type is not supported")
            sys.exit(1)

        self.cap = cv2.VideoCapture(self.config["detecting"]["movie_path"])
        if not self.cap.isOpened():
            print("Movie file cannot be opened")
            sys.exit(1)

        self.frame_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_movie_path = os.path.join(self.save_movie_dir, file_name + "_detect.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(output_movie_path, fourcc, fps, (width, height))

        os.makedirs(self.save_movie_dir, exist_ok=True)

    def _load_image(self):
        image_path = self.image_list[self.frame_index]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            sys.exit(1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_movie(self):
        ret, image = self.cap.read()
        if not ret:
            print("Failed to read frame from the video.")
            sys.exit(1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _get_image(self):
        try:
            if self.config["detecting"]["input_type"] == "image":
                image = self._load_image()
            if self.config["detecting"]["input_type"] == "movie":
                image = self._load_movie()

            self.frame_index += 1
            return image
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def _is_valid_image_size(self, image):
        height, width, channels = image.shape
        return (
            width >= self.min_image_size[1]
            and width <= self.max_image_size[1]
            and height >= self.min_image_size[0]
            and height <= self.max_image_size[0]
            and channels == 3
        )

    def _determine_model_input_shape(self, image_width, image_height):
        # Calculate the remainder when dividing the height and width by a specific multiple
        height_remainder = image_height % 32
        width_remainder = image_width % 32

        # If there is a remainder in the height direction
        if height_remainder != 0:
            # Calculate the quotient when dividing by the multiple
            height_quotient = image_height // 32

            # Calculate the nearest multiple to the height value
            dst_image_height = 32 * (height_quotient + 1)
        else:
            dst_image_height = image_height

        # If there is a remainder in the width direction
        if width_remainder != 0:
            # Calculate the quotient when dividing by the multiple
            width_quotient = image_width // 32

            # Calculate the nearest multiple to the width value
            dst_image_width = 32 * (width_quotient + 1)
        else:
            dst_image_width = image_width

        return dst_image_width, dst_image_height

    def _preprocess_image(self, image):
        resize_images = []
        for scale in self.config["detecting"]["scale_list"]:
            if not (0.0 < scale <= 1.0):
                print("Scale value is not supported.")
                sys.exit(1)

            self.inv_scales.append(1 / scale)

            scale_image_height = int(image.shape[0] * scale)
            scale_image_width = int(image.shape[1] * scale)

            input_width, input_height = self._determine_model_input_shape(scale_image_width, scale_image_height)
            tmp_image = cv2.resize(
                image, dsize=(scale_image_width, scale_image_height), interpolation=cv2.INTER_LINEAR
            )
            resize_image = np.zeros([input_height, input_width, 3], dtype=np.float32)
            resize_image[0:scale_image_height, 0:scale_image_width, :] = tmp_image
            resize_images.append(resize_image)

        input_images = []
        for resize_image in resize_images:
            normalize_image = resize_image / 255
            reshape_image = np.transpose(normalize_image, (2, 0, 1))
            input_image = np.expand_dims(reshape_image, 0)
            input_images.append(input_image)

        return input_images

    def _sigmoid(self, x):
        # Consider overflow
        mask = x >= 0
        y = np.zeros_like(x)  # Create an array initialized with 0s having the same shape as x
        y[mask] = 1.0 / (1 + np.exp(-x[mask]))  # x >= 0, f(x) = 1/(1+e^(-x))
        y[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))  # x < 0, f(x) = e^x/(e^x+1)
        return y

    def _decode_outputs(self, outputs):
        # List to store height and width of each feature map (e.g., for 480×640 input: [(60,80), (30,40), (15,20)])
        hw_list = []

        # Flatten each feature map
        reshaped_outputs = []
        for output in outputs:
            # Record height and width of feature map
            h, w = output.shape[-2:]
            hw_list.append((h, w))

            # Flatten feature map: (batch_size, 4+1+num_classes, h, w) → (4+1+num_classes, h*w)
            reshaped_output = output[0].reshape(output.shape[1], -1)
            reshaped_outputs.append(reshaped_output)

        # Concatenate flattened feature maps: (4+1+num_classes, total grid cells)
        concatenated_outputs = np.concatenate(reshaped_outputs, axis=1)

        # Transpose axes: (4+1+num_classes, total grid cells) → (total grid cells, 4+1+num_classes)
        transposed_outputs = concatenated_outputs.T

        # Initialize array to store decoded results
        decoded_data = np.zeros_like(transposed_outputs)

        # Apply sigmoid to confidence scores and class probabilities to scale to [0,1] range
        decoded_data[:, 4:] = self._sigmoid(transposed_outputs[:, 4:])

        # Calculate grid coordinates and stride values for each feature map
        grids = []
        strides = []

        # Stride values (downsampling ratios): 8, 16, 32
        for (h, w), stride in zip(hw_list, [8, 16, 32]):
            # Generate grid coordinates for each feature map level
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grid = np.stack((grid_x, grid_y), axis=2).reshape(-1, 2)
            grids.append(grid)

            # Create array of stride values for each grid cell
            strides.append(np.full((grid.shape[0], 1), stride))

        # Concatenate grid coordinates from all feature maps
        grids = np.concatenate(grids, axis=0).astype(decoded_data.dtype)

        # Concatenate stride values from all feature maps
        strides = np.concatenate(strides, axis=0).astype(decoded_data.dtype)

        # Calculate bounding box center coordinates (cx, cy):
        # (tx + gx) * stride, (ty + gy) * stride
        decoded_data[:, :2] = (transposed_outputs[:, :2] + grids) * strides

        # Calculate bounding box width and height (w, h):
        # e^tw * stride, e^th * stride
        decoded_data[:, 2:4] = np.exp(transposed_outputs[:, 2:4]) * strides

        return decoded_data

    def _get_detect_results(self, decode_data_list):
        # Initialize array to store all detection results
        # Format: [cx, cy, width, height, ltx, lty, rbx, rby, objectness, class_prob, prob, class_pred]
        detections = np.zeros((0, 12))

        # Process each scale's decoded data
        for scale_index, decode_data in enumerate(decode_data_list):
            # Create array for bounding box information
            # Format: [cx, cy, width, height, ltx, lty, rbx, rby]
            bbox_info = np.zeros((decode_data.shape[0], 8), np.float32)

            # Store center coordinates and dimensions
            bbox_info[:, 0] = decode_data[:, 0]  # cx
            bbox_info[:, 1] = decode_data[:, 1]  # cy
            bbox_info[:, 2] = decode_data[:, 2]  # width
            bbox_info[:, 3] = decode_data[:, 3]  # height

            # Calculate corner coordinates
            bbox_info[:, 4] = bbox_info[:, 0] - bbox_info[:, 2] / 2  # ltx (left-top x)
            bbox_info[:, 5] = bbox_info[:, 1] - bbox_info[:, 3] / 2  # lty (left-top y)
            bbox_info[:, 6] = bbox_info[:, 0] + bbox_info[:, 2] / 2  # rbx (right-bottom x)
            bbox_info[:, 7] = bbox_info[:, 1] + bbox_info[:, 3] / 2  # rby (right-bottom y)

            # Get maximum class probability for each detection
            class_prob = np.max(decode_data[:, 5 : 5 + self.config["model"]["num_classes"]], axis=1, keepdims=True)

            # Get predicted class index (convert to float for consistency)
            class_pred = np.argmax(
                decode_data[:, 5 : 5 + self.config["model"]["num_classes"]], axis=1, keepdims=True
            ).astype(np.float32)

            # Get objectness confidence scores
            objectness = decode_data[:, 4, np.newaxis]

            # Calculate final probability (objectness * class probability)
            prob = objectness * class_prob

            # Create mask for detections above threshold
            prob_mask = prob[:, 0] >= self.config["detecting"]["detect_threshold"]

            # Combine all detection information
            # Format: [cx, cy, width, height, ltx, lty, rbx, rby, objectness, class_prob, prob, class_pred]
            detect_data = np.concatenate((bbox_info, objectness, class_prob, prob, class_pred), axis=1)

            # Filter detections using confidence threshold
            detect_data = detect_data[prob_mask]

            # Skip if no detections passed the threshold
            if detect_data.shape[0] == 0:
                continue

            # Get inverse scale factor for this scale level
            inv_scale = np.full((detect_data.shape[0], 1), self.inv_scales[scale_index])

            # Scale coordinates back to original image dimensions
            detect_data[:, 0:4] = detect_data[:, 0:4] * inv_scale

            # Recalculate corner coordinates with scaled dimensions
            detect_data[:, 4:6] = detect_data[:, 0:2] - (detect_data[:, 2:4] / 2.0)  # ltx, lty
            detect_data[:, 6:8] = detect_data[:, 0:2] + (detect_data[:, 2:4] / 2.0)  # rbx, rby

            # Add these detections to the overall results
            detections = np.concatenate([detections, detect_data], axis=0)

        # Apply Non-Maximum Suppression to remove overlapping detections
        nms_index = self._non_maximum_supression(
            detections[:, 4:8], detections[:, 10], detections[:, 11].astype(np.int32)
        )

        # Filter detections using NMS indices (if any detections remain)
        if len(detections) != 0:
            detections = detections[nms_index]

        # Convert array data to DetectionResult objects
        results = []
        for result in detections:
            detection_result = DetectionResult()
            detection_result.center_x = int(result[0])
            detection_result.center_y = int(result[1])
            detection_result.width = int(result[2])
            detection_result.height = int(result[3])
            detection_result.left_top_x = int(result[4])
            detection_result.left_top_y = int(result[5])
            detection_result.right_bottom_x = int(result[6])
            detection_result.right_bottom_y = int(result[7])
            detection_result.objectness = float(result[8])
            detection_result.class_prob = float(result[9])
            detection_result.prob = float(result[10])
            detection_result.class_id = int(result[11])

            results.append(detection_result)

        return results

    def _non_maximum_supression(self, boxes, scores, class_ids):
        # Get unique class IDs
        unique_classes = np.unique(class_ids)

        # Initialize array to store indices of kept boxes
        keep_indices = []

        # Process each class separately
        for class_id in unique_classes:
            # Get indices of boxes for this class
            class_mask = class_ids == class_id
            class_indices = np.where(class_mask)[0]

            # Get boxes and scores for this class
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            # Apply NMS for this class
            class_keep = self._standard_nms(class_boxes, class_scores)

            # Map back to original indices
            keep_indices.extend(class_indices[class_keep])

        return np.array(keep_indices)

    def _standard_nms(self, boxes, scores):
        # Get the indices of the detection bounding boxes when sorted in ascending order of scores
        idxs = scores.argsort()

        # Variable to store the detection bounding boxes that will be kept after NMS processing
        keep = []

        while idxs.size > 0:
            # Store the index of the detection bounding box with the highest score
            max_score_index = idxs[-1]

            # Store the coordinates of the detection bounding box with the highest score
            # max_score_box => (1, 4)
            max_score_box = boxes[max_score_index][None, :]

            # Append the index of the selected detection bounding box to keep
            keep.append(max_score_index)

            if idxs.size == 1:
                break

            # Remove the index of the selected detection bounding box
            idxs = idxs[:-1]

            # Store the coordinates of the remaining detection bounding boxes that are not selected
            other_boxes = boxes[idxs]

            # Calculate the IoU between the selected detection bounding box and the remaining detection bounding boxes
            ious = data_utils.compute_iou(max_score_box, other_boxes)

            # Keep only the detection bounding boxes with IoU below the threshold in idxs
            idxs = idxs[ious < self.config["detecting"]["iou_threshold"]]

        return np.array(keep)

    def _detect_image(self, image):
        if not self._is_valid_image_size(image):
            raise ValueError("Input image size is not supported.")

        input_images = self._preprocess_image(image)

        decode_data_list = []
        for input_image in input_images:
            outputs = self.model.run(None, {self.config["onnx"]["input_blob_name"]: input_image})

            decode_data = self._decode_outputs(outputs)
            decode_data_list.append(decode_data)

        results = self._get_detect_results(decode_data_list)

        return results

    def _plot_one_box(self, pt1, pt2, image, category, prob, color):
        tl = round(0.001 * max(image.shape[0:2])) + 1  # line thickness
        cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color, thickness=tl)

        tf = max(tl - 2, 1)  # font thickness
        text = category + " " + f"{prob:.2f}"
        t_size, _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3, thickness=tf)
        cv2.rectangle(
            img=image,
            pt1=pt1,
            pt2=(pt1[0] + t_size[0], pt1[1] - t_size[1] - 3),
            color=color,
            thickness=-1,
        )
        cv2.putText(
            img=image,
            text=text,
            org=(pt1[0], pt1[1] - 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=tl / 3,
            color=[0, 0, 0],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    def run(self):
        sum_process_time = 0
        pbar = tqdm(total=self.frame_length)

        while True:
            image = self._get_image()
            if image is None:
                break

            start_time = time.perf_counter()
            results = self._detect_image(image)
            end_time = time.perf_counter()
            process_time = end_time - start_time
            if len(results) != 0:
                for result in results:
                    class_name = self.config["detecting"]["id_to_class"][result.class_id]
                    pt1 = (result.left_top_x, result.left_top_y)
                    pt2 = (result.right_bottom_x, result.right_bottom_y)

                    self._plot_one_box(
                        pt1,
                        pt2,
                        image,
                        class_name,
                        result.prob,
                        self.config["detecting"]["plot_colors"][result.class_id],
                    )

            if self.config["detecting"]["input_type"] == "image":
                image_path = self.image_list[self.frame_index - 1]
                file_name = os.path.basename(image_path)
                output_image_path = os.path.join(self.save_image_dir, file_name)
                cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if self.config["detecting"]["input_type"] == "movie":
                # file_name = os.path.splitext(os.path.basename(self.config["detecting"]["movie_path"]))[0]
                # frame_index = self.frame_index - 1
                # digit = len(str(self.frame_length))
                # output_image_path = (
                #     self.save_movie_dir + file_name + "_frame_" + str(frame_index).zfill(digit) + ".jpg"
                # )
                # cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            sum_process_time += process_time
            pbar.update()

            # Wait for ESC key to exit
            if cv2.waitKey(30) == 27:
                break

            if self.frame_length == self.frame_index:
                break

        pbar.close()

        average_process_time = sum_process_time / self.frame_length
        fps = 1 / average_process_time

        print(f"Process time: {average_process_time:.4f}[s], fps: {fps:.4f}")

        if self.config["detecting"]["input_type"] == 1:
            self.cap.release()
            cv2.destroyAllWindows()
            self.video_writer.release()


def main(args):
    detector = Detector(args)
    detector.run()

    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting program for object-detection model")
    parser.add_argument(
        "--config_path", type=str, default="config/config.yml", help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    main(args)
