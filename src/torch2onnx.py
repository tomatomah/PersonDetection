import argparse
import os
import sys

import onnx
import torch
import torch.nn as nn
import utils
from models import create_model
from onnxsim import simplify


class ObjectDetectionPostProcessor(nn.Module):
    """
    Post-processing module for object detection models.

    This module takes the raw outputs from an object detection model and applies
    decoding and filtering to produce the final detection results.
    """

    def __init__(self, base_model, config):
        super(ObjectDetectionPostProcessor, self).__init__()
        self.base_model = base_model
        self.num_classes = config["model"]["num_classes"]
        self.detect_threshold = config["detecting"]["detect_threshold"]
        self.iou_threshold = config["detecting"]["iou_threshold"]

        # Feature map strides for each detection level
        self.strides = nn.Parameter(torch.tensor([8, 16, 32], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # Get raw outputs from base model
        outputs = self.base_model(x)

        # Decode the outputs into box coordinates and scores
        decoded_outputs = self._decode_outputs(outputs)

        # Filter detections based on confidence
        filtered_boxes, filtered_scores, filtered_classes, valid_count = self._filter_detections(decoded_outputs)

        return filtered_boxes, filtered_scores, filtered_classes, valid_count

    def _decode_outputs(self, outputs):
        """
        Decode raw model outputs into box coordinates and class scores.
        """
        batch_size = outputs[0].shape[0]
        all_outputs = []

        # Process each feature map level
        for i, output in enumerate(outputs):
            _, _, h, w = output.shape
            stride = self.strides[i]

            # Create grid coordinates
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack((grid_x, grid_y), dim=2).float()
            grid_xy = grid_xy.view(1, h, w, 2).expand(batch_size, h, w, 2)

            # Reshape output tensor for easier processing
            feat = output.permute(0, 2, 3, 1)

            # Extract components from feature map
            box_xy = feat[:, :, :, :2]  # Center coordinates
            box_wh = feat[:, :, :, 2:4]  # Width and height
            obj_conf = feat[:, :, :, 4:5]  # Objectness confidence
            cls_conf = feat[:, :, :, 5:]  # Class confidence

            # Apply grid offset and activation functions
            box_xy = box_xy + grid_xy
            box_wh = torch.exp(box_wh)
            obj_conf = torch.sigmoid(obj_conf)
            cls_conf = torch.sigmoid(cls_conf)

            # Scale coordinates by stride
            box_xy = box_xy * stride
            box_wh = box_wh * stride

            # Convert from center coordinates to corner coordinates
            box_x1y1 = box_xy - box_wh / 2  # Top-left corner
            box_x2y2 = box_xy + box_wh / 2  # Bottom-right corner
            boxes = torch.cat([box_x1y1, box_x2y2], dim=-1)

            # Combine all outputs into final feature map
            output_map = torch.cat(
                [
                    boxes,  # [x1, y1, x2, y2] - 4 dims
                    obj_conf,  # objectness - 1 dim
                    cls_conf * obj_conf,  # class scores - num_classes dims
                ],
                dim=-1,
            )

            # Reshape to [batch, h*w, channels]
            output_map = output_map.reshape(batch_size, -1, 5 + self.num_classes)
            all_outputs.append(output_map)

        # Concatenate outputs from all levels
        decoded_outputs = torch.cat(all_outputs, dim=1)
        return decoded_outputs

    def _filter_detections(self, decoded_outputs):
        """
        Filter detections based on confidence threshold and select top-k.
        """
        batch_size, max_detections = decoded_outputs.shape[0:2]

        # Initialize output tensors
        filtered_boxes = torch.zeros((batch_size, max_detections, 4))
        filtered_scores = torch.zeros((batch_size, max_detections))
        filtered_classes = torch.zeros((batch_size, max_detections), dtype=torch.int64)
        valid_detections = torch.zeros((batch_size,), dtype=torch.int32)

        # Process each image in batch
        for b in range(batch_size):
            boxes = decoded_outputs[b, :, :4]
            class_scores = decoded_outputs[b, :, 5 : 5 + self.num_classes]

            # Get highest class score and corresponding class index
            max_scores, class_ids = torch.max(class_scores, dim=1)

            # Filter by detection threshold
            score_mask = max_scores > self.detect_threshold

            # Apply mask by setting scores below threshold to -1 (to exclude in topk)
            masked_scores = torch.where(score_mask, max_scores, -1)

            # Select top-k detections
            topk_scores, topk_indices = torch.topk(masked_scores, k=max_detections, largest=True)
            topk_boxes = boxes[topk_indices]
            topk_classes = class_ids[topk_indices]

            # Count valid detections (those with positive scores)
            valid_mask = topk_scores > 0
            valid_count = torch.sum(valid_mask.int())

            # Store results
            filtered_boxes[b] = topk_boxes
            filtered_scores[b] = torch.where(valid_mask, topk_scores, torch.zeros_like(topk_scores))
            filtered_classes[b] = topk_classes
            valid_detections[b] = valid_count

        return filtered_boxes, filtered_scores, filtered_classes, valid_detections


def export_onnx_model(model, dummy_input, output_path, input_name, output_names, dynamic_axes):
    """
    Export PyTorch model to ONNX format.
    """
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"  Model has been saved to {output_path}")
    except Exception as e:
        print(f"  Error during model export: {str(e)}")
        sys.exit(1)


def simplify_onnx_model(input_path, output_path, max_iterations=5, min_size_improvement=0.01):
    """
    Simplify an ONNX model to reduce complexity and improve inference speed.
    This function applies iterative simplification until model size stabilizes,
    ensuring at least one simplification pass is always performed.
    """
    try:
        # Load the initial model
        best_model = onnx.load(input_path)
        original_size = os.path.getsize(input_path)
        best_size = original_size

        print(f"Starting model simplification. Original size: {original_size/1024:.2f} KB")

        # Iteratively simplify the model until size stabilizes
        for iteration in range(max_iterations):
            # Simplify the current best model
            simplified_model, check = simplify(
                best_model, perform_optimization=True, skip_fuse_bn=False, skip_shape_inference=False
            )

            if not check:
                if iteration == 0:
                    print(f"Failed to simplify model from {input_path}")
                    return False
                else:
                    # We already have a simplified model from a previous iteration
                    break

            # Save the model temporarily to check its size
            temp_path = output_path + ".temp"
            onnx.save(simplified_model, temp_path)
            current_size = os.path.getsize(temp_path)

            # Calculate improvement
            size_improvement = (best_size - current_size) / best_size

            # Different messaging based on iteration number
            if iteration == 0:
                print(
                    f"  Iteration {iteration+1}: Size = {current_size/1024:.2f} KB, "
                    f"Improvement = {size_improvement*100:.2f}%"
                )
                # If first iteration has low improvement, inform that we'll still try one more
                if size_improvement < min_size_improvement:
                    print(
                        f"  First iteration improvement below threshold ({min_size_improvement*100}%), "
                        f"but continuing to next iteration to ensure at least one complete simplification."
                    )
            else:
                print(
                    f"  Iteration {iteration+1}: Size = {current_size/1024:.2f} KB, "
                    f"Improvement = {size_improvement*100:.2f}%"
                )
                # For non-first iterations, check if we should stop
                if size_improvement < min_size_improvement:
                    print(f"  Size improvement below threshold ({min_size_improvement*100}%). Stopping.")
                    break

            # Update best model if this one is smaller
            if current_size < best_size:
                best_model = simplified_model
                best_size = current_size

        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Save the best model
        onnx.save(best_model, output_path)

        # Report total improvement
        total_improvement = (original_size - best_size) / original_size
        print(
            f"  Simplification complete. Final size: {best_size/1024:.2f} KB, "
            f"Total reduction: {total_improvement*100:.2f}%"
        )

        return True

    except Exception as e:
        print(f"  Error during model simplification: {str(e)}")
        return False


def prepare_dynamic_axes(input_blob_name, output_blob_names, with_postprocessing):
    """
    Prepare dynamic axes configuration for ONNX export.
    """
    dynamic_axes = {
        input_blob_name: {0: "batch_size", 2: "height", 3: "width"},
    }

    if not with_postprocessing:
        # For raw model outputs, dimensions depend on input size
        for output_name in output_blob_names:
            dynamic_axes[output_name] = {0: "batch_size", 2: "height", 3: "width"}
    else:
        # For post-processed outputs, dimensions are fixed except batch and detections
        dynamic_axes["boxes"] = {0: "batch_size", 1: "num_detections"}
        dynamic_axes["scores"] = {0: "batch_size", 1: "num_detections"}
        dynamic_axes["class_ids"] = {0: "batch_size", 1: "num_detections"}
        dynamic_axes["valid_detections"] = {0: "batch_size"}

    return dynamic_axes


def export_model_with_precision(
    model, config, save_dir, input_blob_name, output_blob_names, dynamic_axes, suffix, use_fp16=False
):
    """
    Export model with specified precision (FP32 or FP16).
    Only the simplified version is kept as the final output.
    """
    input_height = config["datasets"]["input_height"]
    input_width = config["datasets"]["input_width"]

    # Create precision-specific path and suffix
    precision_suffix = "_fp16" if use_fp16 else ""
    precision_name = "FP16" if use_fp16 else "FP32"

    # Temporary path for the initial export
    temp_model_path = os.path.join(save_dir, f"model{suffix}{precision_suffix}_temp.onnx")

    # Final path for the simplified model
    final_model_path = os.path.join(save_dir, f"model{suffix}{precision_suffix}.onnx")

    print(f"\nExporting {precision_name} model:")

    # Export the model with specified precision
    if use_fp16:
        with torch.autocast(config["onnx"]["device"], dtype=torch.float16):
            dummy_input = torch.randn((1, 3, input_height, input_width))
            export_onnx_model(model, dummy_input, temp_model_path, input_blob_name, output_blob_names, dynamic_axes)
    else:
        dummy_input = torch.randn((1, 3, input_height, input_width))
        export_onnx_model(model, dummy_input, temp_model_path, input_blob_name, output_blob_names, dynamic_axes)

    # Simplify the exported model
    print(f"Simplifying {precision_name} model:")
    success = simplify_onnx_model(temp_model_path, final_model_path)

    # Delete temporary model
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
        print(f"Deleted temporary unsimplified model: {temp_model_path}")

    if success:
        print(f"{precision_name} model exported and simplified successfully: {final_model_path}")
        return final_model_path
    else:
        print(f"WARNING: Simplification failed for {precision_name} model. Using original export.")
        # If simplification failed but we have an original model, rename it to the final path
        if os.path.exists(temp_model_path):
            os.rename(temp_model_path, final_model_path)
            print(f"Renamed original model to: {final_model_path}")
            return final_model_path

        print(f"ERROR: No valid model available for {precision_name}")
        return None


def main(args):
    """
    Main function to convert PyTorch model to ONNX format.
    """
    # Load configuration from YAML file
    config = utils.load_config(args.config_path)

    # Create and load the base model
    base_model = create_model(config["model"]["type"], config["model"]["num_classes"])
    model_path = os.path.join(config["training"]["save_dir"], "train", "best.pt")
    base_model.load_state_dict(
        torch.load(model_path, weights_only=False, map_location=torch.device(config["onnx"]["device"]))["model"]
    )
    base_model.eval()

    # Determine if post-processing is enabled
    with_postprocessing = config["onnx"]["postprocessing"]

    # Prepare model with or without post-processing
    if with_postprocessing:
        model = ObjectDetectionPostProcessor(base_model, config)
        output_blob_names = ["boxes", "scores", "class_ids", "valid_detections"]
    else:
        model = base_model
        output_blob_names = config["onnx"]["output_blob_name"]

    model.eval()

    # Create output directory
    save_dir = os.path.join(config["onnx"]["save_dir"], "onnx")
    os.makedirs(save_dir, exist_ok=True)

    # Get input tensor name
    input_blob_name = config["onnx"]["input_blob_name"]

    # Prepare dynamic axes configuration
    dynamic_axes = prepare_dynamic_axes(input_blob_name, output_blob_names, with_postprocessing)

    # Determine model filename suffix
    suffix = "_with_postprocessing" if with_postprocessing else ""

    print("\n" + "=" * 60)
    print("Starting PyTorch to ONNX conversion")
    print("=" * 60)

    # Track exported models for summary
    exported_models = []

    # Export FP32 model
    fp32_model_path = export_model_with_precision(
        model, config, save_dir, input_blob_name, output_blob_names, dynamic_axes, suffix, use_fp16=False
    )
    if fp32_model_path:
        exported_models.append(("FP32", fp32_model_path))

    # Export FP16 model
    fp16_model_path = export_model_with_precision(
        model, config, save_dir, input_blob_name, output_blob_names, dynamic_axes, suffix, use_fp16=True
    )
    if fp16_model_path:
        exported_models.append(("FP16", fp16_model_path))

    # Print summary of exported models
    print("\n" + "=" * 60)
    print("ONNX Export Summary")
    print("=" * 60)

    if exported_models:
        print("\nSuccessfully exported models:")
        for precision, path in exported_models:
            model_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
            print(f"  - {precision}: {path} ({model_size:.2f} MB)")
    else:
        print("\nWARNING: No models were successfully exported!")

    print("\nConversion complete!")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from PyTorch model to ONNX model")
    parser.add_argument(
        "--config_path", type=str, default="config/config.yml", help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    main(args)
