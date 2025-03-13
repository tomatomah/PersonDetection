import argparse
import os
import sys

import onnx
import torch
import utils
from models import create_model
from onnxsim import simplify


def export_onnx_model(model, dummy_input, output_path, input_blob_name, output_blob_names, dynamic_axes):
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[input_blob_name],
            output_names=output_blob_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"Model has been saved to {output_path}")
    except Exception as e:
        print(f"Error during model export: {str(e)}")
        sys.exit(1)


def simplify_onnx_model(input_path, output_path):
    onnx_model = onnx.load(input_path)
    model_simplified, check = simplify(onnx_model)

    if check:
        onnx.save(model_simplified, output_path)
        print(f"Simplified model has been saved to {output_path}")
    else:
        print(f"Failed to simplify model from {input_path}")
        sys.exit(1)


def main(args):
    config = utils.load_config(args.config_path)

    model = create_model(config["model"]["type"], config["model"]["num_classes"])
    model_path = os.path.join(config["training"]["save_dir"], "train", "best.pt")
    model.load_state_dict(torch.load(model_path, weights_only=False)["model"])

    save_dir = os.path.join(config["onnx"]["save_dir"], "onnx")
    os.makedirs(save_dir, exist_ok=True)

    input_blob_name = config["onnx"]["input_blob_name"]
    output_blob_names = config["onnx"]["output_blob_name"]
    input_height = config["datasets"]["input_height"]
    input_width = config["datasets"]["input_width"]

    dynamic_axes = {
        input_blob_name: {0: "batch_size", 2: "height", 3: "width"},
    }
    for output_name in output_blob_names:
        dynamic_axes[output_name] = {0: "batch_size", 2: "height", 3: "width"}

    # FP32 model
    output_model_path = os.path.join(save_dir, "model.onnx")
    dummy_input = torch.randn((1, 3, input_height, input_width))
    export_onnx_model(model, dummy_input, output_model_path, input_blob_name, output_blob_names, dynamic_axes)

    output_model_path_simplified = os.path.join(save_dir, "model_sim.onnx")
    simplify_onnx_model(output_model_path, output_model_path_simplified)
    print("FP32 model exported")

    # FP16 model
    output_model_path_fp16 = os.path.join(save_dir, "model_fp16.onnx")
    with torch.autocast(config["onnx"]["device"], dtype=torch.float16):
        dummy_input = torch.randn((1, 3, input_height, input_width))
        export_onnx_model(model, dummy_input, output_model_path_fp16, input_blob_name, output_blob_names, dynamic_axes)

    output_model_path_fp16_simplified = os.path.join(save_dir, "model_fp16_sim.onnx")
    simplify_onnx_model(output_model_path_fp16, output_model_path_fp16_simplified)
    print("FP16 model exported")

    print("All models have been converted from PyTorch to ONNX")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from pytorch model to onnx model")
    parser.add_argument(
        "--config_path", type=str, default="config/config.yml", help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    main(args)
