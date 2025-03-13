import argparse
import os

import torch
import utils
from models import create_model


def main(args):
    config = utils.load_config(args.config_path)
    model = create_model(config["model"]["type"], config["model"]["num_classes"])
    model_path = os.path.join(config["training"]["save_dir"], "train", "best.pt")
    model.load_state_dict(torch.load(model_path, weights_only=False)["model"])

    save_dir = os.path.join(config["onnx"]["save_dir"], "onnx")
    os.makedirs(save_dir, exist_ok=True)

    output_model_path = os.path.join(save_dir, "model.onnx")
    dummy_input = torch.randn((1, 3, config["datasets"]["input_height"], config["datasets"]["input_width"]))
    torch.onnx.export(
        model,
        dummy_input,
        output_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[config["onnx"]["input_blob_name"]],
        output_names=config["onnx"]["output_blob_name"],
        dynamic_axes={
            config["onnx"]["input_blob_name"]: {0: "batch_size", 2: "height", 3: "width"},
            config["onnx"]["output_blob_name"][0]: {0: "batch_size", 2: "height", 3: "width"},
            config["onnx"]["output_blob_name"][1]: {0: "batch_size", 2: "height", 3: "width"},
            config["onnx"]["output_blob_name"][2]: {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(f"FP32 model has been saved to {output_model_path}")

    output_model_path_fp16 = os.path.join(save_dir, "model_fp16.onnx")
    with torch.autocast(config["onnx"]["device"], dtype=torch.float16):
        dummy_input = torch.randn((1, 3, config["datasets"]["input_height"], config["datasets"]["input_width"]))
        torch.onnx.export(
            model,
            dummy_input,
            output_model_path_fp16,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[config["onnx"]["input_blob_name"]],
            output_names=config["onnx"]["output_blob_name"],
            dynamic_axes={
                config["onnx"]["input_blob_name"]: {0: "batch_size", 2: "height", 3: "width"},
                config["onnx"]["output_blob_name"][0]: {0: "batch_size", 2: "height", 3: "width"},
                config["onnx"]["output_blob_name"][1]: {0: "batch_size", 2: "height", 3: "width"},
                config["onnx"]["output_blob_name"][2]: {0: "batch_size", 2: "height", 3: "width"},
            },
        )
    print(f"FP16 model has been saved to {output_model_path_fp16}")

    print("Both FP32 and FP16 models have been converted from PyTorch to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from pytorch model to onnx model")
    parser.add_argument(
        "--config_path", type=str, default="config/config.yml", help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    main(args)
