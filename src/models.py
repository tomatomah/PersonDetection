import custom_model


def create_model(model_size: str, num_classes: int):
    if model_size == "nano":
        return custom_model.yolo_n(num_classes)
    elif model_size == "tiny":
        return custom_model.yolo_t(num_classes)
    elif model_size == "small":
        return custom_model.yolo_s(num_classes)
    elif model_size == "medium":
        return custom_model.yolo_m(num_classes)
    elif model_size == "large":
        return custom_model.yolo_l(num_classes)
    elif model_size == "xlarge":
        return custom_model.yolo_x(num_classes)
    else:
        return custom_model.yolo_s(num_classes)
