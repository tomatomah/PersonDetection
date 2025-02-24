import yolo


def create_model(model_size: str, num_classes: int):
    if model_size == "nano":
        return yolo.yolo_n(num_classes)
    elif model_size == "tiny":
        return yolo.yolo_t(num_classes)
    elif model_size == "small":
        return yolo.yolo_s(num_classes)
    elif model_size == "medium":
        return yolo.yolo_m(num_classes)
    elif model_size == "large":
        return yolo.yolo_l(num_classes)
    elif model_size == "xlarge":
        return yolo.yolo_x(num_classes)
    else:
        return yolo.yolo_n(num_classes)
