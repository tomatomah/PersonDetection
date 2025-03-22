import custom_model


def create_model(model_size: str, num_classes: int):
    if model_size == "small":
        return custom_model.yolo_s(num_classes)
    elif model_size == "medium":
        return custom_model.yolo_m(num_classes)
    elif model_size == "large":
        return custom_model.yolo_l(num_classes)
    else:
        return custom_model.yolo_s(num_classes)
