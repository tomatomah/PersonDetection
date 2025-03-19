import custom_model

def create_model(model_size: str, num_classes: int):
    if model_size == "small":
        return custom_model.yolo(num_classes)
    elif model_size == "medium":
        return None
    elif model_size == "large":
        return None
    else:
        return None
