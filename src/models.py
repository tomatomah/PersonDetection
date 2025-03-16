import yolox


def create_model(model_size: str, num_classes: int):
    if model_size == "nano":
        return yolox.yolox_n(num_classes)
    elif model_size == "tiny":
        return yolox.yolox_t(num_classes)
    elif model_size == "small":
        return yolox.yolox_s(num_classes)
    elif model_size == "medium":
        return yolox.yolox_m(num_classes)
    elif model_size == "large":
        return yolox.yolox_l(num_classes)
    elif model_size == "xlarge":
        return yolox.yolox_x(num_classes)
    else:
        return yolox.yolox_n(num_classes)
