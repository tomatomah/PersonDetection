def load_datasets(config_datasets):
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    for config_dataset in config_datasets["info"]:
        if config_dataset["name"] == "coco":
            from load_coco_dataset import load_dataset

            train_coco_data, train_coco_labels = load_dataset(config_dataset, "train", config_datasets["class_to_id"])
            val_coco_data, val_coco_labels = load_dataset(config_dataset, "val", config_datasets["class_to_id"])
            train_data.extend(train_coco_data)
            train_labels.extend(train_coco_labels)
            val_data.extend(val_coco_data)
            val_labels.extend(val_coco_labels)

        if config_dataset["name"] == "custom":
            from load_custom_dataset import load_dataset

            train_custom_data, train_custom_labels = load_dataset(config_dataset, config_datasets["class_to_id"])
            train_data.extend(train_custom_data)
            train_labels.extend(train_custom_labels)

    return train_data, train_labels, val_data, val_labels
