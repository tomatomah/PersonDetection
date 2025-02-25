def load_datasets(datasets_info):
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    for dataset in datasets_info:
        if dataset["name"] == "coco":
            from load_coco_dataset import load_dataset

            train_coco_data, train_coco_labels = load_dataset(dataset, "train")
            val_coco_data, val_coco_labels = load_dataset(dataset, "val")
            train_data.extend(train_coco_data)
            train_labels.extend(train_coco_labels)
            val_data.extend(val_coco_data)
            val_labels.extend(val_coco_labels)

        if dataset["name"] == "custom_dataset":
            from load_custom_dataset import load_dataset

            train_custom_data, train_custom_labels, val_custom_data, val_custom_labels = load_dataset(dataset)
            train_data.extend(train_custom_data)
            train_labels.extend(train_custom_labels)
            val_data.extend(val_custom_data)
            val_labels.extend(val_custom_labels)

    return train_data, train_labels, val_data, val_labels
