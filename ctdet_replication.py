import os


def main():
    COCO_ROOT = "/data/Datasets/MSCOCO/"
    COCO_TRAIN_IMG = os.path.join(COCO_ROOT, "train2017")
    COCO_TRAIN_ANNO = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")

    data = torchvision.datasets.CocoDetection(
        root=COCO_TRAIN_IMG,
        annFile=COCO_TRAIN_ANNO,
        transform=torchvision.transforms.ToTensor(),
    )

    loader = DataLoader(dataset=data, batch_size=1)

    for i, sample in enumerate(loader, start=1):
        print(f"Sample {i}")
        print("End")


if __name__ == "__main__":
    main()
