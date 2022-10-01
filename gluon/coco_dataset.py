import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
import torchvision  # type: ignore[import]
from PIL.Image import Image  # type: ignore[import]
from pydantic import BaseModel
from torch.utils.data import DataLoader

from gluon.interfaces.data import GluonDataPoint, GluonInputs, GluonLabels


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    bbox: List[float]
    coco_category_id: int
    gluon_class_id: int
    iscrowd: bool

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def make_many(cls, list_of_annos: List[Dict[str, Any]]):
        return [cls(**anno) for anno in list_of_annos]


class COCOSample(BaseModel):
    id: int
    file_path: str
    annotations: List[COCOAnnotation]


class _COCOClassMapper:

    def __init__(self, coco_categories: List[dict], class_mapping: Dict[str, int]):
        self.coco_categories = {cat["id"]: cat for cat in coco_categories}
        self.class_mapping = class_mapping

    def transform(self, coco_category_id: int) -> Optional[int]:
        coco_category_descriptor = self.coco_categories[coco_category_id]
        category_name = coco_category_descriptor["name"]
        if category_name not in self.class_mapping:
            return None
        gluon_category_id = self.class_mapping[category_name]
        return gluon_category_id


class MSCOCODetection(torch.utils.data.Dataset[GluonDataPoint]):

    default_input_element_key = "image"
    default_coco_categories_json = "data/mscoco_2017_instances_categories.json"
    default_class_mapping_json = "data/mscoco_2017_instances_gluon_mapping.json"
    default_train_images_root = "/data/Datasets/MSCOCO/train2017"
    default_val_images_root = "/data/Datasets/MSCOCO/val2017"
    default_annotations_root = "/data/Datasets/MSCOCO/annotations"

    def __init__(
            self,
            images_root: str,
            annotation_file: str,
            class_mapping_json: str,
            coco_categories_json: str,
            filters: Optional[List[Callable[[COCOSample], bool]]] = None,
            transforms: Optional[List[Callable[[GluonDataPoint], GluonDataPoint]]] = None,
            gluon_input_element_key: Optional[str] = None,
    ):
        super().__init__()
        filters = filters or []
        self.transforms = transforms or []
        self.images_root = images_root
        self.data = json.load(open(annotation_file))
        self.index: List[COCOSample] = []
        self.class_mapper = _COCOClassMapper(
            coco_categories=json.load(open(coco_categories_json)),
            class_mapping=json.load(open(class_mapping_json)),
        )
        self.num_classes = len(self.class_mapper.class_mapping)
        self.input_element_key = gluon_input_element_key or self.default_input_element_key

        image_id_to_meta = {meta["id"]: meta for meta in self.data["images"]}
        image_id_to_annos: Dict[int, List[COCOAnnotation]] = defaultdict(list)
        for anno in self.data["annotations"]:
            image_w = image_id_to_meta[anno["image_id"]]["width"]
            image_h = image_id_to_meta[anno["image_id"]]["height"]
            if len(anno["bbox"]) != 4:  # Invalid box
                continue
            if anno["bbox"][2] < 2 or anno["bbox"][3] < 2:  # Dimension too small
                continue
            if anno["bbox"][0] + anno["bbox"][2] / 2 >= image_w:  # Center out of bounds
                continue
            if anno["bbox"][1] + anno["bbox"][3] / 2 >= image_h:  # Center out of bounds
                continue
            gluon_class_id = self.class_mapper.transform(anno["category_id"])
            if gluon_class_id is None:
                continue
            coco_anno = COCOAnnotation(
                id=anno["id"],
                image_id=anno["image_id"],
                bbox=anno["bbox"],
                coco_category_id=anno["category_id"],
                gluon_class_id=gluon_class_id,
                iscrowd=anno["iscrowd"],
            )
            image_id_to_annos[coco_anno.image_id].append(coco_anno)
        for image_id, coco_annos in image_id_to_annos.items():
            image_path = os.path.join(images_root, image_id_to_meta[image_id]["file_name"])
            coco_sample = COCOSample(id=image_id, file_path=image_path, annotations=coco_annos)
            if all(flt(coco_sample) for flt in filters):
                self.index.append(coco_sample)

    def __getitem__(self, index: int) -> GluonDataPoint:
        coco_sample = self.index[index]
        image_tensor = torchvision.io.read_image(str(coco_sample.file_path))
        image_tensor = image_tensor.float() / 255.
        if len(image_tensor.size()) == 3 and image_tensor.size(0) == 1:
            image_tensor = torch.cat([image_tensor]*3, dim=0)
        assert len(image_tensor) == 3, f"Shape: {image_tensor.size()}"

        spatial_dims = np.array(image_tensor.size()[1:])[::-1]
        box_corners = []
        class_ids = []
        for anno in coco_sample.annotations:
            if anno.iscrowd:
                continue
            if len(anno.bbox) == 0:
                continue

            class_ids.append(anno.gluon_class_id)
            box = np.array(anno.bbox)
            box_x0y0 = box[:2] / spatial_dims
            box_x1y1 = (box[:2] + box[2:]) / spatial_dims
            assert np.all(box_x0y0 < box_x1y1)
            keypoint_repr = np.concatenate([box_x0y0, box_x1y1], axis=0)

            box_corners.append(keypoint_repr)

        box_corners_xy = np.array(box_corners).astype(np.float32)[:, ::-1]
        result = GluonDataPoint(
            metadata={"image_id": coco_sample.id},
            inputs=GluonInputs(elements={self.input_element_key: image_tensor}),
            labels=GluonLabels(
                box_corners=box_corners_xy,
                class_ids=np.array(class_ids, dtype=np.int64),
                scores=None,
            ),
        )
        for transform in self.transforms:
            result = transform(result)

        return result

    def get_loader(self, batch_size, shuffle, num_workers=0):
        loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=no_collate,
            prefetch_factor=2,
        )
        return loader

    def __len__(self):
        return len(self.index)


def no_collate(batch):
    return batch

