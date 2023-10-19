import cv2
from segment_anything import SamPredictor, sam_model_registry
from .utils import add_bbox_around_pred
import matplotlib.pyplot as plt
import numpy


def read_segm_anything(
    checkpoint="/Users/samet/Documents/models/segment_anything/sam_vit_h_4b8939.pth",
    model_type="vit_h",
):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    segm_predictor = SamPredictor(sam)

    return segm_predictor


def segmentation_main(segm_predictor, frame, object_center):
    segm_predictor.set_image(frame)
    masks, _, _ = segm_predictor.predict(
        point_coords=numpy.array(object_center),
        point_labels=numpy.array([0, 1]),
    )
    return masks
