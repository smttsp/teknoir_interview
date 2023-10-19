import cv2
import torch

from example_template.utils import add_bbox_around_pred
from example_template.model import get_object_centers
from example_template.segment_anything import (
    segmentation_main,
    read_segm_anything,
)


def video_processing_segmentation(
    model, segm_predictor, video_path, engine_img
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video playback finished.")
            break

        object_centers = get_object_centers(model, frame, engine_img)
        results = segmentation_main(segm_predictor, frame, object_centers)
        # Display the frame

        frame_p = add_bbox_around_pred(frame, results, pred_classes=("person"))
        cv2.imshow("Video", frame_p)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Video playback interrupted by user.")
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = torch.hub.load("ultralytics/yolov5:v6.0", "yolov5s")
    segm_predictor = read_segm_anything()

    video_path = "/users/samet/downloads/trimmed 1.mov"
    engine_img = cv2.imread("/users/samet/desktop/engine.png", cv2.IMREAD_GRAYSCALE)

    video_processing_segmentation(
        model, segm_predictor, video_path, engine_img
    )
    print("Done!")
