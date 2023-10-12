import cv2
import torch

from example_template.model import get_pred


def add_bbox_around_pred(frame, results, pred_classes=("person")):

    for idx, row in results.iterrows():
        classname = row["name"]
        if classname not in pred_classes:
            continue

        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return frame


def read_and_display_video(model, video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Loop through the frames and display them
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video is over
        if not ret:
            print("Video playback finished.")
            break

        results = get_pred(model, frame)
        # Display the frame

        frame_p = add_bbox_around_pred(frame, results, pred_classes=("person"))
        cv2.imshow('Video', frame_p)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback interrupted by user.")
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s')

    video_path = "/users/samet/downloads/trimmed 1.mov"
    read_and_display_video(model, video_path)
    print("Done!")