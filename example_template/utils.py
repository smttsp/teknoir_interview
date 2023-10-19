import cv2


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
