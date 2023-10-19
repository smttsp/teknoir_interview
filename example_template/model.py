import cv2
import numpy

from .utils import add_bbox_around_pred
engine_shape = (700, 1250)


def find_engine_in_frame(frame, engine):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(frame_gray, engine, cv2.TM_CCOEFF_NORMED)

    mx = numpy.amax(result)
    match_positions = []
    threshold = 0.4
    if mx > threshold:
        loc = numpy.where(result == mx)

        for pt in zip(*loc):
            center_x = pt[0] + engine_shape[1] // 2
            center_y = pt[1] + engine_shape[0] // 2
            match_positions.append((center_x, center_y))

    return match_positions


def find_persons_in_frame(results, pred_classes=("person")):
    persons = []
    for idx, row in results.iterrows():
        classname = row["name"]
        if classname not in pred_classes:
            continue

        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])

        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2

        persons.append((x_center, y_center))

    return persons


def get_pred(model, frame):
    results = model(frame)

    return results.pandas().xyxy[0]


def get_object_centers(model, frame, engine):
    results = get_pred(model, frame)

    persons = find_persons_in_frame(results)
    match_positions = find_engine_in_frame(frame, engine)
    persons.extend(match_positions)
    return persons
