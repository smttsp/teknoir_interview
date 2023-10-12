import torch


def get_pred(model, frame):
    # Load the YOLOv5 model from Torch Hub

    # Load a sample image (replace 'example.jpg' with your image path)
    # img_path = 'example.jpg'
    # img = torch.from_numpy(torch.load(img_path)).unsqueeze(0).to('cuda')

    # Get predictions for the image
    results = model(frame)

    return results.pandas().xyxy[0]
    # Print the results
    # results.print()
    #
    # # Access the predictions
    # predictions = results.pred[0]
    #
    # # `predictions` is a tensor containing information about detected objects
    # # For example, to get the predicted classes, boxes, and scores:
    # predicted_classes = predictions[:, 5].cpu().numpy().astype(int)
    # predicted_boxes = predictions[:, :4].cpu().numpy()
    # predicted_scores = predictions[:, 4].cpu().numpy()
    #
    # # Now you can work with predicted_classes, predicted_boxes, and predicted_scores
