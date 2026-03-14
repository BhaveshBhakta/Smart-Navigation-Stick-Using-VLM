from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    results = model(image_path, verbose=False)
    objects = []

    # image with bounding boxes
    annotated_image = results[0].plot()

    for r in results:

        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            x1, y1, x2, y2 = box.xyxy[0]

            width = x2 - x1
            height = y2 - y1

            area = width * height

            # simple distance estimation
            if area > 150000:
                distance = "about 1 meter away"
            elif area > 50000:
                distance = "about 3 meters away"
            else:
                distance = "far away"

            object_text = f"{label} ({distance})"

            if object_text not in objects:
                objects.append(object_text)

    return objects, annotated_image