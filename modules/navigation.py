from modules.caption import generate_caption
from modules.detection import detect_objects

def navigation_description(image_path):

    caption = generate_caption(image_path)
    objects, annotated_image = detect_objects(image_path)
    object_text = ", ".join(objects)
    important = ["person", "car", "bus", "bicycle", "motorcycle"]

    warnings = []

    for obj in objects:
        if obj in important:
            warnings.append(obj)

    if warnings:
        warning_text = "Warning: " + ", ".join(warnings) + " detected ahead."
    else:
        warning_text = "Path appears clear."

    final_output = f"{warning_text}\n\nDetected objects: {object_text}\nScene: {caption}"

    return final_output, annotated_image