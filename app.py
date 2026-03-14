import gradio as gr
from modules.caption import generate_caption
from modules.tts import text_to_speech
from modules.video_processing import process_video
from modules.webcam_processing import run_webcam
from modules.navigation import navigation_description
import cv2
import time

last_time = 0
last_caption = "Starting webcam..."


# IMAGE PROCESSING
def process_image(image):

    caption, annotated_image = navigation_description(image)

    audio = text_to_speech(caption)

    return annotated_image, caption, audio


# VIDEO PROCESSING
def process_video_file(video):

    video_path = video.name

    caption, audio = process_video(video_path)

    return caption, audio


# WEBCAM PROCESSING
def process_webcam(frame):
    global last_time, last_caption

    if frame is None:
        return "No frame detected", None

    current_time = time.time()

    # Only run BLIP every 3 seconds
    if current_time - last_time > 3:

        frame_path = "webcam_frame.jpg"

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_path, frame_rgb)

        caption = generate_caption(frame_path)

        audio_path = text_to_speech(caption)

        last_caption = caption
        last_time = current_time

        return caption, audio_path

    # Return previous caption while waiting
    return last_caption, None


# GRADIO INTERFACE
interface = gr.TabbedInterface(

    [

        # IMAGE TAB
        gr.Interface(
            fn=process_image,
            inputs=gr.Image(type="filepath"),
            outputs=[
                gr.Image(label="Detected Objects"),
                gr.Textbox(label="Navigation Description"),
                gr.Audio(label="Audio Guidance")
            ],
            title="Smart Navigation - Image"
        ),

        # VIDEO TAB
        gr.Interface(
            fn=process_video_file,
            inputs=gr.File(file_types=[".mp4", ".avi", ".mov", ".mkv"]),
            outputs=[
                gr.Textbox(label="Video Description"),
                gr.Audio(label="Audio Description")
            ],
            title="Video Captioning"
        ),

        # WEBCAM TAB
        gr.Interface(
            fn=process_webcam,
            inputs=gr.Image(type="numpy", streaming=True),
            outputs=[
                gr.Textbox(label="Live Caption"),
                gr.Audio(label="Audio Guidance")
            ],
            live=True,
            title="Live Webcam Navigation"
        )

    ],

    tab_names=["Image", "Video", "Webcam"]

)

interface.launch()