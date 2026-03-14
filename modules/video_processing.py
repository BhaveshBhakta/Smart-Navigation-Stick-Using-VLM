import cv2
import os
from modules.caption import generate_caption
from modules.tts import text_to_speech

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    captions = []
    os.makedirs("temp_frames", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process every 30th frame
        if frame_count % 30 == 0:
            frame_path = f"temp_frames/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            caption = generate_caption(frame_path)
            captions.append(caption)
            print("Frame", frame_count, ":", caption)
        frame_count += 1

    cap.release()
    final_caption = ". ".join(captions)
    audio_path = text_to_speech(final_caption)
    return final_caption, audio_path