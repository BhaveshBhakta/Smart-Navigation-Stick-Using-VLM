import cv2
import time
from modules.caption import generate_caption
from modules.tts import text_to_speech


def run_webcam():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")
    last_caption_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam Navigation", frame)
        current_time = time.time()
        
        # Generate caption every 5 seconds
        if current_time - last_caption_time > 5:
            frame_path = "webcam_frame.jpg"
            cv2.imwrite(frame_path, frame)
            caption = generate_caption(frame_path)
            print("Scene:", caption)
            text_to_speech(caption)
            last_caption_time = current_time
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()