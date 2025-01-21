import cv2
import time
import logging
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import SCRFD
from config import data_config
from utils.helpers import get_model, draw_bbox, draw_gaze
import mlx_whisper
import pyaudio
import threading
from queue import Queue
import os
import datetime
import re

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation with camera input")
    parser.add_argument("--arch", type=str, default="resnet50", help="Model name")
    parser.add_argument("--gaze-weights", type=str, default="resnet50.pt", help="Path to gaze estimation model weights")
    parser.add_argument("--face-weights", type=str, default="weights/det_10g.onnx", help="Path to face detection model weights")
    parser.add_argument("--output", type=str, help="Path to save output video file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    args = parser.parse_args()

    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return args

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.notes_dir = "notes"
        self.notes_file = "all_notes.txt"
        self.recording_start_time = None
        self.RECORD_SECONDS = 5  # Set recording duration to 5 seconds

        if not os.path.exists(self.notes_dir):
            os.makedirs(self.notes_dir)

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.recording_start_time = time.time()
        threading.Thread(target=self._record_audio).start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if len(self.frames) > 0:
                audio_data = np.hstack(self.frames, dtype=np.float32) / 32768.0
                self.transcribe_and_process(audio_data)

    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=16000,
                       input=True,
                       frames_per_buffer=1024)

        while self.recording:
            if time.time() - self.recording_start_time >= self.RECORD_SECONDS:
                self.stop_recording()
                break

            data = stream.read(1024, exception_on_overflow=False)
            self.frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()
        p.terminate()

    def transcribe_and_process(self, audio_data):
        try:
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo="mlx-community/whisper-small-mlx"
            )
            transcribed_text = result['text'].strip()
            print(f"\nTranscribed: {transcribed_text}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save to individual file
            filename = f"note_{timestamp.replace(' ', '_').replace(':', '')}.txt"
            filepath = os.path.join(self.notes_dir, filename)
            with open(filepath, 'w') as f:
                f.write(transcribed_text)

            # Append to main notes file
            with open(self.notes_file, 'a') as f:
                f.write(f"\n[{timestamp}] {transcribed_text}")

            print(f"Note saved!")
            print(f"Individual file: {filepath}")
            print(f"Added to: {self.notes_file}\n")

        except Exception as e:
            print(f"Transcription error: {e}")

def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),  # Reduced resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0)

def main(params):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = SCRFD(model_path=params.face_weights)
    gaze_detector = get_model(params.arch, params.bins, inference_mode=True)
    state_dict = torch.load(params.gaze_weights, map_location=device)
    gaze_detector.load_state_dict(state_dict)
    gaze_detector.to(device)
    gaze_detector.eval()

    cap = cv2.VideoCapture(params.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        logging.error("Failed to open camera")
        return

    high_confidence_start = None
    locked_in = False
    audio_recorder = AudioRecorder()

    recording_cooldown = 0
    frame_counter = 0
    detection_interval = 1  # Detect faces every 5 frames

    try:
        with torch.no_grad():
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                if frame_counter % detection_interval == 0:
                    bboxes, keypoints = face_detector.detect(frame)
                else:
                    bboxes, keypoints = [], []

                for bbox, keypoint in zip(bboxes, keypoints):
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])

                    if x_min < 0 or y_min < 0 or x_max > frame.shape[1] or y_max > frame.shape[0]:
                        continue

                    face_img = frame[y_min:y_max, x_min:x_max]
                    if face_img.size == 0:
                        continue

                    face_input = pre_process(face_img).to(device)
                    pitch, yaw = gaze_detector(face_input)

                    pitch_predicted = F.softmax(pitch, dim=1)
                    yaw_predicted = F.softmax(yaw, dim=1)

                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                    pitch_predicted = np.radians(pitch_predicted.cpu())
                    yaw_predicted = np.radians(yaw_predicted.cpu())

                    draw_bbox(frame, bbox)
                    confidence_score = draw_gaze(frame, bbox, pitch_predicted, yaw_predicted)

                    if confidence_score > 75 and not audio_recorder.recording and recording_cooldown <= 0:
                        if high_confidence_start is None:
                            high_confidence_start = time.time()
                        elif time.time() - high_confidence_start > 2:
                            logging.info("Starting recording...")
                            audio_recorder.start_recording()
                            recording_cooldown = 7
                    elif confidence_score <= 75:
                        high_confidence_start = None

                    if audio_recorder.recording:
                        cv2.putText(frame, "RECORDING", (50, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if recording_cooldown > 0:
                    recording_cooldown -= 0.1

                cv2.imshow('Gaze Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                elapsed_time = time.time() - start_time
                sleep_time = max(0, 0.1 - elapsed_time)  # Target 10 FPS
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        if audio_recorder.recording:
            audio_recorder.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    main(args)
