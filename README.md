### Face Detection with Confidence Score and Audio Recording

This feature combines gaze estimation with face detection confidence scoring and automatic audio recording. When a person maintains consistent eye contact (high confidence score) for a specified duration, the system automatically records audio and transcribes it.

#### Features
- Real-time face detection with confidence scoring
- Automatic audio recording triggered by sustained eye contact
- Speech-to-text transcription using MLX Whisper
- Automatic note-taking with timestamp logging

#### Requirements
Additional dependencies for audio recording and transcription:
```bash
pip install requirements.txt
pip install pyaudio mlx-whisper
```

#### Usage
```bash
python face_detection_w_confidence_score.py --arch [arch_name] --gaze-weights [path_gaze_weights] --face-weights [face_det_weights] --camera-id [camera_id] --dataset [dataset_name]
```

#### Arguments
```
--arch            Model architecture (default: "resnet50")
--gaze-weights    Path to gaze estimation model weights
--face-weights    Path to face detection model weights (default: "weights/det_10g.onnx")
--output         Path to save output video file
--dataset        Dataset name (default: "gaze360")
--camera-id      Camera device ID (default: 0)
```

#### How it Works
1. The system continuously monitors face detection and gaze estimation
2. When confidence score exceeds 75% for 2 seconds:
   - Audio recording automatically starts
   - Recording continues for 5 seconds
   - Speech is transcribed and saved to:
     - Individual note files in the `notes` directory
     - Consolidated `all_notes.txt` file with timestamps
3. 7-second cooldown period between recordings
4. Press 'q' to quit the application

#### Output
- Individual note files: `notes/note_YYYY-MM-DD_HHMMSS.txt`
- Consolidated notes: `all_notes.txt`
- Real-time visualization with confidence scores and recording status
`
