"""
Webcam-based behavioral tracking for AX-CPT task
Uses MediaPipe for face mesh detection and tracking
Tier 1: Blinks and head pose stability
Extensible for Tier 2/3: Gaze tracking, facial expressions
"""
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import os
import urllib.request
import glob

# MediaPipe imports (0.10+ API)
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


@dataclass
class FrameMetrics:
    """Metrics extracted from a single frame."""
    timestamp: float
    trial_index: int
    
    # Head pose (position and orientation)
    head_x: float
    head_y: float
    head_z: float  # Distance from camera
    head_pitch: float  # Up/down rotation
    head_yaw: float    # Left/right rotation
    head_roll: float   # Tilt rotation
    
    # Blink detection
    left_eye_aspect_ratio: float
    right_eye_aspect_ratio: float
    is_blinking: bool
    
    # Placeholder for future Tier 2/3 metrics
    # gaze_x: Optional[float] = None
    # gaze_y: Optional[float] = None
    # pupil_diameter: Optional[float] = None
    # looking_away: Optional[bool] = None


class WebcamTracker:
    """Handles webcam capture and behavioral tracking."""

    # Eye aspect ratio threshold for blink detection
    EAR_THRESHOLD = 0.21

    # MediaPipe face mesh landmark indices for eyes
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # Model file URL and path
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = "face_landmarker.task"

    def __init__(self, enabled: bool = True, camera_index: int = 0,
                 capture_fps: int = 30, jpeg_quality: int = 85):
        """
        Initialize webcam tracker.

        Args:
            enabled: Whether tracking is enabled
            camera_index: Camera device index (0 for default webcam)
            capture_fps: Frame rate for capturing frames (default: 30)
            jpeg_quality: JPEG compression quality 0-100 (default: 85)
        """
        self.enabled = enabled
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.face_landmarker = None

        # Capture settings
        self.capture_fps = capture_fps
        self.jpeg_quality = jpeg_quality
        self.frame_counter = 0  # For frame rate limiting

        # Temp directory for raw frames
        self.frames_temp_dir: Optional[str] = None
        self.captured_frame_paths: List[Dict[str, Any]] = []  # Metadata for captured frames

        # Frame buffer for current trial (used during post-processing)
        self.frame_buffer: List[FrameMetrics] = []
        self.current_trial_index: int = -1

        if self.enabled:
            self._download_model_if_needed()
            self._initialize_camera()
            # Don't initialize MediaPipe yet - will do during post-processing
    
    def _download_model_if_needed(self):
        """Download MediaPipe face landmarker model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading face landmarker model...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                print(f"✓ Model downloaded to {self.MODEL_PATH}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Tracking will be disabled.")
                self.enabled = False

    def _initialize_camera(self):
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Warning: Could not open webcam. Tracking disabled.")
            self.enabled = False
            return

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("Webcam initialized successfully")

    def _initialize_mediapipe(self):
        """Initialize MediaPipe Face Landmarker (0.10+ API)."""
        try:
            base_options = mp_tasks.BaseOptions(
                model_asset_path=self.MODEL_PATH
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe Face Landmarker initialized")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            print("Tracking will be disabled.")
            self.enabled = False

    def set_temp_directory(self, session_dir: str):
        """
        Set up temporary directory for storing raw frames.

        Args:
            session_dir: Session directory path (e.g., 'results/2026-03-04T19-40-25')
        """
        if not self.enabled:
            return

        self.frames_temp_dir = f"{session_dir}/temp_frames"
        os.makedirs(self.frames_temp_dir, exist_ok=True)
        print(f"Temp frames directory created: {self.frames_temp_dir}")

    def capture_frame_to_disk(self, trial_index: int) -> Optional[Dict[str, Any]]:
        """
        Capture frame from camera and save to disk (no processing).
        Called during test execution at game loop rate (60 FPS).

        Args:
            trial_index: Current trial index

        Returns:
            Dictionary with frame metadata or None if skipped/failed
        """
        if not self.enabled or self.cap is None or self.frames_temp_dir is None:
            return None

        # Frame rate limiting: only capture every Nth frame
        # E.g., if capture_fps=30 and game loop is 60 FPS, capture every 2nd frame
        self.frame_counter += 1
        skip_interval = 60 // self.capture_fps  # Assumes 60 FPS game loop
        if self.frame_counter % skip_interval != 0:
            return None  # Skip this frame

        # Capture frame from camera
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Generate unique filename
        timestamp = time.perf_counter()
        frame_filename = f"{self.frames_temp_dir}/frame_{timestamp:.6f}_{trial_index}.jpg"

        # Save frame to disk as JPEG
        cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        # Store metadata
        frame_metadata = {
            'timestamp': timestamp,
            'trial_index': trial_index,
            'frame_path': frame_filename
        }
        self.captured_frame_paths.append(frame_metadata)

        return frame_metadata

    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Where p1-p6 are eye landmarks in order
        
        Args:
            eye_landmarks: Array of 6 eye landmark coordinates
            
        Returns:
            Eye aspect ratio (lower values indicate closed eye)
        """
        # Vertical distances
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _estimate_head_pose(self, landmarks, image_shape) -> Dict[str, float]:
        """
        Estimate head pose (position and orientation).

        Args:
            landmarks: MediaPipe face landmarks (list of NormalizedLandmark objects)
            image_shape: Shape of the input image (height, width, channels)

        Returns:
            Dictionary with head pose metrics
        """
        # Use key facial landmarks for pose estimation
        # Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
        landmark_indices = [1, 152, 33, 263, 61, 291]

        # 3D model points (approximate facial landmarks in mm)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)

        # 2D image points from landmarks
        # landmarks is a list, not an object with .landmark attribute
        image_points = np.array([
            (landmarks[idx].x * image_shape[1],
             landmarks[idx].y * image_shape[0])
            for idx in landmark_indices
        ], dtype=np.float64)
        
        # Camera internals (approximate)
        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                'head_x': 0.0, 'head_y': 0.0, 'head_z': 0.0,
                'head_pitch': 0.0, 'head_yaw': 0.0, 'head_roll': 0.0
            }

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Extract Euler angles (pitch, yaw, roll) from rotation matrix
        # Using the convention: R = Rz(roll) * Ry(yaw) * Rx(pitch)
        pitch = np.arctan2(-rotation_mat[2][0],
                          np.sqrt(rotation_mat[2][1]**2 + rotation_mat[2][2]**2))
        yaw = np.arctan2(rotation_mat[1][0], rotation_mat[0][0])
        roll = np.arctan2(rotation_mat[2][1], rotation_mat[2][2])

        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        # Translation vector gives position (normalized)
        head_x = float(translation_vec[0][0]) / 1000.0  # Normalize to reasonable scale
        head_y = float(translation_vec[1][0]) / 1000.0
        head_z = float(translation_vec[2][0]) / 1000.0

        return {
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'head_pitch': pitch_deg,
            'head_yaw': yaw_deg,
            'head_roll': roll_deg
        }

    def _process_single_frame(self, frame: np.ndarray, timestamp: float,
                             trial_index: int) -> Optional[FrameMetrics]:
        """
        Process a single frame and extract metrics.
        Called during post-processing after test completes.

        Args:
            frame: BGR image from cv2.imread()
            timestamp: Frame timestamp
            trial_index: Trial index

        Returns:
            FrameMetrics object or None if processing failed
        """
        if not self.enabled or self.face_landmarker is None:
            return None

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Process with MediaPipe Face Landmarker
        detection_result = self.face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            # No face detected
            return None

        # Get first face (we only track one person)
        face_landmarks = detection_result.face_landmarks[0]

        # Extract eye landmarks for blink detection
        h, w, _ = frame.shape
        landmarks_array = np.array([
            [lm.x * w, lm.y * h] for lm in face_landmarks
        ])

        left_eye = landmarks_array[self.LEFT_EYE_INDICES]
        right_eye = landmarks_array[self.RIGHT_EYE_INDICES]

        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)

        # Determine if blinking (both eyes below threshold)
        is_blinking = (left_ear + right_ear) / 2.0 < self.EAR_THRESHOLD

        # Estimate head pose
        head_pose = self._estimate_head_pose(face_landmarks, frame.shape)

        # Create metrics object
        metrics = FrameMetrics(
            timestamp=timestamp,  # Use provided timestamp
            trial_index=trial_index,
            head_x=head_pose['head_x'],
            head_y=head_pose['head_y'],
            head_z=head_pose['head_z'],
            head_pitch=head_pose['head_pitch'],
            head_yaw=head_pose['head_yaw'],
            head_roll=head_pose['head_roll'],
            left_eye_aspect_ratio=left_ear,
            right_eye_aspect_ratio=right_ear,
            is_blinking=is_blinking
        )

        return metrics

    def initialize_mediapipe_for_processing(self):
        """
        Initialize MediaPipe for post-processing.
        Called once before processing all captured frames.
        """
        if not self.enabled:
            return

        if self.face_landmarker is None:
            self._initialize_mediapipe()
            print("MediaPipe initialized for post-processing")

    def process_all_captured_frames(self, progress_callback=None) -> List[FrameMetrics]:
        """
        Process all captured frames from disk.
        Called once after test completes.

        Args:
            progress_callback: Optional callback function(current, total, progress_fraction)
                              called periodically to report progress

        Returns:
            List of all FrameMetrics objects
        """
        if not self.enabled or not self.captured_frame_paths:
            print("No frames to process")
            return []

        # Initialize MediaPipe if not already done
        self.initialize_mediapipe_for_processing()

        print(f"\n{'='*60}")
        print(f"Processing {len(self.captured_frame_paths)} captured frames...")
        print(f"{'='*60}\n")

        all_metrics = []
        total_frames = len(self.captured_frame_paths)

        for i, frame_metadata in enumerate(self.captured_frame_paths):
            # Load frame from disk
            frame = cv2.imread(frame_metadata['frame_path'])

            if frame is None:
                print(f"Warning: Could not load frame {frame_metadata['frame_path']}")
                continue

            # Process frame
            metrics = self._process_single_frame(
                frame=frame,
                timestamp=frame_metadata['timestamp'],
                trial_index=frame_metadata['trial_index']
            )

            if metrics:
                all_metrics.append(metrics)

            # Call progress callback if provided
            if progress_callback:
                progress_fraction = (i + 1) / total_frames
                progress_callback(i + 1, total_frames, progress_fraction)

            # Progress indicator (every 100 frames) - only if no callback
            if not progress_callback and (i + 1) % 100 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"  Processed {i + 1}/{total_frames} frames ({progress:.1f}%)")

        print(f"\n{'='*60}")
        print(f"Processing complete: {len(all_metrics)} frames successfully processed")
        print(f"{'='*60}\n")

        return all_metrics

    def cleanup_temp_frames(self):
        """
        Delete all temporary frame files and directory.
        Called after post-processing is complete.
        """
        if not self.frames_temp_dir or not os.path.exists(self.frames_temp_dir):
            return

        try:
            # Delete all frame files
            frame_files = glob.glob(f"{self.frames_temp_dir}/*.jpg")
            for file_path in frame_files:
                os.remove(file_path)

            # Delete directory
            os.rmdir(self.frames_temp_dir)
            print(f"Temp frames cleaned up: {len(frame_files)} files deleted")
        except Exception as e:
            print(f"Warning: Could not clean up temp frames: {e}")

    def get_capture_stats(self) -> Dict[str, Any]:
        """
        Get statistics about captured frames.

        Returns:
            Dictionary with frame count, estimated size, etc.
        """
        if not self.captured_frame_paths:
            return {
                'total_frames': 0,
                'estimated_size_mb': 0.0
            }

        # Estimate size (75 KB per frame average)
        estimated_size_mb = len(self.captured_frame_paths) * 0.075

        return {
            'total_frames': len(self.captured_frame_paths),
            'estimated_size_mb': estimated_size_mb,
            'capture_fps': self.capture_fps,
            'jpeg_quality': self.jpeg_quality
        }

    def start_trial(self, trial_index: int):
        """Mark the start of a new trial."""
        self.current_trial_index = trial_index
        self.frame_buffer.clear()

    def end_trial(self) -> Dict[str, Any]:
        """
        End current trial and compute trial-level metrics.

        Returns:
            Dictionary of aggregated trial metrics
        """
        if not self.frame_buffer:
            return self._get_empty_trial_metrics()

        # Count blinks (transitions from not blinking to blinking)
        blink_count = 0
        for i in range(1, len(self.frame_buffer)):
            if self.frame_buffer[i].is_blinking and not self.frame_buffer[i-1].is_blinking:
                blink_count += 1

        # Calculate trial duration
        if len(self.frame_buffer) > 1:
            trial_duration = (self.frame_buffer[-1].timestamp -
                            self.frame_buffer[0].timestamp)
            blink_rate = blink_count / trial_duration if trial_duration > 0 else 0.0
        else:
            blink_rate = 0.0

        # Head position statistics
        head_z_values = [f.head_z for f in self.frame_buffer]
        mean_head_distance = np.mean(head_z_values)

        # Head movement variance (used for posture consistency metric)
        head_x_values = [f.head_x for f in self.frame_buffer]
        head_y_values = [f.head_y for f in self.frame_buffer]
        head_movement_variance = np.var(head_x_values) + np.var(head_y_values)

        # Looking away detection (head turned too far)
        looking_away_count = sum(
            1 for f in self.frame_buffer
            if abs(f.head_yaw) > 30 or abs(f.head_pitch) > 30
        )

        return {
            'blink_count': blink_count,
            'blink_rate': blink_rate,
            'mean_head_distance': mean_head_distance,
            'head_movement_variance': head_movement_variance,
            'looking_away_count': looking_away_count,
            'frames_tracked': len(self.frame_buffer)
        }

    def _get_empty_trial_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no tracking data available."""
        return {
            'blink_count': 0,
            'blink_rate': 0.0,
            'mean_head_distance': 0.0,
            'head_movement_variance': 0.0,
            'looking_away_count': 0,
            'frames_tracked': 0
        }

    def get_all_frame_data(self) -> List[FrameMetrics]:
        """Get all frame data collected so far."""
        return self.frame_buffer.copy()

    def release(self):
        """Release webcam and cleanup resources."""
        if self.cap is not None:
            self.cap.release()
        if self.face_landmarker is not None:
            self.face_landmarker.close()

        # Clean up temp frames if they still exist
        self.cleanup_temp_frames()

        print("Webcam tracker released")

