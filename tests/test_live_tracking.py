#!/usr/bin/env python3
"""
Live webcam tracking test - displays real-time metrics with readable update rate.
Combines continuous display updates (Option 1) with event notifications (Option 2).
"""
import time
import sys
import os

# Add parent directory to path so we can import webcam_tracker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webcam_tracker import WebcamTracker


def clear_lines(n):
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write('\033[F')  # Move cursor up
        sys.stdout.write('\033[K')  # Clear line


def test_live_tracking():
    """Display live tracking metrics with readable update rate + event notifications."""
    print("=" * 70)
    print("LIVE WEBCAM TRACKING TEST")
    print("=" * 70)
    print("\nInitializing webcam...")
    
    tracker = WebcamTracker(enabled=True, camera_index=0)
    
    if not tracker.enabled:
        print("ERROR: Webcam could not be initialized!")
        return
    
    print("✓ Webcam initialized")
    print("\nStarting live tracking... (Press Ctrl+C to stop)")
    print("Display updates every 0.5 seconds")
    print("Events (blinks, looking away) are shown immediately")
    print("-" * 70)
    time.sleep(1)
    
    tracker.start_trial(0)
    
    # Counters
    frame_count = 0
    blink_count = 0
    looking_away_count = 0
    
    # State tracking
    last_blink_state = False
    was_looking_away = False
    last_yaw = 0
    last_pitch = 0
    
    # Display timing
    last_display_time = time.time()
    DISPLAY_UPDATE_INTERVAL = 0.5  # Update display every 0.5 seconds
    
    # Store latest metrics
    latest_metrics = None
    
    # Track if we've shown the first display
    first_display_shown = False
    
    try:
        while True:
            frame_metrics = tracker.process_frame(0)
            
            if frame_metrics:
                frame_count += 1
                latest_metrics = frame_metrics
                
                # === EVENT DETECTION (Option 2) - Print immediately ===
                
                # Detect blink START
                if frame_metrics.is_blinking and not last_blink_state:
                    blink_count += 1
                    # Clear display area, print event, will redraw display next
                    if first_display_shown:
                        clear_lines(13)
                    print(f"[{time.strftime('%H:%M:%S')}] 🔴 BLINK #{blink_count} detected!")
                    print()  # Blank line for separation
                    first_display_shown = False  # Force redraw
                
                # Detect looking away START
                looking_away = (abs(frame_metrics.head_yaw) > 30 or 
                               abs(frame_metrics.head_pitch) > 30)
                
                if looking_away and not was_looking_away:
                    looking_away_count += 1
                    if first_display_shown:
                        clear_lines(13)
                    print(f"[{time.strftime('%H:%M:%S')}] ⚠️  LOOKING AWAY #{looking_away_count} - Yaw: {frame_metrics.head_yaw:.1f}°, Pitch: {frame_metrics.head_pitch:.1f}°")
                    print()
                    first_display_shown = False
                
                # Detect large head movements (>15° change)
                yaw_change = abs(frame_metrics.head_yaw - last_yaw)
                pitch_change = abs(frame_metrics.head_pitch - last_pitch)
                
                if yaw_change > 15 and frame_count > 30:  # Skip initial frames
                    if first_display_shown:
                        clear_lines(13)
                    print(f"[{time.strftime('%H:%M:%S')}] ↔️  Large yaw movement: {frame_metrics.head_yaw:.1f}° (Δ{yaw_change:.1f}°)")
                    print()
                    first_display_shown = False
                
                if pitch_change > 15 and frame_count > 30:
                    if first_display_shown:
                        clear_lines(13)
                    print(f"[{time.strftime('%H:%M:%S')}] ↕️  Large pitch movement: {frame_metrics.head_pitch:.1f}° (Δ{pitch_change:.1f}°)")
                    print()
                    first_display_shown = False
                
                # Update state
                last_blink_state = frame_metrics.is_blinking
                was_looking_away = looking_away
                last_yaw = frame_metrics.head_yaw
                last_pitch = frame_metrics.head_pitch
                
                # === CONTINUOUS DISPLAY (Option 1) - Update every 0.5s ===
                current_time = time.time()
                if current_time - last_display_time >= DISPLAY_UPDATE_INTERVAL:
                    # Clear previous display
                    if first_display_shown:
                        clear_lines(13)
                    
                    # Display current metrics
                    print(f"{'LIVE TRACKING':<30} {'VALUE':<20} {'STATUS':<20}")
                    print("-" * 70)
                    print(f"{'Frames Processed':<30} {frame_count:<20}")
                    print(f"{'Blinks Detected':<30} {blink_count:<20}")
                    print(f"{'Looking Away Events':<30} {looking_away_count:<20}")
                    print(f"{'Currently Blinking':<30} {'YES' if frame_metrics.is_blinking else 'NO':<20} {'🔴' if frame_metrics.is_blinking else '✓'}")
                    print(f"{'Currently Looking Away':<30} {'YES' if looking_away else 'NO':<20} {'⚠️ ' if looking_away else '✓'}")
                    print(f"{'Head Distance (Z)':<30} {frame_metrics.head_z:<20.2f}")
                    print(f"{'Head Yaw (L/R)':<30} {frame_metrics.head_yaw:<20.1f}°")
                    print(f"{'Head Pitch (U/D)':<30} {frame_metrics.head_pitch:<20.1f}°")
                    print(f"{'Head Roll (Tilt)':<30} {frame_metrics.head_roll:<20.1f}°")
                    print(f"{'Left Eye Openness':<30} {frame_metrics.left_eye_aspect_ratio:<20.3f}")
                    print(f"{'Right Eye Openness':<30} {frame_metrics.right_eye_aspect_ratio:<20.3f}")
                    
                    sys.stdout.flush()
                    last_display_time = current_time
                    first_display_shown = True
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("TRACKING STOPPED")
        print("=" * 70)
        print(f"Total frames: {frame_count}")
        print(f"Total blinks: {blink_count}")
        print(f"Looking away events: {looking_away_count}")
        print(f"Duration: ~{frame_count / 30:.1f} seconds")
    
    finally:
        tracker.release()
        print("✓ Webcam released")


if __name__ == "__main__":
    test_live_tracking()

