"""
Simple test script for webcam tracking functionality
Tests the tracking components without running the full experiment
"""
import time
from webcam_tracker import WebcamTracker
from tracking_logger import TrackingLogger


def test_tracking_basic():
    """Test basic tracking functionality."""
    print("=" * 60)
    print("Testing Webcam Tracking System")
    print("=" * 60)
    
    # Initialize tracker
    print("\n1. Initializing webcam tracker...")
    tracker = WebcamTracker(enabled=True, camera_index=0)
    
    if not tracker.enabled:
        print("ERROR: Webcam could not be initialized!")
        return False
    
    print("✓ Webcam initialized successfully")
    
    # Initialize logger
    print("\n2. Initializing tracking logger...")
    logger = TrackingLogger(enabled=True)
    print("✓ Tracking logger initialized")
    
    # Simulate a few trials
    print("\n3. Simulating 3 trials (5 seconds each)...")
    print("   Please look at the camera and blink naturally")
    
    for trial_idx in range(3):
        print(f"\n   Trial {trial_idx + 1}/3...")
        tracker.start_trial(trial_idx)
        
        # Collect frames for 5 seconds
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:
            frame_metrics = tracker.process_frame(trial_idx)
            
            if frame_metrics:
                logger.log_frame(frame_metrics)
                frame_count += 1
            
            time.sleep(0.033)  # ~30 FPS
        
        # End trial and get metrics
        trial_metrics = tracker.end_trial()
        logger.log_trial(trial_metrics)
        
        print(f"   ✓ Collected {frame_count} frames")
        print(f"   ✓ Detected {trial_metrics.get('blink_count', 0)} blinks")
        print(f"   ✓ Blink rate: {trial_metrics.get('blink_rate', 0):.2f} blinks/sec")
        print(f"   ✓ Head movement variance: {trial_metrics.get('head_movement_variance', 0):.6f}")
    
    # Test data saving
    print("\n4. Testing data saving...")

    # Save frame data
    frame_file = logger.save_frame_data("results/test_tracking_frames.csv")
    print(f"   ✓ Frame data saved: {frame_file}")

    # Save session summary
    session_file = logger.save_session_summary("results/test_tracking_session.csv")
    print(f"   ✓ Session summary saved: {session_file}")
    
    # Display session summary
    print("\n5. Session Summary:")
    summary = logger.calculate_session_summary()
    print(f"   Total frames tracked: {summary['total_frames_tracked']}")
    print(f"   Total trials tracked: {summary['total_trials_tracked']}")
    print(f"   Total blinks: {summary['total_blinks']}")
    print(f"   Overall blink rate: {summary['overall_blink_rate']:.3f} blinks/sec")
    print(f"   Mean head stability: {summary['mean_head_stability']:.6f}")
    print(f"   Engagement score: {summary['engagement_score']:.3f}")
    print(f"   Fatigue indicator: {summary['fatigue_indicator']:.3f}")
    
    # Cleanup
    print("\n6. Cleaning up...")
    tracker.release()
    print("   ✓ Webcam released")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_tracking_basic()
        if not success:
            print("\n✗ Tests failed!")
            exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

