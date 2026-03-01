#!/usr/bin/env python3
"""
Test script to check what FPS rates the webcam actually supports.
"""
import cv2
import time

def test_fps(requested_fps):
    """Test if webcam supports a specific FPS."""
    print(f"\n{'='*60}")
    print(f"Testing {requested_fps} FPS...")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return None
    
    # Set desired FPS
    cap.set(cv2.CAP_PROP_FPS, requested_fps)
    
    # Get actual FPS that was set
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested FPS: {requested_fps}")
    print(f"Actual FPS (from camera): {actual_fps}")
    
    # Measure real frame rate by capturing frames
    print("\nMeasuring actual frame rate (5 second test)...")
    frame_count = 0
    start_time = time.time()
    test_duration = 5.0
    
    while (time.time() - start_time) < test_duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
    
    elapsed = time.time() - start_time
    measured_fps = frame_count / elapsed
    
    print(f"Frames captured: {frame_count}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Measured FPS: {measured_fps:.2f}")
    
    cap.release()
    
    return {
        'requested': requested_fps,
        'reported': actual_fps,
        'measured': measured_fps
    }

def main():
    print("="*60)
    print("MacBook Air Webcam FPS Test")
    print("="*60)
    
    # Test common frame rates
    fps_to_test = [15, 24, 30, 60, 120]
    results = []
    
    for fps in fps_to_test:
        result = test_fps(fps)
        if result:
            results.append(result)
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Requested':<12} {'Reported':<12} {'Measured':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['requested']:<12} {r['reported']:<12.1f} {r['measured']:<12.2f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    # Find the highest measured FPS
    max_measured = max(r['measured'] for r in results)
    
    if max_measured >= 55:
        print("✓ Your webcam supports 60 FPS!")
        print("  Recommendation: Use 60 FPS for maximum tracking precision")
    elif max_measured >= 28:
        print("✓ Your webcam supports 30 FPS")
        print("  Recommendation: Use 30 FPS (sufficient for blink/head tracking)")
    else:
        print(f"⚠ Your webcam appears limited to ~{max_measured:.0f} FPS")
        print(f"  Recommendation: Use {max_measured:.0f} FPS")

if __name__ == "__main__":
    main()

