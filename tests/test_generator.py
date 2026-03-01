"""
Test script for stimulus generator (no pygame required)
"""
from stimulus_generator import StimulusGenerator


def test_generator():
    """Test the stimulus generator."""
    print("Testing Stimulus Generator...")
    print("-" * 50)
    
    # Create generator
    gen = StimulusGenerator(total_trials=100, target_probability=0.15)
    
    # Generate sequence
    sequence = gen.generate_sequence()
    
    print(f"Total stimuli generated: {len(sequence)}")
    print(f"Expected: {100 * 2} (2 stimuli per trial)")
    
    # Count trial types
    trial_types = {"AX": 0, "BX": 0, "AY": 0, "BY": 0, "NONE": 0}
    
    for i in range(1, len(sequence)):
        trial_type = gen.get_trial_type(sequence[i], sequence[i-1])
        trial_types[trial_type] += 1
    
    print("\nTrial Type Distribution:")
    for tt, count in trial_types.items():
        if tt != "NONE":
            percentage = (count / 100) * 100
            print(f"  {tt}: {count} ({percentage:.1f}%)")
    
    print(f"\nTarget (AX) probability: {trial_types['AX'] / 100:.2%}")
    print(f"Expected: ~15%")
    
    # Show first 20 stimuli
    print("\nFirst 20 stimuli:")
    print(" ".join(sequence[:20]))
    
    # Verify trial types
    print("\nFirst 10 trials:")
    for i in range(0, min(20, len(sequence)), 2):
        if i + 1 < len(sequence):
            cue = sequence[i]
            probe = sequence[i + 1]
            trial_type = gen.get_trial_type(probe, cue)
            print(f"  Trial {i//2 + 1}: {cue}{probe} → {trial_type}")
    
    print("\n" + "-" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_generator()

