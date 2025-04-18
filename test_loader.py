import keyframe_tracker

# Create a keyframe tracker with our test file
tracker = keyframe_tracker.KeyframeTracker("test_keyframes.json")

# Print the keyframe count
print(f"Loaded {len(tracker.keyframes)} keyframes")

# Check if limit is reached (should be True)
is_limit_reached = tracker.check_keyframe_limit()
print(f"Keyframe limit reached: {is_limit_reached}")

# Print the loaded keyframes
print("\nLoaded keyframes:")
for i, kf in enumerate(tracker.keyframes):
    print(f"{i+1}. {kf['name']} - {kf['current_characters']}")

print("\nAll characters:", tracker.keyframes[-1]["all_characters"]) 