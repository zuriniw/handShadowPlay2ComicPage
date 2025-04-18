import os
import time
import json
import math
from datetime import datetime
import traceback

class KeyframeTracker:
    def __init__(self, json_filename=None):
        self.keyframes = []
        
        # Generate timestamped filename if none provided
        if json_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"keyframes_{timestamp}.json"
        else:
            self.filename = json_filename
            
        print(f"Recording keyframes to: {self.filename}")
        
        self.start_time = time.time()
        self.last_hand_seen_time = time.time()
        self.last_keyframe_time = time.time()
        self.min_time_between_keyframes = 2.0  # Minimum seconds between keyframes to avoid duplicates
        self.last_distance_state = None  # "close", "far", or None
        self.prev_characters = set()  # Track previously seen characters
        self.announced_characters = set()  # Track characters that have already been announced
        
        # Write empty keyframes file to start fresh
        self.save_keyframes()
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds since tracker was created"""
        return time.time() - self.start_time
    
    def add_keyframe(self, name, current_characters, all_characters):
        """Add a new keyframe with the given information"""
        # Check if enough time has passed since last keyframe
        current_time = time.time()
        if current_time - self.last_keyframe_time < self.min_time_between_keyframes:
            print(f"Skipping keyframe due to timing: {name}")
            return False
        
        # Check if this keyframe is a duplicate of the previous one
        if self.keyframes and self.keyframes[-1]['name'] == name:
            print(f"Skipping duplicate keyframe: {name}")
            return False
        
        # Create keyframe data
        timestamp = self.get_elapsed_time()
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        
        keyframe = {
            "timestamp": round(timestamp, 2),
            "time": timestamp_str,
            "name": name,
            "current_characters": list(current_characters),
            "all_characters": list(all_characters)
        }
        
        # Add to keyframes - insert in chronological order
        self.keyframes.append(keyframe)
        # Sort keyframes by timestamp to ensure they're always in chronological order
        self.keyframes.sort(key=lambda x: x.get('timestamp', 0))
        
        self.last_keyframe_time = current_time
        self.save_keyframes()
        print(f"Keyframe recorded: {name} at {timestamp_str}")
        
        # If this is an "add new character" keyframe, mark the character as announced
        if name.startswith("add new character"):
            character_name = name.replace("add new character ", "").strip()
            self.announced_characters.add(character_name)
        elif name.startswith("from blank add a new animal:"):
            character_name = name.replace("from blank add a new animal:", "").strip()
            self.announced_characters.add(character_name)
        
        return True
    
    def save_keyframes(self):
        """Save keyframes to JSON file"""
        try:
            # Always ensure keyframes are sorted before saving
            self.keyframes.sort(key=lambda x: x.get('timestamp', 0))
            
            # Make sure we're recording non-empty data
            if len(self.keyframes) > 0:
                print(f"Saving {len(self.keyframes)} keyframes to {self.filename}")
                
                # Use more robust saving approach with flush to ensure data is written
                with open(self.filename, 'w') as f:
                    json.dump(self.keyframes, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force OS to write to disk
                    
                # Verify the file was saved correctly
                if os.path.exists(self.filename):
                    file_size = os.path.getsize(self.filename)
                    print(f"Successfully saved keyframes file: {self.filename} ({file_size} bytes)")
                else:
                    print(f"Warning: File {self.filename} doesn't exist after save attempt")
            else:
                # Just create an empty array in the file
                with open(self.filename, 'w') as f:
                    json.dump([], f)
                    f.flush()
                    os.fsync(f.fileno())
                    
                print(f"Created empty keyframes file: {self.filename}")
                
        except Exception as e:
            print(f"Error saving keyframes: {str(e)}")
            traceback.print_exc()
            
    def reset_keyframes(self):
        """Reset keyframes list and save empty file"""
        # Generate new timestamped filename when resetting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"keyframes_{timestamp}.json"
        print(f"Resetting keyframes. New file: {self.filename}")
        
        self.keyframes = []
        self.announced_characters = set()  # Reset announced characters as well
        self.save_keyframes()
    
    def check_terminate(self, any_hands_visible, all_characters_history):
        """Check if no hands have been visible for more than 5 seconds"""
        current_time = time.time()
        
        if any_hands_visible:
            # Reset last seen time when hands are visible
            self.last_hand_seen_time = current_time
            return False
        
        # If no hands are visible for more than 5 seconds
        if current_time - self.last_hand_seen_time > 5.0:
            # Add terminate keyframe
            self.add_keyframe("terminate", [], all_characters_history)
            # Reset timer after recording
            self.last_hand_seen_time = current_time
            return True
        
        return False
    
    def check_character_changes(self, current_character_ids, all_characters_history):
        """Check for character additions or removals"""
        # Convert to set of character IDs for comparison
        current_chars = set(current_character_ids.values())

        # Helper to prevent duplicate consecutive keyframes
        def should_add_keyframe(name):
            return (not self.keyframes or self.keyframes[-1]['name'] != name)

        # Special case: from blank to first character
        if not self.prev_characters and len(current_chars) == 1:
            first_char = list(current_chars)[0]
            
            # Only announce if this character hasn't already been announced through the direct method
            if first_char not in self.announced_characters:
                name = f"from blank add a new animal: {first_char}"
                if should_add_keyframe(name):
                    self.add_keyframe(name, current_chars, all_characters_history)
                    self.announced_characters.add(first_char)
        else:
            # Check for new characters
            new_chars = current_chars - self.prev_characters
            for char in new_chars:
                # Only announce if this character hasn't already been announced through the direct method
                if char not in self.announced_characters:
                    name = f"add new character {char}"
                    if should_add_keyframe(name):
                        self.add_keyframe(name, current_chars, all_characters_history)
                        self.announced_characters.add(char)

        # Update previous characters
        self.prev_characters = current_chars

    def check_distance(self, character_ids, handedness, landmarks, all_characters_history):
        """Check distance between characters if there are at least two with IDs"""
        # Helper to prevent duplicate consecutive keyframes
        def should_add_keyframe(name):
            return (not self.keyframes or self.keyframes[-1]['name'] != name)

        # If we don't have enough data, return
        if len(handedness) < 2 or len(landmarks) < 2 or len(character_ids) < 2:
            return

        # Collect hands with character IDs
        characters_with_ids = []
        hand_landmarks = []

        # Find hands that have character IDs
        for i, hand_type in enumerate(handedness):
            if i < len(landmarks) and hand_type in character_ids and character_ids[hand_type]:
                characters_with_ids.append(character_ids[hand_type])
                hand_landmarks.append(landmarks[i])

        # We need at least two hands with character IDs
        if len(characters_with_ids) < 2:
            return

        # Get the first two hands with character IDs
        char1 = characters_with_ids[0]
        char2 = characters_with_ids[1]
        hand1_wrist = hand_landmarks[0][0]  # First point is wrist
        hand2_wrist = hand_landmarks[1][0]  # First point is wrist

        # Calculate Euclidean distance
        distance = math.sqrt((hand1_wrist[0] - hand2_wrist[0])**2 + 
                           (hand1_wrist[1] - hand2_wrist[1])**2)

        # Check for close/far thresholds
        current_distance_state = None
        current_chars = set(character_ids.values())

        if distance < 320:
            current_distance_state = "close"
            name = f"close enough {char1} <-> {char2}"
            if self.last_distance_state != "close" and should_add_keyframe(name):
                self.add_keyframe(name, current_chars, all_characters_history)
        elif distance > 550:
            current_distance_state = "far"
            name = f"far enough {char1} <-> {char2}"
            if self.last_distance_state != "far" and should_add_keyframe(name):
                self.add_keyframe(name, current_chars, all_characters_history)

        # Update last distance state
        self.last_distance_state = current_distance_state

    def add_keyframe_force(self, name, current_characters, all_characters):
        """Add a new keyframe with the given information, bypassing time restrictions.
        This is used for critical events like add and quit that must be recorded."""
        
        # Check if this keyframe is a duplicate of the previous one
        if self.keyframes and self.keyframes[-1]['name'] == name:
            print(f"Skipping duplicate keyframe: {name}")
            return False
        
        # Create keyframe data
        timestamp = self.get_elapsed_time()
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        
        keyframe = {
            "timestamp": round(timestamp, 2),
            "time": timestamp_str,
            "name": name,
            "current_characters": list(current_characters),
            "all_characters": list(all_characters)
        }
        
        # Add to keyframes - insert in chronological order
        self.keyframes.append(keyframe)
        # Sort keyframes by timestamp to ensure they're always in chronological order
        self.keyframes.sort(key=lambda x: x.get('timestamp', 0))
        
        # Update the last keyframe time to avoid adding too many keyframes
        self.last_keyframe_time = time.time()
        
        # Save immediately
        self.save_keyframes()
        print(f"Keyframe recorded: {name} at {timestamp_str}")
        
        # Track announced characters
        if name.startswith("add new character"):
            character_name = name.replace("add new character ", "").strip()
            self.announced_characters.add(character_name)
        elif name.startswith("from blank add a new animal:"):
            character_name = name.replace("from blank add a new animal:", "").strip()
            self.announced_characters.add(character_name)
        
        return True

    def verify_keyframes_file(self):
        """Verify the keyframes file exists and contains the correct data"""
        try:
            if not os.path.exists(self.filename):
                print(f"WARNING: Keyframes file {self.filename} does not exist!")
                self.save_keyframes()
                return False
            
            # Read the file to verify its contents
            with open(self.filename, 'r') as f:
                file_data = json.load(f)
            
            # Check if the data matches what we expect
            if len(file_data) != len(self.keyframes):
                print(f"WARNING: File contains {len(file_data)} keyframes but memory has {len(self.keyframes)}")
                # Force a save to correct it
                self.save_keyframes()
                return False
            
            # Count event types for verification
            add_events = [kf for kf in file_data if kf['name'].startswith('add new character')]
            quit_events = [kf for kf in file_data if kf['name'].startswith('quit character')]
            close_events = [kf for kf in file_data if 'close enough' in kf['name']]
            far_events = [kf for kf in file_data if 'far enough' in kf['name']]
            
           
            return True
        except Exception as e:
            print(f"Error verifying keyframes file: {str(e)}")
            traceback.print_exc()
            return False

# Create global keyframe tracker with timestamp in filename
def create_keyframe_tracker():
    """Create a fresh keyframe tracker with timestamped filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"keyframes_{timestamp}.json"
    return KeyframeTracker(filename)

# Initialize the global tracker on import
_keyframe_tracker = create_keyframe_tracker()

def get_keyframe_tracker():
    """Get the global keyframe tracker instance"""
    global _keyframe_tracker
    return _keyframe_tracker 