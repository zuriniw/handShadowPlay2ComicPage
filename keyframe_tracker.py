import os
import time
import json
import math
import socket
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
        
        # Add flag to track if keyframe limit has been reached
        self.keyframe_limit_reached = False
        
        # Try to load existing keyframes if the file exists
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    file_data = json.load(f)
                
                # Handle both old and new formats
                if isinstance(file_data, dict) and "keyframe_sequence" in file_data:
                    # New format with keyframe_sequence and all_characters
                    keyframe_sequence = file_data.get("keyframe_sequence", [])
                    all_characters = file_data.get("all_characters", [])
                    
                    # Convert the simplified keyframes back to our internal format
                    for kf in keyframe_sequence:
                        # Create full keyframe with all required fields
                        full_keyframe = {
                            "timestamp": kf.get("timestamp", 0),
                            "name": kf.get("name", ""),
                            "current_characters": kf.get("current_characters", []),
                            "all_characters": all_characters  # Use the global all_characters
                        }
                        self.keyframes.append(full_keyframe)
                        
                        # Track announced characters
                        if full_keyframe["name"].startswith("add new character"):
                            character_name = full_keyframe["name"].replace("add new character ", "").strip()
                            self.announced_characters.add(character_name)
                        elif full_keyframe["name"].startswith("from blank add a new animal:"):
                            character_name = full_keyframe["name"].replace("from blank add a new animal:", "").strip()
                            self.announced_characters.add(character_name)
                elif isinstance(file_data, list):
                    # Old format (directly a list of keyframes)
                    self.keyframes = file_data
                    
                    # Track announced characters from loaded keyframes
                    for kf in self.keyframes:
                        if kf["name"].startswith("add new character"):
                            character_name = kf["name"].replace("add new character ", "").strip()
                            self.announced_characters.add(character_name)
                        elif kf["name"].startswith("from blank add a new animal:"):
                            character_name = kf["name"].replace("from blank add a new animal:", "").strip()
                            self.announced_characters.add(character_name)
                
                print(f"Loaded {len(self.keyframes)} keyframes from {self.filename}")
            except Exception as e:
                print(f"Error loading keyframes from {self.filename}: {str(e)}")
                traceback.print_exc()
                # Start with empty keyframes
                self.keyframes = []
        
        # Write empty keyframes file if we didn't load anything
        if not self.keyframes:
            self.save_keyframes()
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds since tracker was created"""
        return time.time() - self.start_time
    
    def send_socket_signal(self, current_characters, keyframe_name):
        """Send a signal to the socket server when a keyframe is added"""
        # Don't send signals if keyframe limit has been reached
        if self.keyframe_limit_reached:
            print("Keyframe limit reached, not sending socket signal")
            return
            
        try:
            # Create message based on number of characters
            message = ""
            num_chars = len(current_characters)
            
            # Check if this is a distance-related keyframe (far or close)
            if "far enough" in keyframe_name:
                # Extract animal names from keyframe name
                # Format typically: "far enough animal1 <-> animal2"
                parts = keyframe_name.replace("far enough ", "").split(" <-> ")
                if len(parts) == 2:
                    animal1, animal2 = parts
                    message = f"there is a {animal1} on the left, and a {animal2} on the right, they are far apart"
                else:
                    # Fallback if parsing fails
                    if num_chars == 2:
                        left_animal = current_characters[0]
                        right_animal = current_characters[1]
                        message = f"There is a black {left_animal} on the left, and a black {right_animal} on the right, they are far apart."
            elif "close enough" in keyframe_name:
                # Extract animal names from keyframe name
                # Format typically: "close enough animal1 <-> animal2"
                parts = keyframe_name.replace("close enough ", "").split(" <-> ")
                if len(parts) == 2:
                    animal1, animal2 = parts
                    message = f"There is a black {animal1} on the left, and a black {animal2} on the right, they are close together."
                else:
                    # Fallback if parsing fails
                    if num_chars == 2:
                        left_animal = current_characters[0]
                        right_animal = current_characters[1]
                        message = f"There is a black {left_animal} on the left, and a black {right_animal} on the right, they are close together."
            else:
                # Standard message format based on character count
                if num_chars == 2:
                    # Format: "there is a xx on the left, and a xx on the right"
                    left_animal = current_characters[0]
                    right_animal = current_characters[1]
                    message = f"There is a black {left_animal} on the left, and a black {right_animal} on the right."
                elif num_chars == 1:
                    # Format: "a single animaltype"
                    animal = current_characters[0]
                    message = f"There is a back {animal}."
                else:
                    # Format: "empty scene"
                    message = "This is a peacefulempty scene."
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', 8000))
                s.sendall(message.encode('utf-8'))
                print(f"Socket signal sent: '{message}'")
        except Exception as e:
            print(f"Error sending socket signal: {str(e)}")
            traceback.print_exc()
    
    def add_keyframe(self, name, current_characters, all_characters):
        """Add a new keyframe with the given information"""
        # Don't add keyframes if limit has been reached
        if self.keyframe_limit_reached:
            print("Keyframe limit reached, not adding new keyframe")
            return False
        
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
        timestamp_str = datetime.now().strftime("%H:%M:%S")  # Only used for console output
        
        # Create simplified keyframe structure
        keyframe = {
            "timestamp": round(timestamp, 2),
            "name": name,
            "current_characters": list(current_characters),
            "all_characters": list(all_characters)  # Keep this for internal use
        }
        
        # Add to keyframes - insert in chronological order
        self.keyframes.append(keyframe)
        # Sort keyframes by timestamp to ensure they're always in chronological order
        self.keyframes.sort(key=lambda x: x.get('timestamp', 0))
        
        self.last_keyframe_time = current_time
        self.save_keyframes()
        print(f"Keyframe recorded: {name} at {timestamp_str}")
        
        # Send socket signal after keyframe is saved
        self.send_socket_signal(list(current_characters), name)
        
        # If this is an "add new character" keyframe, mark the character as announced
        if name.startswith("add new character"):
            character_name = name.replace("add new character ", "").strip()
            self.announced_characters.add(character_name)
        elif name.startswith("from blank add a new animal:"):
            character_name = name.replace("from blank add a new animal:", "").strip()
            self.announced_characters.add(character_name)
        
        # Check if we've reached the keyframe limit
        limit_result = self.check_keyframe_limit()
        
        return True
    
    def save_keyframes(self):
        """Save keyframes to JSON file in a more concise format"""
        # Don't save if keyframe limit has been reached
        if self.keyframe_limit_reached:
            print("Keyframe limit reached, not saving keyframes")
            return
            
        try:
            # Always ensure keyframes are sorted before saving
            self.keyframes.sort(key=lambda x: x.get('timestamp', 0))
            
            # Make sure we're recording non-empty data
            if len(self.keyframes) > 0:
                print(f"Saving {len(self.keyframes)} keyframes to {self.filename}")
                
                # Create a simplified version of keyframes - removing time and all_characters
                simplified_keyframes = []
                
                # Get all_characters from the last keyframe if available
                all_characters = []
                if self.keyframes:
                    all_characters = self.keyframes[-1].get('all_characters', [])
                
                # Create simplified version of each keyframe
                for kf in self.keyframes:
                    simplified_kf = {
                        "timestamp": kf.get('timestamp', 0),
                        "name": kf.get('name', ''),
                        "current_characters": kf.get('current_characters', [])
                    }
                    simplified_keyframes.append(simplified_kf)
                
                # Create the new structure with keyframe_sequence and all_characters
                output_data = {
                    "keyframe_sequence": simplified_keyframes,
                    "all_characters": all_characters
                }
                
                # Use more robust saving approach with flush to ensure data is written
                with open(self.filename, 'w') as f:
                    json.dump(output_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force OS to write to disk
                    
                # Verify the file was saved correctly
                if os.path.exists(self.filename):
                    file_size = os.path.getsize(self.filename)
                    print(f"Successfully saved keyframes file: {self.filename} ({file_size} bytes)")
                else:
                    print(f"Warning: File {self.filename} doesn't exist after save attempt")
            else:
                # Just create an empty JSON structure
                empty_data = {
                    "keyframe_sequence": [],
                    "all_characters": []
                }
                with open(self.filename, 'w') as f:
                    json.dump(empty_data, f, indent=2)
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
        if current_time - self.last_hand_seen_time > 1000:
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

        if distance < 260:
            current_distance_state = "close"
            name = f"close enough {char1} <-> {char2}"
            if self.last_distance_state != "close" and should_add_keyframe(name):
                self.add_keyframe(name, current_chars, all_characters_history)
        elif distance > 580:
            current_distance_state = "far"
            name = f"far enough {char1} <-> {char2}"
            if self.last_distance_state != "far" and should_add_keyframe(name):
                self.add_keyframe(name, current_chars, all_characters_history)

        # Update last distance state
        self.last_distance_state = current_distance_state

    def add_keyframe_force(self, name, current_characters, all_characters):
        """Add a new keyframe with the given information, bypassing time restrictions.
        This is used for critical events like add and quit that must be recorded."""
        
        # Don't add forced keyframes if limit has been reached
        if self.keyframe_limit_reached:
            print("Keyframe limit reached, not adding forced keyframe")
            return False
        
        # Check if this keyframe is a duplicate of the previous one
        if self.keyframes and self.keyframes[-1]['name'] == name:
            print(f"Skipping duplicate keyframe: {name}")
            return False
        
        # Create keyframe data
        timestamp = self.get_elapsed_time()
        timestamp_str = datetime.now().strftime("%H:%M:%S")  # Only used for console output
        
        # Create simplified keyframe structure
        keyframe = {
            "timestamp": round(timestamp, 2),
            "name": name,
            "current_characters": list(current_characters),
            "all_characters": list(all_characters)  # Keep this for internal use
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
        
        # Send socket signal after keyframe is saved
        self.send_socket_signal(list(current_characters), name)
        
        # Track announced characters
        if name.startswith("add new character"):
            character_name = name.replace("add new character ", "").strip()
            self.announced_characters.add(character_name)
        elif name.startswith("from blank add a new animal:"):
            character_name = name.replace("from blank add a new animal:", "").strip()
            self.announced_characters.add(character_name)
        
        # Check if we've reached the keyframe limit
        limit_result = self.check_keyframe_limit()
        
        return True

    def check_keyframe_limit(self):
        """Check if the number of keyframes has reached or exceeded 6.
        If so, set the keyframe_limit_reached flag but don't terminate the program.
        """
        if len(self.keyframes) >= 6:
            print("\n==================================")
            print("Thanks for playing!")
            print("6 keyframes have been recorded")
            print("No more keyframes will be recorded or sent")
            print("==================================\n")
            self.keyframe_limit_reached = True
            return True
        return False
        
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
            
            # Check if the data has the correct structure
            if not isinstance(file_data, dict) or "keyframe_sequence" not in file_data:
                print(f"WARNING: File structure is incorrect, missing keyframe_sequence")
                self.save_keyframes()
                return False
                
            # Get the keyframe sequence and all_characters
            keyframe_sequence = file_data.get("keyframe_sequence", [])
            all_characters = file_data.get("all_characters", [])
            
            # Check if the count matches what we expect
            if len(keyframe_sequence) != len(self.keyframes):
                print(f"WARNING: File contains {len(keyframe_sequence)} keyframes but memory has {len(self.keyframes)}")
                # Force a save to correct it
                self.save_keyframes()
                return False
            
            # Ensure our in-memory keyframes are consistent with the file
            # This shouldn't normally be necessary, but helps recover from file/memory mismatches
            if len(keyframe_sequence) > 0 and len(self.keyframes) > 0:
                # Compare timestamps and names to ensure consistency
                file_timestamps = [kf.get('timestamp', 0) for kf in keyframe_sequence]
                memory_timestamps = [kf.get('timestamp', 0) for kf in self.keyframes]
                
                if file_timestamps != memory_timestamps:
                    print("WARNING: Keyframe timestamps in file don't match memory")
                    self.save_keyframes()
                    return False
            
            # Count event types for verification
            add_events = [kf for kf in keyframe_sequence if kf['name'].startswith('add new character')]
            quit_events = [kf for kf in keyframe_sequence if kf['name'].startswith('quit character')]
            close_events = [kf for kf in keyframe_sequence if 'close enough' in kf['name']]
            far_events = [kf for kf in keyframe_sequence if 'far enough' in kf['name']]
            
            return True
        except Exception as e:
            print(f"Error verifying keyframes file: {str(e)}")
            traceback.print_exc()
            return False

    def is_limit_reached(self):
        """Return whether the keyframe limit has been reached"""
        return self.keyframe_limit_reached

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