import argparse
import cv2
import numpy as np
import os

# Usage: 'python object_verification.py Reference_ID --camera 0 --size 200 200'

def load_reference_ids(reference_path, target_size=(200, 200)):
    """Load all reference IDs from the specified directory.

    For consistency with `face_recog.py` we resize reference images to 200x200,
    convert to grayscale and apply histogram equalization.
    """
    reference_ids = {}

    for dirname, _, filenames in os.walk(reference_path):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                person_name = os.path.basename(dirname)
                filepath = os.path.join(dirname, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    # Resize image to the target size used for matching
                    img = cv2.resize(img, target_size)
                    # Convert to grayscale (SIFT uses gray images)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Improve contrast like face_recog
                    img = cv2.equalizeHist(img)
                    reference_ids.setdefault(person_name, []).append(img)
                    print(f"Loaded and preprocessed ID for {person_name}: {filename}")
    return reference_ids

def initialize_sift():
    """Initialize SIFT detector."""
    return cv2.SIFT_create()

def initialize_flann():
    """Initialize FLANN-based matcher."""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)

def verify_id(frame, reference_ids, sift, flann, min_matches=10, ratio_thresh=0.7, target_size=(200,200)):
    """
    Verify if the ID shown in frame matches any reference ID.
    
    Args:
        frame: Current frame from camera
        reference_ids: Dictionary of reference IDs
        sift: SIFT detector
        flann: FLANN matcher
        min_matches: Minimum number of good matches required
        ratio_thresh: Ratio threshold for Lowe's ratio test
        
    Returns:
        best_match: Name of the best matching ID or None
        match_quality: Number of good matches for the best match
    """
    # Resize frame to the same size as reference images (smaller -> faster)
    frame = cv2.resize(frame, target_size)

    # Convert frame to grayscale and equalize to match reference preprocessing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    # Find keypoints and descriptors for the frame
    frame_kp, frame_des = sift.detectAndCompute(frame_gray, None)
    
    if frame_des is None:
        return None, 0
    
    best_match = None
    best_match_count = 0

    # Compare with each reference ID (note: reference_ids now maps name -> list of imgs)
    for person_name, ref_imgs in reference_ids.items():
        for ref_img in ref_imgs:
            # ref_img is already grayscale and preprocessed
            ref_kp, ref_des = sift.detectAndCompute(ref_img, None)
            if ref_des is None:
                continue

            # Find matches using FLANN matcher
            matches = flann.knnMatch(frame_des, ref_des, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for m_n in matches:
                # knnMatch can return less than k matches for a descriptor; guard that
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            # Update best match if this one is better
            if len(good_matches) > best_match_count and len(good_matches) >= min_matches:
                best_match = person_name
                best_match_count = len(good_matches)
    
    return best_match, best_match_count

def start_id_verification(reference_path='Reference_ID', camera_index=0, target_size=(200,200), skip_frames=1, memory_size=5, stability_threshold=2, min_matches=10, ratio_thresh=0.7):
    """
    Start ID verification using webcam.
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)

    # Load reference IDs
    print("Loading reference IDs...")
    reference_ids = load_reference_ids(reference_path, target_size)
    if not reference_ids:
        print("No reference IDs found in", reference_path)
        return
    
    # Initialize SIFT and FLANN
    sift = initialize_sift()
    flann = initialize_flann()
    
    print("Ready to verify IDs. Press 'q' to quit.")
    
    # Initialize status memory
    last_matches = []  # Keep track of last N matches
    current_status = {"name": None, "quality": 0}
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Only process every nth frame
        if frame_count % skip_frames == 0:
            # Verify ID in current frame
            match_name, match_quality = verify_id(frame, reference_ids, sift, flann, min_matches=min_matches, ratio_thresh=ratio_thresh, target_size=target_size)
            
            # Update match history
            last_matches.append((match_name, match_quality))
            if len(last_matches) > memory_size:
                last_matches.pop(0)
            
            # Count occurrences of each match in history
            match_counts = {}
            for name, quality in last_matches:
                if name:
                    match_counts[name] = match_counts.get(name, 0) + 1
            
            # Update current status if we have a stable match
            max_count = 0
            stable_match = None
            for name, count in match_counts.items():
                if count > max_count:
                    max_count = count
                    stable_match = name
            
            if max_count >= stability_threshold:
                # We have a stable match
                current_status["name"] = stable_match
                # Average the quality scores for this match
                qualities = [q for n, q in last_matches if n == stable_match]
                current_status["quality"] = sum(qualities) / len(qualities)
            elif len(last_matches) >= memory_size and max_count < 2:
                # Clear current status if no consistent matches
                current_status["name"] = None
                current_status["quality"] = 0
        
        # Display the current status (this happens every frame)
        if current_status["name"]:
            # Create background rectangle for text
            text = f"ID Match: {current_status['name']}"
            text2 = f"Quality: {int(current_status['quality'])}"
            
            # Calculate text sizes
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            (text_width2, text_height2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            # Draw background rectangles
            cv2.rectangle(frame, (5, 5), (text_width + 15, 45), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 45), (text_width2 + 15, 85), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text2, (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Draw background rectangle for "No ID Match" text
            (text_width, text_height), _ = cv2.getTextSize("No ID Match", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (5, 5), (text_width + 15, 45), (0, 0, 0), -1)
            
            cv2.putText(frame, "No ID Match", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        frame_count += 1
        
        # Show the frame
        cv2.imshow('ID Verification', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object (ID) verification using SIFT + FLANN')
    parser.add_argument('reference', nargs='?', default='Reference_ID', help='Reference ID folder')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--size', type=int, nargs=2, metavar=('W','H'), default=[200,200], help='Target size to resize images to (W H)')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every Nth frame (default: 1)')
    parser.add_argument('--memory-size', type=int, default=5, help='Number of frames to remember for stability')
    parser.add_argument('--stability-threshold', type=int, default=2, help='Number of consistent frames needed to show a match')
    parser.add_argument('--min-matches', type=int, default=10, help='Minimum good matches required')
    parser.add_argument('--ratio', type=float, default=0.7, help="Lowe's ratio threshold")

    args = parser.parse_args()

    start_id_verification(reference_path=args.reference,
                          camera_index=args.camera,
                          target_size=(args.size[0], args.size[1]),
                          skip_frames=max(1, args.skip_frames),
                          memory_size=max(1, args.memory_size),
                          stability_threshold=max(1, args.stability_threshold),
                          min_matches=max(1, args.min_matches),
                          ratio_thresh=max(0.01, args.ratio))
