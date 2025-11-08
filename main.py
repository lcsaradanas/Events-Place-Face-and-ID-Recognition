import numpy as np
import os
import sys
import cv2
from read_images import read_images

def lbph_recognition():
    names = ['Aradanas', 'Guarino', 'Valdez', 'Bona']
    
    if len(sys.argv) < 2:
        print("USAGE: python main.py </path/to/images>")
        sys.exit()

    print("Loading training data...")
    [X, y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)
        
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y)
    
    print("LBPH model trained successfully!")
    
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    test_count = 0
    results = []

    while True:
        ret, img = camera.read()
        if not ret:
            break
            
        # Mirror the image horizontally
        img = cv2.flip(img, 1)  

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with adjusted parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Extract face ROI and preprocess
            face_roi = gray[y:y + h, x:x + w]
            
            # Equalize histogram of face region
            face_roi = cv2.equalizeHist(face_roi)
            
            # Resize to standard size
            roi = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                prediction = model.predict(roi)
                person_id = prediction[0]
                confidence = prediction[1]
                
                confidence_threshold = 150 # Lower camera resolution thus higher threshold
                
                if person_id < len(names) and confidence < confidence_threshold:
                    label = names[person_id]
                    display_text = f"{label}, Confidence: {confidence:.1f}"
                    # NOT YET IMPLEMENTED
                    results.append({
                        'person_id': person_id,
                        'person_name': label,
                        'confidence': confidence
                    })
                    
                else:
                    label = "Unknown"
                    display_text = f"Unknown, Confidence: {confidence:.1f}"
                    
                    results.append({
                        'person_id': -1,
                        'person_name': 'Unknown',
                        'confidence': confidence
                    })
                
                cv2.putText(img, display_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            except Exception as e:
                cv2.putText(img, "Error", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Event Entrance (LBPH Face Recognition)", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    lbph_recognition()