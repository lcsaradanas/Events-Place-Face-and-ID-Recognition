import cv2
import os

def prepare_reference_image(input_path, output_path, target_size=(400, 250)):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read image: {input_path}")
        return False
    
    # Resize image
    img_resized = cv2.resize(img, target_size)
    
    # Convert to grayscale and apply histogram equalization
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    
    # Save both original color and preprocessed versions
    cv2.imwrite(output_path, img_resized)
    preprocessed_path = output_path.replace('.jpg', '_preprocessed.jpg')
    cv2.imwrite(preprocessed_path, gray_eq)
    
    print(f"Processed and saved: {output_path}")
    return True

def main():
    # Create Reference_ID directory if it doesn't exist
    if not os.path.exists('Reference_ID'):
        os.makedirs('Reference_ID')
    
    # Process each image in the current directory
    for filename in os.listdir('.'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0].upper()
            output_path = os.path.join('Reference_ID', f'{name}.jpg')
            
            if prepare_reference_image(filename, output_path):
                print(f"Successfully processed {filename}")
            else:
                print(f"Failed to process {filename}")

if __name__ == "__main__":
    main()