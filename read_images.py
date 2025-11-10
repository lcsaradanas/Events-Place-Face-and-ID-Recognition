import numpy as np
import os
import errno
import sys
import cv2


def read_images(path, sz=(200, 200)):
  c = 0
  X, y = [], []
  print("\n[DEBUG] Starting image loading from dataset...")

  for dirname, dirnames, filenames in os.walk(path):
    for subdirname in dirnames:
      subject_path = os.path.join(dirname, subdirname)
      print(f"\n[DEBUG] Processing folder: {subdirname}")
      image_count = 0
      for filename in os.listdir(subject_path):
        try:
          if(filename == ".directory"):
            continue
          filepath = os.path.join(subject_path, filename)
          im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
          
          if im is None:
            print(f"[ERROR] Could not read image {filepath}")
            continue

          print(f"[DEBUG] Image {filename}: Original size {im.shape}")
          
          if (sz is not None):
            im = cv2.resize(im, (200,200))
            print(f"[DEBUG] Image {filename}: Resized to {im.shape}")

          if len(im.shape) != 2:
            print(f"[WARNING] Image {filename} is not grayscale!")
            
          X.append(np.asarray(im, dtype=np.uint8))
          y.append(c)
          image_count += 1

        except IOError as e:
          print(f"I/O Error({e.errno}): {e.strerror}")
        except:
          print("Unexpected error:", sys.exc_info()[0])
          raise
      print(f"[DEBUG] Processed {image_count} images for {subdirname}")
      c = c+1
  
  print(f"\n[DEBUG] Total images loaded: {len(X)}")
  print(f"[DEBUG] Number of people: {c}")
  return [X, y]