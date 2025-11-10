# Face + ID Verification

This project contains two independent scripts:

- `face_recog.py` — LBPH-based face recognition (uses `Dataset/` folders)
- `object_verification.py` — SIFT + FLANN-based ID (object) verification (uses `Reference_ID/`)

Requirements

- Python 3.8+ (tests here used CPython 3.12)
- OpenCV with contrib modules (for SIFT and cv2.face)
- NumPy

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy
```

How to run

1) Face recognition (train & run):

```powershell
python face_recog.py Dataset --camera 0 --confidence 150
```

- The script will train an LBPH model from the `Dataset/` directory (one folder per person) and open the webcam.
- Press `q` to quit.

2) ID verification (SIFT + FLANN):

```powershell
python object_verification.py Reference_ID --camera 0 --size 200 200 --skip-frames 1 --memory-size 5 --stability-threshold 2 --min-matches 8 --ratio 0.7
```

- Hold a reference ID in front of the camera. The overlay will show a stable match if recognized.
- Press `q` to quit.

Optional

- `prepare_reference.py` can help resize and preprocess images into `Reference_ID/` if needed. Run it from the project folder:

```powershell
python prepare_reference.py
```

Tuning

- `--size` controls the preprocessing resolution for object matching; smaller = faster but fewer features.
- `--skip-frames` reduces CPU by processing every Nth frame.
- `--memory-size` and `--stability-threshold` control how stable a match must be before it is displayed.
- `--min-matches` and `--ratio` control the FLANN/Lowe filter strictness.

Troubleshooting

- If `SIFT_create` or `cv2.face` is missing, reinstall contrib:

```powershell
python -m pip uninstall opencv-python opencv-contrib-python -y
python -m pip install opencv-contrib-python
```

- If the camera doesn't open, try a different camera index (e.g., `--camera 1`) or close other apps using the camera.

If you want, I can: add a combined launcher script, or save preprocessed references to disk to speed startup. Let me know which next step you'd like.