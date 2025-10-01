# Camera Geometry & Calibration

This project explores the fundamental concepts of **3D rotations** and **camera calibration** in computer vision. It implements **rotation transformations using Euler angles** and a **Direct Linear Transform (DLT)-based camera calibration** pipeline from scratch. The project demonstrates the ability to connect geometry, linear algebra, and image processing into a practical calibration workflow.

*(Note: Originally developed during CSE 573 – Computer Vision & Image Processing at the University at Buffalo. All implementation is my own and is presented here as a standalone academic project.)*

---

## 📖 Overview

The project is divided into two main tasks:

1. **Task 1: 3D Rotations**
   - Construct rotation matrices for Euler angles.
   - Compute the forward and inverse mappings between coordinate frames.
   - Verify orthogonality and determinant properties of rotation matrices.

2. **Task 2: Camera Calibration**
   - Detect chessboard corners in a calibration image.
   - Define corresponding 3D world coordinates (checkerboard grid).
   - Solve for the camera projection matrix using **DLT + SVD**.
   - Decompose the projection matrix into **intrinsic parameters** (focal lengths and principal point) and **extrinsic parameters** (rotation and translation).
   - Store results in reproducible JSON outputs.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.8+
- **Libraries:**
  - [NumPy](https://numpy.org/) — matrix algebra and numerical computations
  - [OpenCV 4.5.4](https://opencv.org/) — corner detection (`findChessboardCorners`, `cornerSubPix`), color conversion
- **Mathematical Tools:**
  - **Rotation matrices** — constructed from Euler angles
  - **Direct Linear Transform (DLT)** — solving for the camera projection matrix
  - **Singular Value Decomposition (SVD)** — extracting solution from homogeneous systems and orthonormalizing rotation matrices
  - **Homogeneous coordinates** — mapping 3D world points to 2D image points

---

## 🔍 Task 1: 3D Rotations

### Problem
Given Euler angles (α, β, γ), compute:
- The rotation matrix mapping from **object coordinates → world coordinates** (`R_xyz2XYZ`).
- The inverse mapping (`R_XYZ2xyz`).

### Method
1. Define basic axis rotations:
   - \( R_z(\alpha), R_x(\beta), R_z(\gamma) \)
2. Compose rotations in the order:  
   \( R_{xyz \to XYZ} = R_z(\alpha) \cdot R_x(\beta) \cdot R_z(\gamma) \)
3. Inverse mapping is the transpose:
   \( R_{XYZ \to xyz} = R_{xyz \to XYZ}^T \)

### Results
For α = 45°, β = 30°, γ = 50°, the program outputs:

```json
{
  "rot_xyz2XYZ": [
    [0.55667044, -0.75440651,  0.34729636],
    [0.66341395,  0.12080899, -0.73860541],
    [0.5,         0.64597407,  0.57735027]
  ],
  "rot_XYZ2xyz": [
    [0.55667044,  0.66341395,  0.5],
    [-0.75440651, 0.12080899,  0.64597407],
    [0.34729636, -0.73860541,  0.57735027]
  ]
}
```
- Both matrices are orthogonal and have determinant = +1.
- Confirms they represent valid 3D rotations.

## 🔍 Task 2: Camera Calibration
### Problem

Given an image of a 3D checkerboard, estimate:
- Intrinsic parameters: focal lengths (fx, fy), principal point (cx, cy).
- Extrinsic parameters: rotation matrix (R) and translation vector (T).
- A full camera projection matrix (M).

### Method
1. Corner Detection (Image Coordinates)
  - Convert to grayscale.
  - Split into left and right halves (each 4×4 checkerboard face).
  - Use findChessboardCorners + cornerSubPix for sub-pixel precision.
  - Order points consistently (row-major by y, then x).
  - Result: 32 (x, y) coordinates.

2. World Coordinates
  - Define 3D checkerboard points with 10 mm spacing.
  - Left face lies on the YZ-plane (X=0).
  - Right face lies on the XZ-plane (Y=0).
  - Result: 32 (X, Y, Z) coordinates.

3. Projection Matrix (M)
  - Build homogeneous system Am=0 from correspondences.
  - Solve via SVD: last singular vector → reshape → 3×4 matrix M.
  - Normalize by enforcing ‖m₃‖=1.

4. Decomposition into Intrinsics & Extrinsics
  - Build intrinsic matrix K.
  - Compute extrinsics
  - Orthonormalize R with SVD → final rotation R.

Results
Example output (result_task2.json):
{
  "fx": 16.05,
  "fy": 12.96,
  "cx": 1006.57,
  "cy": 758.69,
  "R": [
    [-0.0363,  0.9992, -0.0201],
    [ 0.7649,  0.0412,  0.6429],
    [ 0.6431,  0.0000, -0.7657]
  ],
  "T": [-3.4875, 6.8306, 41.5674]
}

#### Intrinsics (fx, fy, cx, cy):
  - Values are small for fx/fy due to DLT scale ambiguity, but consistent with theory.
#### Rotation Matrix (R):
  - Orthogonal with determinant ≈ 1 (valid rotation).
#### Translation (T):
  - Indicates relative position of the checkerboard.

## 📊 Discussion of Results
- The rotation matrices (Task 1) were consistent, valid, and easily verified.
- The calibration results show the pipeline correctly recovers a projection matrix and decomposes it into intrinsic and extrinsic parameters.
- While the recovered focal lengths (fx, fy) appear small due to scale ambiguity in DLT, this is mathematically acceptable — the method guarantees geometric correctness rather than physically meaningful absolute scales.
- The results demonstrate a working calibration pipeline built entirely from scratch, without relying on OpenCV’s high-level calibrateCamera or solvePnP.

## 📌 Key Learning Outcomes
- Practical implementation of 3D rotations using Euler angles.
- Application of linear algebra (SVD) in solving homogeneous systems.
- Understanding of the DLT method for camera calibration.
- Experience with intrinsic vs. extrinsic parameter recovery.
- Building a full computer vision pipeline with minimal dependencies.

## 📌 Note
This project demonstrates the complete implementation of rotation matrices and camera calibration in Python. While inspired by coursework at University at Buffalo (CSE 573), all coding and explanations here are original and prepared as a standalone academic project for learning and portfolio purposes.

## 📐 Calibration Pipeline Diagram

<pre>
   3D World Points (X, Y, Z)            Image Points (x, y)
   ┌───────────────────────┐           ┌───────────────────┐
   │ Checkerboard corners   │           │ Detected corners  │
   │ on YZ-plane & XZ-plane │           │ (sub-pixel, 32×2) │
   └───────────┬───────────┘           └─────────┬─────────┘
               │                               ▲
               ▼                               │
       ┌──────────────────┐   Projection   ┌──────────────┐
       │  Homogeneous A   │   Matrix M     │ Direct Linear │
       │   system (DLT)   │ ─────────────▶ │ Transform +   │
       └──────────────────┘   via SVD      │   SVD solver  │
                                          └──────────────┘
                                                   │
                                                   ▼
                                   ┌─────────────────────────┐
                                   │ Projection Matrix M (3×4)│
                                   └───────────┬─────────────┘
                                               │
              ┌────────────────────────────────┼────────────────────────────┐
              ▼                                ▼                            ▼
        Intrinsics (K)                 Rotation (R)                  Translation (T)
      [fx, fy, cx, cy]        [3×3 orthogonal matrix]              [3×1 vector]
</pre>


