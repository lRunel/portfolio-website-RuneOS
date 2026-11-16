# ARWear — Real-Time Augmented Reality Garment Try-On

## Overview
ARWear enables instant, photorealistic 3D garment try-ons directly in the browser and mobile devices using MediaPipe pose landmark detection and real-time mesh deformation.

## Key Features
- **33 Pose Landmarks:** Uses 33 body landmarks to track torso, shoulders, and waist rotations with 60 FPS performance.
- **Deformable 3D Mesh:** Physics-based rigging aligns custom 3D cloth models (e.g. ) with live user posture.
- **Occlusion Handling:** Real-time limb depth ordering to prevent clipping artifacts during movement.
- **Lightweight WebGL/Three.js Pipeline:** Runs entirely client-side without cloud video transmission.
