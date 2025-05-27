# Hand-Eye Calibration System

A hand-eye calibration system based on FastAPI + Gradio, supporting automatic sampling, image stream, error visualization, and nonlinear optimization.



![image](https://github.com/user-attachments/assets/cc3d590d-5645-4e40-9039-497676add318)

For learning and reference purposes. Now, this only support eye on base.

## Features.

   - Supports ZED camera image acquisition and AprilTag pose estimation

   - Supports robot control (e.g., Aubo) and pose sampling

   - Supports sample selection based on optical axis angle, distance, and translation constraints

   - Supports Spectral/KMeans clustering and nonlinear optimization

   - Supports Web UI (Gradio) and API calls (FastAPI)
   - Only python

## PRPARE
1. INSTALLTION AUBO python PACK, Ref:
https://www.bilibili.com/video/BV1aM411x7Ko/?spm_id_from=333.337.search-card.all.click&vd_source=7482ad936e799333160f2fbb0cb32509
or  https://www.youtube.com/watch?v=ITNtUZekr7s
2. Note tag families
3. and others ...

## Quick Start
bash run.bash

## REFERENCES
> 1. Olga Sorkine-Hornung and Michael Rabinovich, Least-Squares Rigid Motion Using SVD
> 2. https://zhuanlan.zhihu.com/p/296329217

## TODO
Precise verifition
