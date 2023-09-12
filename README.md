# DenseSphere: Multi-Modal 3D Object Detection under Sparse Point Cloud Based on Spherical Coordinates

Jong Won Jung, Jae Hyun Yoon, Seok Bong Yoo

Abstract— Multi-modal 3D detection has gained significant
attention due to the fusion of light detection and range (LiDAR)
and RGB data. Existing 3D detection models in autonomous
navigation are typically trained using dense point cloud data
from high-spec LiDAR sensors. However, budgetary constraints
often lead to the adoption of low point-per-second (PPS)
LiDAR sensors in real-world scenarios. This issues and several
hardware defects can generate sparse point cloud. This means
that the 3D object detection models trained on dense data with a
high PPS cannot achieve optimal performance when sparse data
input. To address these problem, we propose DenseSphere, an
approach that enhances multi-modal 3D object detection with
sparse LiDAR data. Considering the data acquisition process of
LiDAR sensor, our DenseSphere involves the spherical coordi-
nate based point upsampler. The points are interpolated in the
horizontal direction of spherical coordinate using a bilateral
interpolation. The interpolated points are refined using dilated
pyramid blocks for various receptive fields. We demonstrate the
performance of DenseSphere by comparing it with other multi-
modal 3D object detection models through several experiments.

pretrained model - [DenseSpehere](https://drive.google.com/file/d/1edJFqp9LXBWVtH6aY-gt4hI8GTLyROS6/view?usp=drive_link)


![그림24](https://github.com/Jung-jongwon/DenseSphere/assets/85870991/bd03c01b-9322-4ea1-bfeb-9bac9e23b432)


## Training & Testing
```
# train
bash scripts/dist_train.sh

# test
bash scripts/dist_test.sh
