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

### Pretrained model - 

- Horizontal(subsampling = 8) [DenseSphere(horizontal)](https://drive.google.com/file/d/1edJFqp9LXBWVtH6aY-gt4hI8GTLyROS6/view?usp=drive_link)

- Vertical(channels = 16) [DenseSphere(vertical)](https://drive.google.com/file/d/1CmHmd0E_4qSsdX1cdQlzspn04qO7E4xe/view?usp=drive_link)
### Comparison of visual detection performance on the KITTI validation dataset

![fig_hor_8_det](https://github.com/Jung-jongwon/DenseSphere/assets/85870991/899a0d7d-99bc-4903-aafc-738a1b6824ac)
Comparison of visual detection performance on the KITTI validation dataset at an s_f of 8 in horizontal resolution. (a) Visual detection results of TED-M. (b) Visual detection results of VirConv-S. (c) Visual detection results of DenseSphere.

![fig_hor_16_det](https://github.com/Jung-jongwon/DenseSphere/assets/85870991/dfddc14a-633a-4b84-8f31-e83600a89151)
Comparison of visual detection performance on the KITTI validation dataset at an s_f of 16 in horizontal resolution. (a) Visual detection results of TED-M. (b) Visual detection results of VirConv-S. (c) Visual detection results of DenseSphere.

![fig_ver_32_det](https://github.com/Jung-jongwon/DenseSphere/assets/85870991/ea345159-3937-41de-9b5f-bbfd40a7fb79)
Comparison of visual detection performance on the KITTI validation dataset at a ch of 32 in vertical resolution. (a) Visual detection results of TED-M. (b) Visual detection results of VirConv-S. (c) Visual detection results of DenseSphere.

![fig_ver_16_det](https://github.com/Jung-jongwon/DenseSphere/assets/85870991/ccfb6abb-1916-4b6d-860d-704801faff4d)
Comparison of visual detection performance on the KITTI validation dataset at a ch of 16 in vertical resolution. (a) Visual detection results of TED-M. (b) Visual detection results of VirConv-S. (c) Visual detection results of DenseSphere.





## Training & Testing
```
# train
bash scripts/dist_train.sh

# test
bash scripts/dist_test.sh

