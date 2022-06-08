# Computer Vision Taxonomy
2022년 2월 기준으로 survey 논문들을 읽어보면서 정리하였다.

1. Problem definition
2. Outputs
3. Examples
4. Algorithms (two or three states-of-the-art)
5. Detail algorithm
6. Benchmark datasets
7. Metrics

# Overview
|분류|문제 정의|출력|알고리즘|데이터셋|성능지표|
|---|---|---|---|---|---|
|Image classification|• 주어진 영상을 사전에 정해진 레이블(label) 중 하나로 분류|Image label|DynamicViT ('21)<br/>Swin ('21)<br/>EfficientNet ('19)<br/>ResNet ('15)<br/>VGG-Net ('14)|[ImageNet](https://www.image-net.org/update-mar-11-2021.php)<br/>[CIFAR-10, CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)|Accuracy (%)|
|[Object detection](object_detection/object_detection.md)|• 주어진 영상에 있는 객체들의 종류와 위치를 파악하는 작업|Bounding box<br/>Confidence|DyHead ('21)<br/>Swin ('21)<br/>YOLO ('21)<br/>FCOS ('19)<br/>CenterNet ('19)<br/>R-CNN ('17)<br/>SSD ('15)|[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)<br/>[COCO](https://cocodataset.org/#home)<br/>[UA-DETRAC](https://detrac-db.rit.albany.edu/)|mAP (%)|
|[Object recognition](object_recognition/object_recognition.md)|• 입력된 두 영상 내 객체의 동일성을 검증<br/>• 영상 내 객체가 DB에 저장된 어떤 객체와 가장 유사한지 식별|True or false|[Prodpoly ('20)](https://openaccess.thecvf.com/content_CVPR_2020/html/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.html)<br/>[Circle Loss ('20)](https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.html)<br/>[FaceNet + Adaptive Threshold ('19)](https://ieeexplore.ieee.org/abstract/document/8695322?casa_token=qJa0Ur26l6sAAAAA:uKki0hc6EYcYHbvz0t-uNA0zppSrYU5cMycuCYEtHNBqoCzY8mmrE_JxfBocHr2cMpDk-gY)<br/>[MTCNN ('16)](https://ieeexplore.ieee.org/abstract/document/7553523?casa_token=YnK0xWzupJQAAAAA:qtls0dKBjlcCECEcJ5vDNPKSMIWWpOYe9wD9ManrBntFkxbloAZ4sqHB04FEG81LeyP9Ghg)<br/>[FaceNet ('15)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)|[LFW](http://vis-www.cs.umass.edu/lfw/)<br/>CASIA WebFace<br/>[VGGFace, VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)<br/>[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)<br/>[CFP-FP](http://www.cfpw.io/)<br/>[YTF](https://www.cs.tau.ac.il/~wolf/ytfaces/)|Accuracy (%)|
|[Object tracking](object_tracking/object_tracking.md)|• 동영상에서 움직이는 객체에 할당된 고유한 레이블을 유지|Weighted bipartite graph|[ByteTrack ('21)](https://arxiv.org/abs/2110.06864)<br/>[Fair MOT ('21)](https://link.springer.com/article/10.1007/s11263-021-01513-4)<br/>[JDE ('20)](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_7)<br/>[DMAN ('18)](https://openaccess.thecvf.com/content_ECCV_2018/html/Ji_Zhu_Online_Multi-Object_Tracking_ECCV_2018_paper.html)<br/>[DeepSort ('17)](https://ieeexplore.ieee.org/abstract/document/8296962?casa_token=edaO2S52z00AAAAA:CI8JyoCAMz_8I5I-jqt06hqvDwEa6aOW7RX7WH6vKMGDXCTG5uKReib5fknT4Rvg6etZKDQ)<br/>[Sort ('16)](https://ieeexplore.ieee.org/abstract/document/7533003?casa_token=p9sWs2QKqcsAAAAA:nzlZPHy7t5bW9akeK9UuX_zQlCg0KbaBkVrt20QIwMd_oxteADaaIdQBJHkJ31puPOUh8eg)<br/>[POI ('16)](https://scholar.google.co.kr/scholar?hl=ko&as_sdt=0%2C5&q=Poi%3A+Multiple+object+tracking+with+high+performance+detection+and+appearance+feature.&btnG=)|[MOTChallenge](https://motchallenge.net/)<br/>[KITTI](http://www.cvlibs.net/datasets/kitti/)<br/>[UA-DETRAC](https://detrac-db.rit.albany.edu/)|MOTA (%)<br/>IDF1|
|[Object re-identification](object_reid/object_reid.md)|• 다수의 카메라 혹은 동일한 카메라이지만 다른 상황에서 촬영된 영상 내 동일한 객체를 식별|True or false|**Person Re-ID**<br/>[CIL ('21)](https://arxiv.org/abs/2111.00880)<br/>[AGW ('21)](https://ieeexplore.ieee.org/abstract/document/9336268?casa_token=H_qQ2w8_u_sAAAAA:teKZXA44HwESMJBOvr0vxZhMxlpMGPr_CXNQfXRdGGmBhjmY-G1w5ayfciUWqPoYL-i-PWk)<br/>[Auto-ReID ('19)](https://openaccess.thecvf.com/content_ICCV_2019/html/Quan_Auto-ReID_Searching_for_a_Part-Aware_ConvNet_for_Person_Re-Identification_ICCV_2019_paper.html)<br/>[Pyramid ('19)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Pyramidal_Person_Re-IDentification_via_Multi-Loss_Dynamic_Training_CVPR_2019_paper.html)|***Person Re-ID***<br/>[Market-1501](https://www.kaggle.com/pengcw1/market-1501/data)<br/>[CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)|mAP (%)<br/>CMC-k (%)<br/>mINP|
||||***Vehicle Re-ID***<br/>PROVID ('18)|***Vehicle Re-ID***<br/>[VeRi-776](https://vehiclereid.github.io/VeRi/)<br/>[VeRi-Wild](https://github.com/PKU-IMRE/VERI-Wild)<br/>[PKU VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html)<br/>[Vehicle-1M](http://www.nlpr.ia.ac.cn/iva/homepage/jqwang/Vehicle1M.htm)<br/>[VRIC](https://qmul-vric.github.io/)||
|Pose estimation|• 영상 내 객체의 자세를 추정|Joint positions|OpenPose ('18)<br/>PoseNet ('17)||pCk|
|Action recognition|• 영상 내 객체의 동작을 인식|Action label|SMART<br/>Two stream|[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)|3-fold accuracy (%)|
|Semantic segmentation|• 영상 내에 동일한 객체에 속하는 픽셀에 동일한 레이블을 할당|Pixel-wise labels|DeepLab<br/>HRNet<br/>Lawin<br/>EfficientPs|[Cityscapes](https://www.cityscapes-dataset.com/)<br/>[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)|Mean IoU (%)|
|Image generation||||||
|Image denoising||||||
|Super-resolution||||||
|Image retrieval||||||