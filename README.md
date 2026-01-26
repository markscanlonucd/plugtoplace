# PlugToPlace 
**Indoor Multimedia Geolocation from Electrical Sockets for Digital Investigation**
# Introduction
We present a three-tiered Computer Vision framework for automated  detection of electrical sockets that as consistent indoor markers for geolocation, since plug socket types are standardised by country or region. The three-stage deep learning pipeline detects plug sockets (YOLOv11, mAP@0.5 = 0.843), classifies them into one of 12 plug socket types (Xception, accuracy = 0.912), and maps the detected socket types to countries (accuracy = 0.96  at >90 % threshold confidence). To address data scarcity, two dedicated datasets were created: socket detection dataset of 2328 annotated images expanded to 40742 through augmentation, and a classification dataset of 3187 images across 12 plug socket classes. The pipeline was evaluated on the Hotels-50K dataset, focusing on the TraffickCam subset of crowd-sourced hotel images, which capture real-world conditions such as poor lighting and amateur angles. This dataset provides a more realistic evaluation than using professional, well-lit, often wide-angle images from travel websites. This framework demonstrates a practical step toward real-world digital forensic applications. 

![Alt ](/Fig/3_stage_updated2.png)

# Abstract

Computer vision is a rapidly evolving field, giving rise to powerful new tools and techniques in digital forensic investigation, and shows great promise for novel digital forensic applications. One such application, indoor multimedia geolocation, has the potential to become a crucial aid for law enforcement in the fight against human trafficking, child exploitation, and other serious crimes. While outdoor multimedia geolocation has been widely explored, its indoor counterpart remains underdeveloped due to challenges such as similar room layouts, frequent renovations, visual ambiguity, indoor lighting variability, unreliable GPS signals, and limited datasets in sensitive domains.

# Trained Models and Datasets

The dataset used in this project can be downloaded from the this GitHub repository or can be obtained from this Roboflow link, and we kindly request that you cite it when using it.

There are two datasets available, which can be utilized for training deep learning models for various tasks:

- Plug Socket Detection: A comparative analysis of different YOLO versions has been conducted. The Google Colab training file, dataset, YOLOv inference code, and trained model weights are all included in the '1. Socket Detection' folder. 
- Plug Socket Type Classification:Once the socket is detected in Step 1, the cropped ROI is passed for socket type classification. Five state-of-the-art CNN architectures were implemented: VGG16, InceptionV3, Xception, ResNet50, and ResNet101. A comparative analysis of these five CNNs was performed; for more details, please refer to the research paper. The Google Colab training code, dataset used for training, and model weights can be found in the '2. Socket Type Classification' folder.
- Geolocation Inference: The test dataset for this experiment was derived from the Hotels-50K TraffickCam dataset, available from its official GitHub repository.


## Requirements
- Python > 3.7
- torch==2.6.0
- torchvision==0.21.0
- ultralytics==8.3.192
- ultralytics-thop==2.0.14
- opencv-python==4.12.0
- pillow==11.0.0
- numpy==2.4.1
- scipy==1.14.1
- PyYAML==6.0.2
- tqdm==4.67.1
- requests==2.32.3
- urllib3==2.2.3
- certifi==2024.8.30


# Authors 
Kanwal Aftab, Graham Adams, Mark Scanlon
