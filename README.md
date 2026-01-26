# PlugToPlace 
**Indoor Multimedia Geolocation from Electrical Sockets for Digital Investigation**

# Abstract
Computer vision is a rapidly evolving field, giving rise to powerful new tools and techniques in digital forensic investigation, and shows great promise for novel digital forensic applications. One such application, indoor multimedia geolocation, has the potential to become a crucial aid for law enforcement in the fight against human trafficking, child exploitation, and other serious crimes. While outdoor multimedia geolocation has been widely explored, its indoor counterpart remains underdeveloped due to challenges such as similar room layouts, frequent renovations, visual ambiguity, indoor lighting variability, unreliable GPS signals, and limited datasets in sensitive domains. This repository provides the Google Colab training notebooks and the trained model weights for each of the steps of proposed framework. 

# Datasets
To address data scarcity, two dedicated datasets were created: 
1. Socket Detection: DatasetA – consisting of 2,328 annotated images, expanded to DatasetB –  4,074 through augmentation.
2. Socket Type Classification: It contains 3,187 images across 12 different socket type classes.
Both datasets can be used to train deep learning models. The datasets for this project are available via [Roboflow link 1.](https://app.roboflow.com/objsocket/plugtoplace_socketdetection_db/models), [Roboflow link 2.](https://app.roboflow.com/objsocket/12plugsockettype/1). Please cite the dataset if you use it in your work. 
3. Geolocation Inference: The pipeline was evaluated on the Hotels-50K dataset (TraffickCam) subset of crowd-sourced hotel images, which capture real-world conditions such as poor lighting and amateur angles. This dataset provides a more realistic evaluation than using professional, well-lit, often wide-angle images from travel websites. The dataset can be downloaded from the [Hotels-50K Github Repo](https://github.com/GWUvision/Hotels-50K).  

# Methodology
Our proposed methodology is a three-tiered computer vision framework (see Figure). It enables the automated detection of electrical sockets. These sockets serve as consistent indoor markers for geolocation, as plug types are standardized by country or region. In Stage 1, we conducted a comparative analysis of various YOLO versions YOLOv11 using K-fold cross-validation achieved the highest performance with an mAP@0.5 of 0.843.
Once a socket is detected, the cropped Region of Interest (ROI) is passed to Stage 2 for classification. We evaluated five state-of-the-art CNN architectures: VGG16, InceptionV3, Xception, ResNet50, and ResNet101. Among these, Xception performed best, classifying sockets into one of 12 types with an accuracy of 0.912. Finally, Stage 3 performs geolocation inference by mapping the identified socket types to specific countries, achieving 0.96 accuracy at a confidence threshold of >90%. The pipeline was validated using the TraffickCam subset of the Hotels-50K dataset, demonstrating a practical application for digital forensics. 

![Alt ](/Fig/3_stage_updated2.png)

## Result
A total of 44,630 TraffickCam images were processed through the algorithmic pipeline. In the first stage, YOLO detected 3,759 potential sockets. To enhance detection accuracy and eliminate false positives, a second-stage classifier was employed to identify and remove noise. Specifically, instances where non-socket objects (e.g., switchboards) were incorrectly detected as sockets in the first stage were classified as noise. This step identified 1,393 noisy detections, leaving 2,366 valid sockets. At a threshold above 70\%, 1,595 predictions were correct, and 146 were incorrect, resulting in an accuracy of 91.61\%. Increasing the threshold above 80\% reduced the number of correct detections to 1,421, while incorrect detections decreased to 95, yielding an improved accuracy of 93.73\%. At the highest threshold of above 90\%, correct detections further decreased to 1,167, with only 45 incorrect predictions, resulting in the highest accuracy of 96.29\%.

 
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


Note: For comprehensive details on the methodology and experimental results, please refer to our research paper. For inquiries, feel free to contact the authors.

# Authors 
Kanwal Aftab, Graham Adams, Mark Scanlon
