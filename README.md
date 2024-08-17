# Traffic management Using Intelligent Techniques
## Overview
This project focuses on improving traffic management, and solving complex traffic scenarios by using advanced deep learning techniques. Utilizing YOLOv4 and Convolutional Neural Networks (CNNs), the project aims to develop a sophisticated system for real-time traffic detection and management. 
## Dataset
CCTV footage from the New Zealand Transport Agency, serve as the foundation for training and evaluating the system. This dataset was taken from the paper "Towards Real-time Traffic Flow Estimation using YOLO and SORT from Surveillance Video Footage".
## Workflow
Core elements of the project is object detection with YOLOv4, object tracking using the SORT algorithm, and real-time communication via Twilio for alert notifications.
Using the Google Colab ipynb file the weights training was done for model detection, after getting weights, using OpenCV vehicle detection was done and for tracking Sort Algorithm was used. After crossing the threshold for traffic congestion, an alert was send via SMS using twillio API. 
## Future works
Future enhancements will aim to scale the system, incorporate advanced algorithms like YOLOv8, and expand its capabilities to include pedestrian and cyclist traffic management, integration with smart city technologies, and environmental monitoring, thus offering a comprehensive solution to modern urban traffic management challenges. 
