
# Video-Analytics-For-Human-Behavior-Tracking

Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. The code is inherited from [yolov4-deepsort by theAIGuysCode](https://github.com/theAIGuysCode/yolov4-deepsort). We assume the movement(walking person) for generating a motion heatmap by the movement in the bottom area of a person's bounding box.


## Installation

  #### Conda (Recommended)
   If any error occurs use opencv-contrib-python instead of opencv-python
  ```bash
    conda create --name myenv
    conda activate myenv
    pip install -r requirements.txt
  ```
## Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
  Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository. https://developer.nvidia.com/cuda-10.1-download-archive-update2   

## Downloading Official YOLOv4 Pre-trained Weights
- [YOLOv4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [YOLOv4-tiny](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

## Running the Tracker with YOLOv4-tiny

