
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

## Convert YOLO weights file to TensorFlow weights file
  To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder.
  ```bash
    # Convert darknet weights to tensorflow model
    python save_model.py --model yolov4 
    # save yolov4-tiny model
    python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny
  ```
## Get ROI (Region of Interest) coordinate
  We use [Makesense](https://www.makesense.ai/) to draw a polygon and the polygon coordinate
  - select object detection
  - select polygon
  - export annotation as "VGG JSON" or "COCO JSON"
## Fill the ROI polygon
  Run the code below to fill the roi 
  ** don't forget to edit some variable in the code to suit your project
  ```bash
  python check_roi.py
  ```

