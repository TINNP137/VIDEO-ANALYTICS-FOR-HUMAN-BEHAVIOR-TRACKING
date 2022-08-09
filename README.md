
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
  We use this image

  ![Cafeteria no people](https://github.com/TINNP137/VIDEO-ANALYTICS-FOR-HUMAN-BEHAVIOR-TRACKING/blob/main/data/helpers/cafetefria_no_people.png)

  Run the code below to fill the roi 
  
  ** don't forget to edit some variable in the code to suit your project
  ```bash
    python check_roi.py
  ```
  Fill the polygon result: 

  ![Cafeteria ground](https://github.com/TINNP137/VIDEO-ANALYTICS-FOR-HUMAN-BEHAVIOR-TRACKING/blob/main/data/helpers/cafetefria_ground.png)


## Image masking
  Find the right hsv value to mask an image 
  ```bash
    python HSV_color_thresholder.py
  ```
  Result:

  ![Masking](https://github.com/TINNP137/VIDEO-ANALYTICS-FOR-HUMAN-BEHAVIOR-TRACKING/blob/main/data/helpers/cafeteria_hsv.png)
  Then write down the lower range and upper range of the HSV value. We're using it in the following file
  ```bash
    python thresh_floor_green.py
  ```
  Result:

  ![Masking](https://github.com/TINNP137/VIDEO-ANALYTICS-FOR-HUMAN-BEHAVIOR-TRACKING/blob/main/data/helpers/cafeteria_mask.png)

## Running the Tracker with YOLOv4
  ```bash
    # Run yolov4 deep sort object tracker on video
    python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

    # Run yolov4 deep sort object tracker on webcam (set video flag to 0)
    python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4
  ```
  The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

  If you want to run yolov3 set the model flag to ``--model yolov3``, upload the yolov3.weights to the 'data' folder and adjust the weights flag in above commands. (see all the available command line flags and descriptions of them in a below section)
## Running the Tracker with YOLOv4-Tiny (We use Yolov4-tiny in this project)
  The following commands will allow you to run yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the 'data' folder in order for commands to work!

  ```bash
    # Run yolov4 deep sort object tracker on video
    python object_tracker_wheatmap_clean.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cafeteria.mp4 --output ./outputs/cafeteria_out_video_tiny.avi --tiny
  ```

## Resulting Video
  As mentioned above, the resulting video will save to wherever you set the --output command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the --output_format flag, by default it is set to AVI codec which is XVID.

  [Result](https://youtu.be/jsapjx3F_PM)

## Our group members
  TINN PONGJITUMPAI - [TINNP137](https://github.com/TINNP137)
  
  NAPAT CHEEPMUANGMAN - [nc-mannequin](https://github.com/nc-mannequin)

