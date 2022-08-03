import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')



import cmapy
import copy
import os
import re


def create_roi(pts,image):
    pts = np.array(pts).astype(np.int32)      # <-- to int
    # Draw polygon lines
    image = cv2.polylines(image,[pts],isClosed=False,color=(0, 255, 40),thickness=2,lineType=cv2.LINE_AA)



def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    # nn_budget = None
    nn_budget = 2
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)





    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)


    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    region_left = [(10,250),(10,1010),(200,1010),(200,250)]
    region_right = [(1710,250),(1710,1010),(1900,1010),(1900,250)]

    # region_floor1=[(1694,906),(1703,825),(1773,980),(1811,1080),(525,1078),(390,1076),(428,1012),(463,981),(476,980),(524,1073),(538,1062),(485,972),(489,957),(527,935),(570,904),(588,903),(588,891),(578,884),(480,943),(458,898),(469,878),(488,886),(493,878),(477,871),(493,863),(483,852),(604,637),(613,628),(587,633),(596,618),(583,568),(745,350),(808,353),(843,331),(908,323),(922,316),(935,282),(1031,287),(1037,274),(1101,274),(1109,300),(1063,316),(1054,338),(1059,363),(1093,371),(1055,381),(1045,371),(1045,299),(1034,288),(997,314),(1030,409),(1011,426),(1000,510),(1016,509),(1029,426),(1034,422),(1031,467),(1044,468),(1047,419),(1100,419),(1086,437),(1096,454),(1090,505),(1024,507),(1001,522),(946,532),(934,533),(929,574),(918,513),(889,525),(848,589),(853,619),(886,676),(917,717),(887,724),(867,855),(886,860),(883,851),(898,749),(915,730),(914,754),(927,756),(929,726),(950,741),(978,888),(994,881),(990,866),(993,756),(1009,761),(1007,747),(1017,744),(1030,762),(1045,825),(1042,863),(1049,865),(1054,865),(1065,864),(1070,854),(1060,830),(1066,802),(1111,825),(1081,860),(1088,864),(1065,979),(1080,981),(1079,976),(1100,864),(1131,825),(1186,818),(1224,838),(1248,952),(1260,942),(1242,837),(1237,823),(1210,810),(1252,731),(1267,715),(1275,715),(1279,766),(1268,795),(1276,825),(1260,880),(1275,888),(1290,839),(1304,853),(1300,877),(1288,890),(1229,1014),(1248,1015),(1300,902),(1331,874),(1397,872),(1427,901),(1432,1020),(1448,1018),(1445,900),(1423,868),(1460,820),(1464,809),(1499,751),(1493,732),(1443,741),(1458,691),(1507,691),(1502,754),(1512,753),(1515,692),(1529,690),(1513,759),(1524,755),(1542,692),(1563,690),(1553,705),(1552,725),(1515,801),(1528,810),(1558,736),(1600,774),(1580,808),(1526,912),(1540,920),(1595,810),(1612,786),(1641,780),(1635,810),(1647,809),(1653,781),(1674,780),(1665,828),(1681,829),(1689,798),(1693,821),(1680,903)]
    # region_floor2=[(1305,390),(1302,421),(1261,421),(1270,388)]
    # region_floor3=[(1191,251),(1233,254),(1231,299),(1199,296),(1202,264)]
    # region_floor4=[(1326,284),(1346,283),(1351,300),(1322,310),(1318,307)]
    # region_floor5=[(1227,381),(1243,420),(1259,389),(1247,385),(1252,377)]
    # region_floor6=[(1218,397),(1209,409),(1218,467),(1227,467)]
    # region_floor7=[(1253,478),(1229,417),(1232,474)]



    # store ID crossing the region
    left_region_ids =set() # all in the area cumulative
    right_region_ids = set() # all in the area cumulative
    floor_ids = set()

    # in_left_region = cv2.pointPolygonTest(left_region,( (int((bbox[0] + bbox[2])/2)),(int(((bbox[1]+bbox[3])/2)))    ),False)
    # in_right_region = cv2.pointPolygonTest(right_region,( (int((bbox[0] + bbox[2])/2)),(int(((bbox[1]+bbox[3])/2)))    ),False)
    # if in_left_region >0:
    #     left_region_ids.add(track.track_id)
    # if in_right_region>0:
    #     right_region_ids.add(track.track_id)


    # background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=10,varThreshold=50,detectShadows=False) #For python3
    first_iteration_indicator = 1


    img = cv2.imread('./data/video/cafetefria_ground.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_channels = cv2.split(hsv)

    rows = img.shape[0]
    cols = img.shape[1]

    for i in range(0, rows):
        for j in range(0, cols):
            h = hsv_channels[0][i][j]

            if h > 35 and h < 80:
                hsv_channels[2][i][j] = 255
            else:
                hsv_channels[2][i][j] = 0



    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()








        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']
        #print(allowed_classes)

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        # if FLAGS.count:
        #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        # # select ROI # Region of interest  # just draw for visual
        # # left_region = cv2.rectangle(frame,(10,750),(200,1010),(0,240,15),thickness=3)
        # # right_region = cv2.rectangle(frame,(1710,750),(1900,1010),(72,140,25),thickness=3)

        # create_roi(region_floor1,frame)
        # create_roi(region_floor2,frame)
        # create_roi(region_floor3,frame)
        # create_roi(region_floor4,frame)
        # create_roi(region_floor5,frame)
        # create_roi(region_floor6,frame)
        # create_roi(region_floor7,frame)


        left_region = cv2.rectangle(frame,(10,250),(200,1010),(0,240,15),thickness=3)
        right_region = cv2.rectangle(frame,(1710,250),(1900,1010),(72,140,25),thickness=3)

        # region_left = [(10,250),(10,1010),(200,1010),(200,250)]
        # region_right = [(1710,250),(1710,1010),(1900,1010),(1900,250)]
        #
        # # store ID crossing the region
        # left_region_ids =set() # all in the area currently
        # right_region_ids = set() # all in the area currently

        # in_left_region = cv2.pointPolygonTest(left_region,( (int((bbox[0] + bbox[2])/2)),(int(((bbox[1]+bbox[3])/2)))    ),False)
        # in_right_region = cv2.pointPolygonTest(right_region,( (int((bbox[0] + bbox[2])/2)),(int(((bbox[1]+bbox[3])/2)))    ),False)
        # if in_left_region >0:
        #     left_region_ids.add(track.track_id)
        # if in_right_region>0:
        #     right_region_ids.add(track.track_id)





        #print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        height, width = frame.shape[:2]
        try_frame = np.zeros((height, width, 1), dtype = "uint8")
        try_frame2 = copy.deepcopy(frame)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()

            # bbox[1]=bbox[3]*0.9 # try to put bbox to foot
            # print("bbox =" + str(bbox))

            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[3]*0.9)), (int(bbox[2]), int(bbox[3])), color, 2)  #focus on foot
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.putText(frame,'chance'+ str(scores) + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-30)),0, 0.75, (255,255,255),2)


            #nui estimate foot area
            cv2.rectangle(try_frame, (int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - bbox[1])/7))) , (int(bbox[2]), int(bbox[3])), (255, 255, 255), -1)



            # if id cross in ROI
            in_left_region = cv2.pointPolygonTest(np.array(region_left),( (int((bbox[0] + bbox[2])/2)),(int(((bbox[1]+bbox[3])/2)))    ),False)
            in_right_region = cv2.pointPolygonTest(np.array(region_right),( (int((bbox[0] + bbox[2])/2)),(int(((bbox[1]+bbox[3])/2)))    ),False)

            # most_ground1 = cv2.pointPolygonTest(np.array(region_floor1),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7 *6) )))     ,False)
            # most_ground2 = cv2.pointPolygonTest(np.array(region_floor2),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7))))     ,False)
            # most_ground3 = cv2.pointPolygonTest(np.array(region_floor3),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7))))     ,False)
            # most_ground4 = cv2.pointPolygonTest(np.array(region_floor4),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7))))     ,False)
            # most_ground5 = cv2.pointPolygonTest(np.array(region_floor5),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7))))     ,False)
            # most_ground6 = cv2.pointPolygonTest(np.array(region_floor6),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7))))     ,False)
            # most_ground7 = cv2.pointPolygonTest(np.array(region_floor7),( int(bbox[0]), int(int(bbox[3]) - int(int(bbox[3] - int(bbox[1])/7))))     ,False)




            if in_left_region >0:
                left_region_ids.add(track.track_id)
            if in_right_region>0:
                right_region_ids.add(track.track_id)
            # if most_ground1 >0:
            #     floor_ids.add(track.track_id)
            # if most_ground2 >0:
            #     floor_ids.add(track.track_id)
            # if most_ground3 >0:
            #     floor_ids.add(track.track_id)
            # if most_ground4 >0:
            #     floor_ids.add(track.track_id)
            # if most_ground5 >0:
            #     floor_ids.add(track.track_id)
            # if most_ground6 >0:
            #     floor_ids.add(track.track_id)
            # if most_ground7 >0:
            #     floor_ids.add(track.track_id)







        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


        # Track ID stored from person
        # cv2.putText(frame,"left" + str(left_region_ids), (5, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 165, 255), 1)
        # cv2.putText(frame,"right" + str(right_region_ids), (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 165, 255), 1)


        # If first frame
        if first_iteration_indicator == 1:

            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            # accum_image = np.zeros((int(height), width), np.uint8)
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
            pre_frame = background_subtractor.apply(first_frame)
            cv2.imwrite('./outputs/frame-18072022-07.jpg', pre_frame)
        else:


            # filter = background_subtractor.apply(frame)  # remove the background
            # cv2.imshow('filter',filter)

            ## tinn
            # filter = background_subtractor.apply(frame)  # remove the background
            # cv2.imshow('filter',filter)
            ## nui
            #filter = background_subtractor.apply(frame)  # remove the background
            masked = background_subtractor.apply(try_frame2)  # remove the background
            filter = cv2.absdiff(pre_frame, masked)
            filter = cv2.subtract(filter, pre_frame)
            pre_frame = masked
            cv2.imshow('filter',filter)
            # cv2.imwrite('./outputs/frame-18072022.jpg', filter)
            filter = cv2.bitwise_and(filter, filter, mask = try_frame)
            # cv2.imwrite('./outputsframe-18072022-02.jpg', frame)
            # cv2.imwrite('./outputs/frame-18072022-03.jpg', masked)
            # cv2.imwrite('./outputs/frame-18072022-04.jpg', filter)
            #try_frame = cv2.absdiff(pre_frame, try_frame)


            threshold = 3
            maxValue = 25
            return_value, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)
            # return_value, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_TOZERO)

            # add to the accumulated image
            th1 = cv2.bitwise_and(hsv_channels[2], th1)
            accum_image = cv2.add(accum_image, th1)

            # heat_map[np.all(img_contours == [0, 255, 0], 2)] += 3 # The 3 can be tweaked depending on how fast you want the colors to respond
            # heat_map[np.any(img_contours != [0, 255, 0], 2)] -= 1
            # heat_map[heat_map < 0] = 0
            # heat_map[heat_map > 255] = 255

            # color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            # color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_JET)
            color_image_video = cv2.applyColorMap(accum_image, cmapy.cmap('jet_r'))
            cv2.imshow("thres",color_image_video)


            frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.3, 0)
            # frame = cv2.addWeighted(frame, 0.7, color_image_video, 2, 0)

        # count object in the area
        person_go_left = len(left_region_ids)
        person_go_right = len(right_region_ids)
        # count object pass the area
        cv2.putText(frame,"left" +"   "+ str(person_go_left) + "  person", (5, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240, 33, 130), 1)
        cv2.putText(frame,"right" + "   "+ str(person_go_right)+ "  person", (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240, 33, 130), 1)

        # cv2.putText(frame,"floor path" + "   "+ str(floor_ids)+ "  person", (5, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240, 33, 130), 1)


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 165, 255), 1)

        result = np.asarray(frame)
        # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break





    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
