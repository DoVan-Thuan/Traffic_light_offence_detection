from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
import tensorflow as tf
import time
import predict

from PIL import Image
from yolo import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tensorflow.keras.models import load_model
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
from collections import Counter
from collections import deque
import math
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D

# from label_image import load_graph, read_tensor_from_image_file, load_labels, classify_light

warnings.filterwarnings('ignore')
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def predict_light(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    print(h,w)

    light = []
    a = int(h / 3)
    b = 2 * a
    light.append(img[0:a, 0:w])
    light.append(img[a:b, 0:w])
    light.append(img[b:h, 0:w])
    light_copy=light.copy()
    # for i in range(0, h, int(h / 3)):
    #     light.append(img[i:i + int(h / 3), 0:w])
    # cal_his(light)
    max_value = -9999
    index = 0
    for i in range(0, len(light)):
        # print(sum(map(sum,light[i])))#=0 het :(((dh kieu j)))
        if (np.mean(light[i]) > max_value):
            # print(max_value)
            max_value = np.mean(light[0])
            index = i
    # index = light_copy.index(sorted(light,key=lambda x:np.mean(x),reverse=True)[0])
    return index

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.75

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    # vgg = keras.applications.MobileNetV2(include_top=False)

    # model = Sequential()

    # # Finetune from VGG:
    # model.add(vgg)

    # # Add new layers
    # model.add(Dropout(rate = 0.5))
    # model.add(Conv2D(filters= 3, kernel_size=(3, 3), padding='same'))
    # model.add(GlobalAveragePooling2D())
    # model.add(BatchNormalization())
    # model.add(Activation('softmax'))
    

    # model.compile(loss='sparse_categorical_crossentropy',
    #           optimizer=keras.optimizers.Adam(lr=1e-4),
    #           metrics=['acc'])
    # model.load_weights("/content/drive/MyDrive/Light_classification/model_new_data.h5")


    file_path = '/content/drive/My Drive/deep-sort-yolov4-low-confidence-track-filtering/data/ngatu_2.mp4'
    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('/content/drive/My Drive/deep-sort-yolov4-low-confidence-track-filtering/res/ngatu_2.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    # counting_traffic = {'car' : 0,'bicycle':0,'motorbike':0,'bus':0,'truck':0}
    # already_counted = deque(maxlen=100)
    # class_counter = Counter()
    memory = {}
    # total_counter = 0
    # up_count = 0
    # down_count = 0
    light_pts = np.array([(1015, 201), (1032, 209), (1017, 254), (998, 248)])
    intersect_info = [] #initialize intersect
    pts = np.array([[790,716],[485,218],[837,235],[1277,603],[1280,720]],np.int32)
    #pts = np.array([[195, 327], [127, 889], [1881, 919], [1873, 333]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    #polygon = Polygon([(195, 327), (127, 889), (1881, 919), (1873, 333)])
    polygon = Polygon([(790,716),(485,218),(837,235),(1277,603),(1280,720)])
    mask = np.zeros((720, 1280, 3), np.uint8)
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255), 2)
    mask2 = cv2.fillPoly(mask.copy(), [pts], (255, 255, 255))
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

        ROI = cv2.bitwise_and(mask2,frame) #frame and with white region = region of frame (ROI)
        cv2.polylines(frame, [pts], True, (0, 0, 255),3)
        image = Image.fromarray(ROI[..., ::-1])  # bgr to rgb

        boxes, confidence, classes = yolo.detect_image(image)
        # print(len(boxes))
        # print(len(classes))
        # print(boxes)
        # print(classes)
        # Edit feature
        for i in range(len(classes)):
          if (classes[i] == "motorbike" or classes[i] == "bicycle"):
            boxes[i][1] = boxes[i][1] + boxes[i][3] - int(1.5*boxes[i][2])
            boxes[i][3] = int(1.5*boxes[i][2])
            # print(classes[i])
            # print(boxes[i])
            # boxes[i][1] = boxes[i][3] - int(3*(boxes[i][2]-boxes[i][0]))
        features = encoder(ROI, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        #line ve thang

        #line = [(0, (int(frame.shape[0] * 1/2))), (int(frame.shape[1]),int(frame.shape[0]*1/2))]  #shape[0] : high shape[1] : width

        #line = [(435,443),(1107,462)]

        #line = [(410,191),(1040,495)]
        #line = [(498, 647), (1601, 647)]
        # light = frame[200:252, 993:1031]
        # light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
        time_start = time.clock()
        light = four_point_transform(frame, light_pts)
        index_max = predict_light(light)
        # cv2.imwrite("light.jpg", light)
        # light = cv2.imread("light.jpg", 1)
        # image = cv2.resize(light , (64, 64))
        # image=np.reshape(image,newshape=(1,64,64,3))
        # result = model.predict_classes(image)
        # print(result)
        # index_max = np.argmax(result)
        time_elapsed = (time.clock() - time_start)
        print(time_elapsed)
        print(index_max)
        
        line = [(632, 399), (906, 428)]      
        cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
        if (index_max == 0):
          cv2.polylines(frame, [light_pts], True, (0, 0, 255),3)
        elif (index_max == 2):
          cv2.polylines(frame, [light_pts], True, (0, 255, 0),3)
        else:
          cv2.polylines(frame, [light_pts], True, (0, 255, 255),3)
        for det in detections:
            bbox = det.to_tlbr()
            if (det.cls == 'motorbike' or det.cls == 'bicycle'):
              bbox[1] = bbox[3] - int(1.5*(bbox[2]-bbox[0]))
            mid = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
            if show_detections and len(classes) > 0   :
                det_cls = det.cls
                score = "%.2f" % (det.confidence * 100) + "%"
                cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 2)
                tl = (bbox[0], bbox[1])
                bl = (bbox[0], bbox[3])
                if (intersect(tl, bl, line[0], line[1])):
                  cv2.line(frame,line[0],line[1],(0,0,255),2)
                else:
                  cv2.line(frame,line[0],line[1],(0,255,0),2)
                #center = (int((bbox[2]-bbox[0])/2),int((bbox[3]-bbox[1])/2))
                if (29*mid[0]-274*mid[1]+90998>=0 and index_max == 0):
                  cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                  # cv2.polylines(frame, [light_pts], True, (0, 0, 255),3)
                  # cv2.rectangle(frame, 200, 993, 252, 1031, (0, 0, 255), 2)
                # elif (29*mid[0]-274*mid[1]+90998<=0 and index_max == 2):
                #   cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                  # cv2.polylines(frame, [light_pts], True, (0, 255, 0),3)
                  # cv2.rectangle(frame, 200, 993, 252, 1031, (0, 255, 0), 2)
                else: 
                  cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                  # cv2.polylines(frame, [light_pts], True, (0, 255, 255),3)
                  # cv2.rectangle(frame, 200, 993, 252, 1031, (0, 255, 255), 2)
                # if(mid[1] >= int(frame.shape[0]/2)):
                #   cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                # else:
                #   cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
    
            bbox = track.to_tlbr()
            # if (det.cls == 'motorbike' or det.cls == 'bicycle'):
            #   bbox[1] = bbox[3] - int(2.5*(bbox[2]-bbox[0]))
            midpoint = (int((bbox[0]+bbox[2])/2) ,int((bbox[1]+bbox[3])/2))
            
            if track.track_id not in memory:
              memory[track.track_id]= deque(maxlen=2)
            memory[track.track_id].append(midpoint)
            previous_midpoint = memory[track.track_id][0] #co 2 gia tri trong queue la gia tri truoc va gia tri hien tai


        #     #center = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

        #     cv2.line(frame,midpoint,previous_midpoint,(0,255,0),2)
        #     #cv2.circle(frame, center,4, (0,0,255), 10)
        #     if (intersect(midpoint, previous_midpoint, line[0], line[1]) and track.track_id not in already_counted):
        #         class_counter[str(track.cls)] += 1
        #         already_counted.append(track.track_id) #luu cac ID da duoc dem, dem roi khong dem nua
        #         counting_traffic[str(track.cls)] += 1
        #         total_counter +=1
        #         print(counting_traffic)
        #         # draw red line
        #         cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
        #         angle = vector_angle(midpoint, previous_midpoint)
        #         #intersect_info.append([track.cls,midpoint,previous_midpoint, angle])

        #         if angle > 0:
        #             up_count += 1
        #         if angle < 0:
        #             down_count += 1

        #         #cv2.putText(frame, str(counting_traffic),(200,200),2,1e-3 * frame.shape[0], (0, 255, 0),2)
            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 2)
            if not show_detections:
                track_cls = track.cls
                cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-0 * frame.shape[0], (0, 255, 0),
                            2)
                cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 2)
        if len(memory) > 100:
            del memory[list(memory)[0]]
        # #cv2.putText(frame, "Total:  " + str(total_counter),(int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,1e-3 * frame.shape[0], (0, 255, 255), 2)
        # cv2.putText(frame, "Total: {} ({} up, {} down)".format(str(total_counter), str(up_count),
        #                 str(down_count)), (int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
        #                 1.5e-3 * frame.shape[0], (0, 255, 255), 2)

        # y = 0.2 * frame.shape[0]
        # for cls in class_counter:
        #     class_count = class_counter[cls]
        #     cv2.putText(frame, str(cls) + " " + str(class_count), (int(0.05 * frame.shape[1]), int(y)), 0,1e-3 * frame.shape[0], (0, 255, 255), 2)
        #     y += 0.05 * frame.shape[0]

      

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())