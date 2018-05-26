# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
# from imutils.video import FileVideoStream
# from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
 

def mouth_aspect_ratio_old(mouth):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates 
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    mar = (A + B + C) / (2.0 * D)
    return mar 
 

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates 
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    mar = (B)* 10  / (2.0 * D)
    return mar 

 

class CaptureReader():
    def __init__(self, capture):
        self.frame_counts = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.time_total =  capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.total_time = self.frame_counts / self.fps

    def time2frameIdx(self, timeSecond):
        return int(self.frame_counts * timeSecond/(self.time_total))

    def frameIdx2time(self, framePos):
        return float(self.total_time * framePos/self.frame_counts)



def fetchTimePairs(path):
    with open(path, 'r') as f:
        line = f.readline().strip()
        timestamps = line.split(' ')
        if len(timestamps) % 2 != 0:
            raise ValueError('odd timestampes, error!')

        timePairs = []
        for idx, i in enumerate(timestamps):
            if idx % 2 ==0:
                timePairs.append( (float(timestamps[idx]), float(timestamps[idx+1])) )
        return timePairs



 




'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
ap.add_argument("--visualize", type=bool, default=True,
    help="visualize the detection")
args = vars(ap.parse_args())
'''

args = {}
args['video'] = '1526964977155.mp4'
args['shape_predictor'] = 'shape_predictor_68_face_landmarks.dat'
args['visualize'] = True

 

 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.80

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

COUNTER_Mouth = 0
TOTAL_Mouth = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")

# loop over frames from the video stream
visualize = args['visualize'] 
cap = cv2.VideoCapture(args['video'])
timestamp_path = args['video'].split('.')[0] + '.txt'
capReader = CaptureReader(cap)
FPS =cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FPS, int(FPS) );
interval = 1# int(1000/FPS)//2


frame_counts = capReader.frame_counts
tm_total =  capReader.time_total

frame_pos = 0
start = time.time()

line_list = []
threshline_list = []


def getTimeVerticalLine(timeSecond, setting):

    height, width, capReader = setting.height, setting.width, setting.capReader
    correspond_framePos = capReader.time2frameIdx(timeSecond)
    x_offset = (correspond_framePos / capReader.frame_counts) * width  

    verticalLine_list = []
    for i in np.linspace(0, height, 100):
        verticalLine_list.append( (x_offset, i) )
    vertical_data = np.array([verticalLine_list]).astype(np.int32)
    return vertical_data


def drawLines(frame, pairs, setting):
    RGB = [(0,0,255), (0,255,0), (255,0,0)]
    for idx, pair in enumerate(pairs):
        vert_data_start = getTimeVerticalLine(timeSecond=pair[0], setting=setting)
        cv2.polylines(frame, vert_data_start, False, RGB[idx % 3], 2)
        vert_data_end = getTimeVerticalLine(timeSecond=pair[1], setting=setting)
        cv2.polylines(frame, vert_data_end, False, RGB[idx % 3], 2)
    return frame

mar_list = []
frame_list = []
time_list = []
 

class Draw_Setting:
    def __init__(self, height, width, capReader):
        self.height = height
        self.width = width
        self.capReader = capReader


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
    #    frame = cv2.resize(frame, (frame.shape[0], frame.shape[1]), interpolation=cv2.INTER_CUBIC)
    #    print(frame.shape)
        height, width  = frame.shape[:2]
        setting = Draw_Setting(height, width, capReader)
        frame_pos += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            outerMouth = shape[48:60]
            innerMouth = shape[60:68]
            mar = mouth_aspect_ratio(innerMouth)
            cv2.drawContours(frame, [innerMouth], -1, (0, 255, 0), 1)

            tm = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            #print(frame_pos, frame_counts)
            y_intervals = height/4/1.2
            y_baseline = int(height/2)
            x_offset = (frame_pos / frame_counts) * width  

            for i in np.linspace(0, width, 100):
                threshline_list.append( (i, int(y_baseline-MOUTH_AR_THRESH*y_intervals)) )
            thresh_data = np.array([threshline_list]).astype(np.int32)
            cv2.polylines(frame, thresh_data, False, (0, 255, 255), 2)


            pairs = fetchTimePairs(timestamp_path)
            frame = drawLines(frame, pairs, setting)
            line_list.append(( x_offset,  int(y_baseline-mar*y_intervals))  )
            data = np.array([line_list]).astype(np.int32)
            cv2.polylines(frame, data, False, (0, 255, 0), 2)
            
            mar_list.append(mar)
            frame_list.append(frame_pos)
            time_list.append(capReader.frameIdx2time(frame_pos))
            if mar > MOUTH_AR_THRESH: 
                COUNTER_Mouth += 1
            else:
                if COUNTER_Mouth >= EYE_AR_CONSEC_FRAMES:
                    TOTAL_Mouth += 1
                    COUNTER_Mouth = 0


        if visualize:
            for idx, (x, y) in enumerate(shape):
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.putText(frame, 'Time: {:.3}'.format(tm), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Mouth Opens: {}".format(TOTAL_Mouth), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", cv2.resize(frame,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_LINEAR))
            #cv2.imshow('Frame',  frame)
            key = cv2.waitKey(interval) & 0xFF
            if key == ord("q"):
                break
    else:
        break
                
# do a bit of cleanup

df_info = pd.DataFrame(data=np.vstack([mar_list, frame_list, time_list]).T, columns=['MAR', 'frame_ID', 'time'])
#print(df_info)

print('------------------------')
print(mar_list)
print('-----------------------')
print(len(mar_list))



def getTimeMask(timeSeries, timeRange):
    start, end = timeRange
    return (timeSeries < end)  &  (timeSeries > start)
 



#df[df['MAR'] > MOUTH_AR_THRESH].groupby(by='rangeID').count()


pairs = fetchTimePairs(timestamp_path)
mask = np.zeros(df_info.time.shape).astype(bool) 

print('-----------------------------------------------------')
df_info_list = []
for idx, pair in enumerate(pairs):
    mask_single = getTimeMask(df_info.time, pair)
    df_info_selected = df_info.loc[mask_single]
    df_info_selected.loc[:, 'rangeID'] = idx
    df_info_selected.loc[:, 'GreaterThanThreshcount'] = (df_info_selected.MAR > MOUTH_AR_THRESH).sum()
    df_info_list.append( df_info_selected )

df = pd.concat(df_info_list, axis=0)
 


end = time.time()
print(end - start)
cap.release()
cv2.destroyAllWindows()
print('---------------------finished---------------------') 
print(TOTAL_Mouth)