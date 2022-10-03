import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import LeakyReLU
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

POSE_LANDMARKS = {'nose': 0,
                  'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
                  'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
                  'left_ear': 7, 'right_ear': 8,
                  'mouth_left': 9, 'mouth_right': 10,
                  'left_shoulder': 11, 'right_shoulder': 12,
                  'left_elbow': 13, 'right_elbow': 14,
                  'left_wrist': 15, 'right_wrist': 16,
                  'left_pinky': 17, 'right_pinky': 18,
                  'left_index': 19, 'right_index': 20,
                  'left_thumb': 21, 'right_thumb': 22,
                  'left_hip': 23, 'right_hip': 24,
                  'left_knee': 25, 'right_knee': 26,
                  'left_ankle': 27, 'right_ankle': 28,
                  'left_heel': 29, 'right_heel': 30,
                  'left_foot_index': 31, 'right_foot_index': 32}

ACTION_ANGLES = {
    'PushUp': [70, 130],
    'Squat': [110, 170],
    'Reverse_Crunches': [80, 140],
    'Lateral_Lunges': [90, 150],
    'Jumbing_Jacks': [30, 90]}

ACTION_JOINTS = {
    'PushUp': [['right_shoulder', 'right_elbow', 'right_wrist'],
                ['left_shoulder', 'left_elbow', 'left_wrist']],
    'Squat': [['right_hip', 'right_knee', 'right_ankle'],
              ['left_hip', 'left_knee', 'left_ankle']],
    'Reverse_Crunches': [['right_shoulder', 'right_hip', 'right_knee'],
                      ['left_shoulder', 'left_hip', 'left_knee']],
    'Lateral_Lunges': [['right_hip', 'right_knee', 'right_ankle'],
              ['left_hip', 'left_knee', 'left_ankle']],
    'Jumbing_Jacks': [['right_elbow', 'right_shoulder', 'right_hip'],
                     ['left_elbow', 'left_shoulder', 'left_hip']]}

# Functions for model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results, mp_drawing, mp_pose): 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose

def get_joint_coordinate(results, joint_name, get_z=False):
    joint = results.pose_landmarks.landmark[POSE_LANDMARKS[joint_name]]
    if get_z:
        return [joint.x, joint.y, joint.z]
    else:
        return [joint.x, joint.y]

def get_three_joint_coordinates(results, joint_names, get_z=False):
    joint_coordinates = []
    for joint_name in joint_names:
        joint_coordinates.append(get_joint_coordinate(results, joint_name, get_z))
    return joint_coordinates

def get_three_joint_coordinates_two_sides(results, joint_names_left,
                            joint_names_right, get_z=False):
    joint_coordinates_left = get_three_joint_coordinates(results, joint_names_left, get_z)
    joint_coordinates_right = get_three_joint_coordinates(results, joint_names_right, get_z)
    return [joint_coordinates_left, joint_coordinates_right]

def calculate_angle(joint_coordinates):
    pt1 = np.array(joint_coordinates[0]) # First
    pt2 = np.array(joint_coordinates[1]) # Mid
    pt3 = np.array(joint_coordinates[2]) # End
    
    # x: 0, y: 1
    angle_radians = np.arctan2(pt3[1] - pt2[1], pt3[0] - pt2[0]) - \
                    np.arctan2(pt1[1] - pt2[1], pt1[0]- pt2[0])
    angle_degree = np.abs(angle_radians*180.0/np.pi)
    
    if angle_degree > 180.0: # Convert to 0-180 degree
        angle_degree = 360 - angle_degree
        
    return angle_degree

def counting(angle, stage, counter, max_angle, min_angle):
    if angle[0] > max_angle:
        stage[0] = "down"
    if angle[0] < min_angle and stage[0] == 'down':
        stage[0] = "up"
        counter[0] += 1
    if angle[1] > max_angle:
        stage[1] = "down"
    if angle[1] < min_angle and stage[1] == 'down':
        stage[1] = "up"
        counter[1] += 1
    return stage, counter

def calculate_angle_two_side(joint_coordinates_left, joint_coordinates_right):
    angle_left = calculate_angle(joint_coordinates_left)
    angle_right = calculate_angle(joint_coordinates_right)
    
    return angle_left, angle_right

class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(VideoThread, self).__init__(parent)
        self.get_file_dir = self.parent().file_dir #Get the file_dir from UI_MainWindow
        

        self.sequence_length = self.parent().sequence_length
        self.capture = self.parent().capture
        self.mp_pose = self.parent().mp_pose
        self.mp_drawing = self.parent().mp_drawing
        self.counter = self.parent().counter
        self.stage = self.parent().stage
        self.sequence = self.parent().sequence
        self.pre_action = self.parent().pre_action
        self.actions = self.parent().actions
        self.model = self.parent().model
    def run(self):
        self.capture = cv2.VideoCapture(self.get_file_dir)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 631)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 621)

        idx = 0
        action_old = ''
        counter_old = 0
        action_new = ''
        counter_new = 0
        while True:
            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while self.capture.isOpened():
                    ret, image = self.capture.read()
                    if not ret:
                        break

                    idx += 1
                    if idx >= 40 & idx % 2:
                        
                        image, results = mediapipe_detection(image, pose)
                        draw_styled_landmarks(image, results, self.mp_drawing, self.mp_pose)
                        keypoints = extract_keypoints(results)
                        self.sequence.append(keypoints)
                        self.sequence = self.sequence[-self.sequence_length:]
                        

                        if len(self.sequence) == self.sequence_length:
                            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                            # print('res', res)
                            # print('np.argmax(res)', np.argmax(res))
                            if np.argmax(res) != 4:
                                try:

                                    action = self.actions[np.argmax(res)]
                                    if action != self.pre_action:
                                        self.counter = [0, 0]
                                        self.stage = [None, None]

                                    action_angle = ACTION_ANGLES[action]
                                    joint_names_left, joint_names_right = ACTION_JOINTS[action]
                                    joint_coordinates_left, joint_coordinates_right = get_three_joint_coordinates_two_sides(results, joint_names_left,
                                                    joint_names_right, get_z=False)
                                    angle = calculate_angle_two_side(joint_coordinates_left, joint_coordinates_right)
                                    self.stage, self.counter = counting(angle, self.stage, self.counter,
                                                            min_angle=action_angle[0], 
                                                            max_angle=action_angle[1])
                                    
                                    print(action, np.max(self.counter))
                                    # self.update_ui_counter(action, np.max(self.counter))
                                    self.pre_action = action

                                    action_new = action.copy()
                                    counter_new = np.max(self.counter).copy()

                                    # self.update_ui_counter(action1, counter1)
                                    # print('im here too')
                                except:
                                    print('NOTHING DONE')
                                    pass
                    if ret:
                        self.change_pixmap_signal.emit(image)
                    
                    if counter_new != counter_old:
                        print('old:', action_new, counter_old)
                        print('new:', action_new, counter_new)
                        self.update_ui_counter(action_new, counter_new)
                        counter_old = counter_new
                    
                    # if change the activity
                    if action_new != action_old:
                        action_old = action_new
                        counter_old = 0
                        counter_new = 0

                    self.msleep(1000//30) # You can change 30 with 60 if you need 60 fps.

    def update_act_tracker(self, action, counter):
        modelLabel_to_mineLabel = {
            'Lateral_Lunges': 'LL',
            'Reverse_Crunches': 'RC',
            'PushUp': 'PU',
            'Squat': 'SQ',
        }        
        # print(action)
        mineLabel = modelLabel_to_mineLabel[action]
        if counter != 0:
            self.parent().act_tracker[mineLabel] += 1

        print('label returned:', mineLabel)
        return mineLabel

    def update_ui_counter(self, action, counter):
        print(action)
        if action == 'Lateral_Lunges':
            # Change style sheet
            self.parent().LLc.setStyleSheet(
                                            '''
                                            color: red; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                                
                                            )
            self.parent().RCc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().PUc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().PUc.setAlignment(QtCore.Qt.AlignCenter)                                
            self.parent().SQc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )

            mineLabel = self.update_act_tracker(action, counter)
            self.parent().LLc.setText(str(self.parent().act_tracker[mineLabel]))
            self.calo_update(mineLabel)
        elif action == 'Reverse_Crunches':
            # Change style sheet
            self.parent().LLc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().RCc.setStyleSheet(
                                            '''
                                            color: red; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                                
                                            )
            self.parent().PUc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().SQc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )

            mineLabel = self.update_act_tracker(action, counter)
            self.parent().RCc.setText(str(self.parent().act_tracker[mineLabel]))
            self.calo_update(mineLabel)
        elif action == 'PushUp':
            mineLabel = self.update_act_tracker(action, counter)
            self.parent().PUc.setText(str(self.parent().act_tracker[mineLabel]))
            self.calo_update(mineLabel)
            # Change style sheet
            self.parent().LLc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                             
                                            )
            self.parent().RCc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                           
                                            )
            self.parent().PUc.setStyleSheet('''
                                            color: red; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                   
                                            )
            self.parent().SQc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                           
                                            )
        elif action == 'Squat':
            # Change style sheet
            self.parent().LLc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().RCc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().PUc.setStyleSheet(
                                            '''
                                            color: #123F6D; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                            
                                            )
            self.parent().SQc.setStyleSheet(
                                            '''
                                            color: red; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                                
                                            )

            mineLabel = self.update_act_tracker(action, counter)
            self.parent().SQc.setText(str(self.parent().act_tracker[mineLabel]))
            self.calo_update(mineLabel)
    
    def calo_update(self, type):
        # call cal_calo function        
        print('here')
        calo = self.parent().calo_per_reps[type]*self.parent().weight
        self.parent().calo_in_total += calo
        self.parent().calo_counter.setStyleSheet(
                                            '''                                        
                                            font: bold 36pt;
                                            color: #bf1e2e; 
                                            qproperty-alignment: AlignCenter;       
                                            '''                                                
                                            )
        self.parent().calo_counter.setText(str(round(self.parent().calo_in_total, 2)))

    def stop(self):
        self.parent().image = None
        self.parent().thread={}

        self.parent().counter = [0, 0]
        self.parent().stage = [None, None]
        self.parent().sequence = []
        self.parent().pre_action = None

        self.parent().capture.release()

        print('Stopping thread...')
        self.continue_run = False
        self.terminate()

class Ui_Form(object):
    def __init__(self):
        super(Ui_Form, self).__init__()
        self.image = None
        self.file_dir = 'testVid.mp4'
        self.capture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.weight = 0
        self.thread={}
        self.calo_in_total = 0

        # act tracker dict
        # self.act_tracker = {
        #     "timer": [],    # store time of each activity eg. [[start_time_act1, end_time_act1], [start_time_act2, end_time_act2]]
        #     "act": [],      # store name of each activity eg. [act1, act2]
        # }
        self.act_tracker = {
            "LL": 0,   
            "PU": 0,      
            "SQ": 0,
            "RC": 0,
            }

        # Calo per reps for 1kg
        self.calo_per_reps = {
            "LL": 24/6350,     # 2 calo per 15 minutes
            "PU": 47/6350,      # 94 calo per 127 reps
            "SQ": 26/6350,
            "RC": 21/6350,
        }

        # self.calo_per_reps = {
        #     "LL": 24,     # 2 calo per 15 minutes
        #     "PU": 47,      # 94 calo per 127 reps
        #     "SQ": 26,
        #     "RC": 21,
        # }

        # Setup for model
        self.mp_pose = mp.solutions.pose # pose model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities

        self.sequence_length = 30
        self.actions = np.array(['Lateral_Lunges', 'PushUp', 'Reverse_Crunches', 'Squat', 'no_action'])

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, input_shape=(self.sequence_length,132)))
        self.model.add(LeakyReLU(alpha=.01))
        self.model.add(LSTM(32, return_sequences=True))
        self.model.add(LeakyReLU(alpha=.01))
        self.model.add(LSTM(16, return_sequences=False))
        self.model.add(LeakyReLU(alpha=.01))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        self.model.load_weights('model/actions_LSTM_5_30_2_seq_v2_prebest.h5')

        self.counter = [0, 0]
        self.stage = [None, None]
        self.sequence = []
        self.pre_action = None

    def setupUi(self, Form):

        Form.setObjectName("Form")
        Form.setEnabled(True)
        Form.resize(1141, 730)
        font = QtGui.QFont()
        font.setItalic(False)
        Form.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/eye-recognition.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setAutoFillBackground(False)

        # dashboard_title
        self.dashboard_title = QtWidgets.QLabel(Form)
        self.dashboard_title.setGeometry(QtCore.QRect(20, 20, 581, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dashboard_title.sizePolicy().hasHeightForWidth())
        self.dashboard_title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.dashboard_title.setFont(font)
        self.dashboard_title.setTextFormat(QtCore.Qt.AutoText)
        self.dashboard_title.setScaledContents(False)
        self.dashboard_title.setObjectName("dashboard_title")

        # creators
        self.creators = QtWidgets.QLabel(Form)
        self.creators.setGeometry(QtCore.QRect(45, 52, 581, 21))
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(10)
        font.setItalic(True)
        self.creators.setFont(font)
        self.creators.setObjectName("creators")

        # header_bgr
        self.header_bgr = QtWidgets.QLabel(Form)
        self.header_bgr.setGeometry(QtCore.QRect(0, -10, 1141, 131))
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setItalic(False)
        self.header_bgr.setFont(font)
        self.header_bgr.setText("")
        self.header_bgr.setPixmap(QtGui.QPixmap("images/header_bgr.png"))
        self.header_bgr.setScaledContents(True)
        self.header_bgr.setObjectName("header_bgr")

        # body_bgr
        self.body_bgr = QtWidgets.QLabel(Form)
        self.body_bgr.setGeometry(QtCore.QRect(-10, 90, 1321, 871))
        self.body_bgr.setText("")
        self.body_bgr.setPixmap(QtGui.QPixmap("images/body_bgr.png"))
        self.body_bgr.setScaledContents(True)
        self.body_bgr.setObjectName("body_bgr")

        # cam_brg
        self.cam_brg = QtWidgets.QLabel(Form)
        self.cam_brg.setGeometry(QtCore.QRect(20, 90, 631, 621))
        self.cam_brg.setText("")
        self.cam_brg.setPixmap(QtGui.QPixmap("images/block_bgr.png"))
        self.cam_brg.setScaledContents(True)
        self.cam_brg.setObjectName("cam_brg")

        # act_brg
        self.act_brg = QtWidgets.QLabel(Form)
        self.act_brg.setGeometry(QtCore.QRect(670, 90, 451, 431))
        self.act_brg.setText("")
        self.act_brg.setPixmap(QtGui.QPixmap("images/block_bgr.png"))
        self.act_brg.setScaledContents(True)
        self.act_brg.setObjectName("act_brg")

        # calo_brg
        self.calo_brg = QtWidgets.QLabel(Form)
        self.calo_brg.setGeometry(QtCore.QRect(670, 540, 451, 171))
        self.calo_brg.setText("")
        self.calo_brg.setPixmap(QtGui.QPixmap("images/block_bgr.png"))
        self.calo_brg.setScaledContents(True)
        self.calo_brg.setObjectName("calo_brg")

        # act_counter
        self.act_counter = QtWidgets.QLabel(Form)
        self.act_counter.setGeometry(QtCore.QRect(690, 110, 171, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.act_counter.sizePolicy().hasHeightForWidth())
        self.act_counter.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.act_counter.setFont(font)
        self.act_counter.setTextFormat(QtCore.Qt.AutoText)
        self.act_counter.setScaledContents(False)
        self.act_counter.setObjectName("act_counter")

        # calo_burnt
        self.calo_burnt = QtWidgets.QLabel(Form)
        self.calo_burnt.setGeometry(QtCore.QRect(690, 560, 141, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calo_burnt.sizePolicy().hasHeightForWidth())
        self.calo_burnt.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.calo_burnt.setFont(font)
        self.calo_burnt.setTextFormat(QtCore.Qt.AutoText)
        self.calo_burnt.setScaledContents(False)
        self.calo_burnt.setObjectName("calo_burnt")

        # app_label
        self.app_label = QtWidgets.QLabel(Form)
        self.app_label.setGeometry(QtCore.QRect(20, 52, 20, 20))
        self.app_label.setText("")
        self.app_label.setPixmap(QtGui.QPixmap("images/eye-recognition.png"))
        self.app_label.setScaledContents(True)
        self.app_label.setObjectName("app_label")

        # start
        self.start = QtWidgets.QPushButton(Form)
        self.start.setGeometry(QtCore.QRect(220, 620, 71, 71))
        self.start.setMouseTracking(False)
        self.start.setAutoFillBackground(False)
        self.start.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.start.setIcon(icon1)
        self.start.setIconSize(QtCore.QSize(60, 60))
        self.start.setCheckable(False)
        self.start.setChecked(False)
        self.start.setAutoDefault(False)
        self.start.setDefault(False)
        self.start.setFlat(True)
        self.start.setObjectName("start")

        self.start.clicked.connect(self.start_webcam)

        # Logo
        self.VinGroupLogo = QtWidgets.QLabel(Form)
        self.VinGroupLogo.setGeometry(QtCore.QRect(690, 20, 80, 51))
        self.VinGroupLogo.setText("")
        self.VinGroupLogo.setPixmap(QtGui.QPixmap("images/1280px-Vingroup_logo.svg.png"))
        self.VinGroupLogo.setScaledContents(True)
        self.VinGroupLogo.setObjectName("VinGroupLogo")
        self.VinBigDataLogo = QtWidgets.QLabel(Form)
        self.VinBigDataLogo.setGeometry(QtCore.QRect(820, 20, 115, 51))
        self.VinBigDataLogo.setText("")
        self.VinBigDataLogo.setPixmap(QtGui.QPixmap("images/logo-VinBigData-2020-01.png"))
        self.VinBigDataLogo.setScaledContents(True)
        self.VinBigDataLogo.setObjectName("VinBigDataLogo")
        self.SOICTlogo = QtWidgets.QLabel(Form)
        self.SOICTlogo.setGeometry(QtCore.QRect(990, 20, 115, 51))
        self.SOICTlogo.setText("")
        self.SOICTlogo.setPixmap(QtGui.QPixmap("images/logo-soict-hust-1.png"))
        self.SOICTlogo.setScaledContents(True)
        self.SOICTlogo.setObjectName("SOICTlogo")

        # act_logo_brg
        self.LLb = QtWidgets.QLabel(Form)
        self.LLb.setGeometry(QtCore.QRect(690, 150, 411, 81))
        self.LLb.setText("")
        self.LLb.setPixmap(QtGui.QPixmap("images/counter_bgr.png"))
        self.LLb.setScaledContents(True)
        self.LLb.setObjectName("LLb")
        self.SQb = QtWidgets.QLabel(Form)
        self.SQb.setGeometry(QtCore.QRect(690, 240, 411, 81))
        self.SQb.setText("")
        self.SQb.setPixmap(QtGui.QPixmap("images/counter_bgr.png"))
        self.SQb.setScaledContents(True)
        self.SQb.setObjectName("SQb")
        self.PUb = QtWidgets.QLabel(Form)
        self.PUb.setGeometry(QtCore.QRect(690, 330, 411, 81))
        self.PUb.setText("")
        self.PUb.setPixmap(QtGui.QPixmap("images/counter_bgr.png"))
        self.PUb.setScaledContents(True)
        self.PUb.setObjectName("PUb")
        self.RCb = QtWidgets.QLabel(Form)
        self.RCb.setGeometry(QtCore.QRect(690, 420, 411, 81))
        self.RCb.setText("")
        self.RCb.setPixmap(QtGui.QPixmap("images/counter_bgr.png"))
        self.RCb.setScaledContents(True)
        self.RCb.setObjectName("RCb")

        # calo_counter
        self.calo_counter = QtWidgets.QLabel(Form)
        self.calo_counter.setGeometry(QtCore.QRect(690, 600, 411, 71))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calo_counter.sizePolicy().hasHeightForWidth())
        self.calo_counter.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.calo_counter.setFont(font)
        self.calo_counter.setTextFormat(QtCore.Qt.AutoText)
        self.calo_counter.setScaledContents(False)
        self.calo_counter.setObjectName("calo_counter")

        # cam
        self.cam = QtWidgets.QLabel(Form)
        self.cam.setGeometry(QtCore.QRect(46, 112, 581, 491))
        self.cam.setFrameShape(QtWidgets.QFrame.Box)
        self.cam.setText("")
        self.cam.setObjectName("cam")

        # stop
        self.stop = QtWidgets.QPushButton(Form)
        self.stop.setGeometry(QtCore.QRect(380, 620, 71, 71))
        self.stop.setMouseTracking(False)
        self.stop.setAutoFillBackground(False)
        self.stop.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/stop-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stop.setIcon(icon2)
        self.stop.setIconSize(QtCore.QSize(60, 60))
        self.stop.setCheckable(False)
        self.stop.setChecked(False)
        self.stop.setAutoDefault(False)
        self.stop.setDefault(False)
        self.stop.setFlat(True)
        self.stop.setObjectName("stop")

        self.stop.clicked.connect(self.stop_webcam)

        # activity logos
        self.LL = QtWidgets.QLabel(Form)
        self.LL.setGeometry(QtCore.QRect(700, 165, 81, 51))
        self.LL.setText("")
        self.LL.setPixmap(QtGui.QPixmap("images/lateralLunges.png"))
        self.LL.setScaledContents(True)
        self.LL.setObjectName("LL")
        self.SQ = QtWidgets.QLabel(Form)
        self.SQ.setGeometry(QtCore.QRect(700, 255, 81, 51))
        self.SQ.setText("")
        self.SQ.setPixmap(QtGui.QPixmap("images/squats.png"))
        self.SQ.setScaledContents(True)
        self.SQ.setObjectName("SQ")
        self.PU = QtWidgets.QLabel(Form)
        self.PU.setGeometry(QtCore.QRect(700, 345, 81, 51))
        self.PU.setText("")
        self.PU.setPixmap(QtGui.QPixmap("images/pushups-man.png"))
        self.PU.setScaledContents(True)
        self.PU.setObjectName("PU")
        self.RC = QtWidgets.QLabel(Form)
        self.RC.setGeometry(QtCore.QRect(700, 435, 81, 51))
        self.RC.setText("")
        self.RC.setPixmap(QtGui.QPixmap("images/ReverseCrunches.png"))
        self.RC.setScaledContents(True)
        self.RC.setObjectName("RC")

        # Activity counters
        self.LLc = QtWidgets.QLabel(Form)
        self.LLc.setGeometry(QtCore.QRect(810, 170, 221, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LLc.sizePolicy().hasHeightForWidth())
        self.LLc.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.LLc.setFont(font)
        self.LLc.setTextFormat(QtCore.Qt.AutoText)
        self.LLc.setScaledContents(False)
        self.LLc.setObjectName("LLc")
        self.SQc = QtWidgets.QLabel(Form)
        self.SQc.setGeometry(QtCore.QRect(810, 260, 221, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SQc.sizePolicy().hasHeightForWidth())
        self.SQc.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.SQc.setFont(font)
        self.SQc.setTextFormat(QtCore.Qt.AutoText)
        self.SQc.setScaledContents(False)
        self.SQc.setObjectName("SQc")
        self.PUc = QtWidgets.QLabel(Form)
        self.PUc.setGeometry(QtCore.QRect(810, 350, 221, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PUc.sizePolicy().hasHeightForWidth())
        self.PUc.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.PUc.setFont(font)
        self.PUc.setTextFormat(QtCore.Qt.AutoText)
        self.PUc.setScaledContents(False)
        self.PUc.setObjectName("PUc")
        self.RCc = QtWidgets.QLabel(Form)
        self.RCc.setGeometry(QtCore.QRect(810, 440, 221, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RCc.sizePolicy().hasHeightForWidth())
        self.RCc.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.RCc.setFont(font)
        self.RCc.setTextFormat(QtCore.Qt.AutoText)
        self.RCc.setScaledContents(False)
        self.RCc.setObjectName("RCc")
        
        self.body_bgr.raise_()
        self.header_bgr.raise_()
        self.dashboard_title.raise_()
        self.creators.raise_()
        self.cam_brg.raise_()
        self.act_brg.raise_()
        self.calo_brg.raise_()
        self.act_counter.raise_()
        self.calo_burnt.raise_()
        self.app_label.raise_()
        self.start.raise_()
        self.VinGroupLogo.raise_()
        self.VinBigDataLogo.raise_()
        self.SOICTlogo.raise_()
        self.LLb.raise_()
        self.SQb.raise_()
        self.PUb.raise_()
        self.RCb.raise_()
        self.calo_counter.raise_()
        self.cam.raise_()
        self.stop.raise_()
        self.LL.raise_()
        self.SQ.raise_()
        self.PU.raise_()
        self.RC.raise_()
        self.LLc.raise_()
        self.SQc.raise_()
        self.PUc.raise_()
        self.RCc.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "AI Fitness Dashboard"))
        self.dashboard_title.setText(_translate("Form", "<html><head/><body><p align=\"justify\"><span style=\" color:#ffffff;\">AI Fitness Dashboard</span></p></body></html>"))
        self.creators.setText(_translate("Form", "<html><head/><body><p><span style=\" color:#e1e1e1;\">AI Powered </span><span style=\" font-style:normal; color:#e1e1e1;\">|</span><span style=\" color:#e1e1e1;\">Hung Tran, Tung Khong, Hieu Doan, Nghia Pham, and Phu Hoang</span></p></body></html>"))
        self.act_counter.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color:#101038;\">Activity Counter</span></p></body></html>"))
        self.calo_burnt.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt; color:#101038;\">Calories Burnt</span></p></body></html>"))
        self.calo_counter.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt; color:#bf1e2e;\">0</span></p></body></html>"))
        self.LLc.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; color:#123F6D;\">0</span></p></body></html>"))
        self.SQc.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; color:#123F6D;\">0</span></p></body></html>"))
        self.PUc.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; color:#123F6D;\">0</span></p></body></html>"))
        self.RCc.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; color:#123F6D;\">0</span></p></body></html>"))

    ####### Funtions to open cam #######
    def start_webcam(self):
        self.weight = float(self.get_weight())
        # if self.weight == 0 or self.weight == 1, then do nothing
        if self.weight <= 0:
            return
        
        self.act_tracker = {
            "LL": 0,   
            "PU": 0,      
            "SQ": 0,
            "RC": 0,
            }
        
        self.calo_in_total = 0

        try:
          self.thread[1].terminate()
        except:
          print("Nothing to terminate!")

        self.timer = QtCore.QTimer(self)
        self.thread[1] = VideoThread(self)
        self.thread[1].start()
        self.thread[1].change_pixmap_signal.connect(self.update_frame)        
        self.timer.start(5)
    
    def update_frame(self, cv_img):
        self.displayImage(cv_img, 1)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def stop_webcam(self):
        self.timer.stop()
        # Create window with warnig message
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText("Are you sure you want to END your workout?")
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        msg.buttonClicked.connect(self.popup_button)
        x = msg.exec_()

    def popup_button(self, i):
        if i.text() == "OK":
            self.thread[1].stop()
        else:
            self.timer.start(5)

    def displayImage(self, img, window=1):
        qformat = QtGui.QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if (img.shape[2]) == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888

        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR > RGB
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.cam.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.cam.setScaledContents(True)

    def get_weight(self):
        weight = QtWidgets.QInputDialog.getInt(self, 'Insert value', 'Insert your Weight (kg):',
            min=0, max=100)
        return str(weight[0])
    
    ####### Funtions to cal_calo #######
    def cal_calo(self, count, type):
        # get cal based on type from calo_per_reps
        cal = self.calo_per_reps[type]
        # get weight from user
        weight = self.weight
        # cal calo
        calo = count * cal * weight
        return calo
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
