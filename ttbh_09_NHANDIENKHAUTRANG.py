from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import os
from gtts import gTTS
import speech_recognition as sr
import numpy as np 
from playsound import playsound
from tkinter import *
"""
Khai báo hằng 
"""
path = 'Images'
language = 'vi'
english = 'en'
# Khoi tao cac module detect mat va facial landmark
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def speak(text,lan=language):
    print("Bot: {}".format(text))
    tts = gTTS(text=text, lang=lan, slow=False)
    tts.save("sound.mp3")
    playsound("sound.mp3", True)
    os.remove("sound.mp3")
def handleSourceVideo():
    vs = VideoStream(src=0).start()
    time.sleep(3.0)

    while True: 
 
	# Doc tu camera
        frame = vs.read()

        # Chuyen ve gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Detect cac mat trong anh
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,		minNeighbors=5, minSize=(100, 100),		flags=cv2.CASCADE_SCALE_IMAGE)
        # là nguồn/bức ảnh xám,độ scale sau mỗi lần quét, tính theo 0.01 = 1%. Nếu như để scaleFactor = 1 thì tấm ảnh sẽ giữ nguyên
        # Duyet qua cac mat
        
        for (x, y, w, h) in faces:

            # Tao mot hinh chu nhat quanh khuon mat
            rect = dlib.rectangle(int(x), int(y), int(x + w),
                int(y + h))
            
            # Nhan dien cac diem landmark
            landmark = landmark_detect(gray, rect)
            landmark = face_utils.shape_to_np(landmark)

            # Capture vung mieng
            (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            mouth = landmark[mStart:mEnd]

            # Lay hinh chu nhat bao vung mieng
            boundRect = cv2.boundingRect(mouth)
            cv2.rectangle(frame,
                        (int(boundRect[0]), int(boundRect[1])),
                        (int(boundRect[0] + boundRect[2]),  int(boundRect[1] + boundRect[3])), (255,0,0), 2)

            # Tinh toan saturation trung binh
            hsv = cv2.cvtColor(frame[int(boundRect[1]):int(boundRect[1] + boundRect[3]),int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_RGB2HSV)
            sum_saturation = np.sum(hsv[:, :, 1]) # Sum the brightness values
            area = int(boundRect[2])*int(boundRect[3])
            avg_saturation = sum_saturation / area

            # Kiem tra va canh bao voi nguong
            if avg_saturation>85:
                speak('Mang khẩu trang vào bạn ơi',language)
                speak('Please take your mask',english)
            else:
                speak('Tốt lắm',language)
                speak('Well done',english)         

	# Hien thi len man hinh
        cv2.imshow("Camera", frame)

        # Bam Esc de thoat
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

def handleSourceImage(img_name):
    frame = cv2.imread(os.path.join(path,img_name))
    while True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cac mat trong anh
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,		minNeighbors=5, minSize=(100, 100),		flags=cv2.CASCADE_SCALE_IMAGE)

        # Duyet qua cac mat
        for (x, y, w, h) in faces:

            # Tao mot hinh chu nhat quanh khuon mat
            rect = dlib.rectangle(int(x), int(y), int(x + w),
                int(y + h))

            # Nhan dien cac diem landmark
            landmark = landmark_detect(gray, rect)
            landmark = face_utils.shape_to_np(landmark)

            # Capture vung mieng
            (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            mouth = landmark[mStart:mEnd]

            # Lay hinh chu nhat bao vung mieng
            boundRect = cv2.boundingRect(mouth)
            cv2.rectangle(frame,
                        (int(boundRect[0]), int(boundRect[1])),
                        (int(boundRect[0] + boundRect[2]),  int(boundRect[1] + boundRect[3])), (0,0,255), 2)

            # Tinh toan saturation trung binh
            hsv = cv2.cvtColor(frame[int(boundRect[1]):int(boundRect[1] + boundRect[3]),int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_RGB2HSV)
            sum_saturation = np.sum(hsv[:, :, 1])
            area = int(boundRect[2])*int(boundRect[3])
            avg_saturation = sum_saturation / area

            # Check va canh bao voi threshold
            if avg_saturation>200:
                cv2.putText(frame, "NO MASK- NO HEALTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                
        # Hien thi len man hinh
        cv2.imshow("Camera", frame) 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
    
def ttbh():
    while (True):
        speak('Xin chào')
        speak("Bạn muốn kiểm tra hình ảnh hay video")
        print("""
              1. Video 
              2. Hình ảnh 
              3. Thoát
              """)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            print("Chọn cách kiểm tra, ví dụ video... ")
            # read the audio data from the default microphone
            audio_data = r.record(source, duration=5)
            # convert speech to text
            try:
                query = r.recognize_google(audio_data,language="vi")
            except:
                continue 
            if query == 'video' :
                speak("Bạn đã chọn {}".format(query))
                handleSourceVideo()
            if query == 'hình ảnh' :
                speak("Bạn đã chọn {}".format(query)) 
                images=[]
                className=[]
                myList=os.listdir(path)#danhsáchtệptin
                #Lấy image và tên
                for cl in myList:
                    image= cv2.imread(f'{path}/{cl}',0)
                    images.append(image)
                    className.append(os.path.basename(cl))
                for i in className:
                    print("{} : {}".format(className.index(i),i))
                print("Nhập tên file hình ảnh")
                img_name = input("Nhập tên file hình ảnh:")
                handleSourceImage(img_name)
            if query == 'thoát' :
                speak('Xin chào, hẹn gặp lại')
                break            
ttbh()

