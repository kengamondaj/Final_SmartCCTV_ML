import cv2
import time 
import threading 
from time import gmtime, strftime,localtime

def cap_stream(FPS,URL,path = ''):
    #print("Before URL")
    cap = cv2.VideoCapture(URL)
    #print("After URL")
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(path + 'outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))
    prev_time = 0
    use = True

    while True:
        cur_time = int(time.time())
        name = strftime("%Y%m%d%H%M%S", localtime())
        if (cur_time%60)%15 == 0 :
            if cur_time != prev_time :
                print((prev_time , cur_time,time.ctime(cur_time)))
                out.release()
                out = cv2.VideoWriter(path + name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))
        prev_time = cur_time

        #reduce frame rate
        ret, frame = cap.read()
#         if use == True :
        out.write(frame)
#             use = False
#         else :
#             use = True 

        #print('About to show frame of Video.')
        # cv2.imshow("Capturing",frame)
        #print('Running..')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

URL = 'http://admin:01290129@192.168.1.120/VIDEO.CGI'
path = './120/'

cap_stream(20,URL,path)