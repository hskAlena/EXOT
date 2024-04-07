import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
from pick_place import pick2,place,generate_postion
import urx
import random
import time
robot = urx.Robot('192.168.1.66')
#i=0
while True :
    data = 3
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878, 0.4674954414367676],acc=0.5,vel=0.5,relative=False)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    pipeline.start(config)

    position=[]
    w_h=[]

    while True:        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
        
        #colorwriter.write(color_image)
        #depthwriter.write(depth_colormap)
        
        cv2.imshow('Stream', color_image)
        
        if cv2.waitKey(1) == ord("q"):

            pipeline.stop()
            cv2.destroyAllWindows()
            #cv2.imwrite("mj_setting/depth_image.png",depth_image)
            #cv2.imwrite("mj_setting/depth_colormap.png",depth_colormap)
            #cv2.imwrite("mj_setting/color_image.png",color_image)
        
            depth_crop=depth_image[80: 80 + 300, 50: 50 + 560]
            img=depth_crop/(depth_crop.max()/255.0)
            cv2.imwrite("mj_setting/sample.png",img)
            img = cv2.imread("mj_setting/sample.png")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("crop_depth2.png",img)
            img_color = cv2.imread("crop_depth2.png")
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            alpha = 1
            img_gray = np.clip(((1 + alpha) * img_gray - 110 * alpha), 0, 255)
        
            ret, img_binary = cv2.threshold(img_gray, 100, 200, 0)
            _,contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #print(depth_crop.shape)
            #print(img_gray.shape)
     
            for cnt in contours:
                area = cv2.contourArea(cnt)
               
                if area>500 and area<6000 :
                    cv2.drawContours(img_gray, [cnt], 0, (255, 0, 0), 3)  # blue
                    M = cv2.moments(cnt)
   

                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    #print(box)
                    box = np.int0(box)
                    cv2.drawContours(img_gray,[box],0,(0,0,255),2)
                    #print(box[0][0])
                    width = np.int0(((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5)
                    height = np.int0(((box[0][0]-box[2][0])**2+(box[0][1]-box[2][1])**2)**0.5)
                    print(width,height)
                    print(area)

                    cv2.circle(img_gray, (cx, cy), 5, (255,0,255), -1)
                    cz = (depth_crop[cy][cx]-588)*0.001*0.70
                    position.append([cx, cy, cz])
                    w_h.append([width,height])

                    #print(cx,cy)
                    print("depth : %d"%depth_crop[cy][cx])

            cv2.imwrite("position.png",img_gray)
            print(position)
            break
    #pick2(position[0])
    for i in range(0, len(position)):        
        data = 0
        with open('robot-cam.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        time.sleep(2)
        z= pick2(position[i])#pick object

        
        x2,y2=generate_postion(w_h[i],z)
        place(x2,y2,z)
        
        

    #data = 0
    #with open('robot-cam.pkl', 'wb') as f:
	#pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    #time.sleep(2)
    #ppr2(position[i+2])

