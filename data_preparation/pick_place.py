import urx
import time
from basis import bs, bsr
from Dependencies.urx_custom.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
robot = urx.Robot('192.168.1.66')
gripper = Robotiq_Two_Finger_Gripper(robot)
import pickle
import random
import numpy as np
import pyrealsense2 as rs
import cv2

def pp(x1,y1,x2,y2):#relative=False, pick(x1,y1), place(x2,y2)
    gripper.open_gripper()
    robot.movej([2.9763903617858887, -0.7628591817668458, 2.009418789540426, -6.1137219868102015, -0.7992594877826136, 1.8710598945617676],acc=0.3,vel=0.3,relative=False)
    #robot.movel(bs(0,y1,0,0,0,0), acc=0.1, vel=0.1, relative=False)  # move y1
    robot.movel(bs(x1, y1, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # move x1
    robot.movel(bs(x1, y1,-0.3, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # down
    time.sleep(0.5)
    gripper.gripper_action(255)
    robot.movel(bs(x1, y1, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # up
    robot.movel(bs(x1, y2, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # move y2
    robot.movel(bs(x2, y2, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # move x2
    robot.movel(bs(x2, y2,-0.3, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # down
    gripper.gripper_action(0)
    robot.movel(bs(x2, y2, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=False)  # up
    robot.movel(bs(x2,0,0,0,0,0), acc=0.1, vel=0.1, relative=False) #move (x2,0,0)
    robot.movel(bs(0,0,0,0,0,0), acc=0.1, vel=0.1, relative=False) #move (0,0,0)
    gripper.gripper_action(0)
    robot.close()

def ppr(x1,y1,x2,y2):#relative=True, pick(+x1,+y1), place(+x1+x2, +y1+y2)
    data = 0


    gripper.gripper_action(0)
    time.sleep(2)
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    robot.movej([2.8075432777404785, 1.2403766351887207, -1.642167091369629, -4.633544107476705, -0.8376925627337855, 0.41880083084106445],acc=0.3,vel=0.3,relative=False)

    time.sleep(2)
    robot.movej([2.976461887359619, -0.7628591817668458, 2.0093711058246058, -6.113685747186178, -0.7992356459247034, 0.1757826805114746],acc=0.3,vel=0.3,relative=False)
    #robot.movel(bsr(0,y1,0,0,0,0), acc=0.1, vel=0.1, relative=True)  # move y1
    robot.movel(bsr(x1, y1, 0, 0, 0, 0), acc=0.2, vel=0.2, relative=True)  # move x1
    robot.movel(bsr(0, 0,-0.18, 0, 0, 0), acc=0.2, vel=0.2, relative=True)  # down
    
    gripper.close_gripper()
    time.sleep(2)
    robot.movel(bsr(0, 0, +0.18, 0, 0, 0), acc=0.2, vel=0.2, relative=True)  # up
    data = 1
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(2)
    data = 3
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(2)
    #robot.movel(bsr(0, y2, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=True)  # move y2
    robot.movel(bsr(x2, y2, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=True)  # move x2
    robot.movel(bsr(0, 0,-0.18, 0, 0, 0), acc=0.1, vel=0.1, relative=True)  # down
    gripper.gripper_action(0)
    time.sleep(2)
    robot.movel(bsr(0, 0, 0.18, 0, 0, 0), acc=0.1, vel=0.1, relative=True)  # up
    gripper.close_gripper()
    time.sleep(2)
    robot.close()

def pick2(position):
    x1 = position[0]*(0.0011)-0.266
    y1 = position[1]*(-0.0011)+0.32
    x2 = random.randrange(50,410)*(0.0011)-0.266
    y2 = random.randrange(50,280)*(-0.0011)+0.32
    z = position[2]+0.18
    print("z:%f"%z)
    data = 0

    gripper.gripper_action(0)
    time.sleep(2)
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(2)
    #data = 1
    #with open('robot-cam.pkl', 'wb') as f:
	#pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    #time.sleep(1)
    robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878, 0.4674954414367676],acc=0.5,vel=0.5,relative=False)
    data = 1
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(3)
    
    robot.movej([2.976461887359619, -0.7628591817668458, 2.0093711058246058, -6.113685747186178, -0.7992356459247034, 0.1757826805114746],acc=0.5,vel=0.5,relative=False)
    #robot.movel(bsr(0,y1,0,0,0,0), acc=0.1, vel=0.1, relative=True)  # move y1
    robot.movel(bsr(x1, y1, 0, 0, 0, 0), acc=0.2, vel=0.2, relative=True)  # move x1
    robot.movel(bsr(0, 0,-z, 0, 0, 0), acc=0.2, vel=0.2, relative=True)  # down
    
    gripper.close_gripper()
    time.sleep(2)
    robot.movel(bsr(0, 0, z, 0, 0, 0), acc=0.2, vel=0.2, relative=True)  # up
    data = 2
    
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    time.sleep(2)
    return z

def generate_postion(rect,z):
    print("generate_postion")
    robot.movej([2.976461887359619, -0.7628591817668458, 2.0093711058246058, -6.113685747186178, -0.7992356459247034, 0.1757826805114746],acc=0.5,vel=0.5,relative=False)
    robot.movel(bsr(0.25,0.05, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)
    a=robot.getj()
    robot.movel(bsr(0, 0,-z, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
    gripper.gripper_action(0)
    time.sleep(2)

    robot.movej([2.7695469856262207, 1.3372728067585449, -1.7796506881713867, -4.625200887719625, -0.8506844679461878, 0.4674954414367676],acc=0.5,vel=0.5,relative=False) #take picture
    

    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    
    i=0
    
    for i in range(1,20):    
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
    
        if not depth_frame:
            continue
        

        d_image = np.asanyarray(depth_frame.get_data())
    pipeline.stop()
    
    d_crop=d_image[80: 80 + 300, 50: 50 + 560]
    robot.movej(a,acc=0.5,vel=0.5,relative=False)
    robot.movel(bsr(0, 0,-z, 0, 0, 0), acc=0.2, vel=0.2, relative=True)
    gripper.close_gripper()
    time.sleep(2)
    robot.movel(bsr(0, 0,z, 0, 0, 0), acc=0.2, vel=0.2, relative=True)   
       
   
    
    retry = True
    
    width=rect[0]
    height=rect[1]
    print(width,height)
    while retry==True :
        flag = 0
        #x2 = random.randrange(50,410)*(0.0011)-0.266
        #y2 = random.randrange(50,410)*(-0.0011)+0.32
        a1 = random.randrange(50,410)
        b1 = random.randrange(50,280)
        if int(b1-height/2) < 0:
            start1 = 0
        else :
            start1=int(b1-height/2)
        if int(b1+height/2) > 300:
            end1 = 300
        else :
            end1=int(b1+height/2)
        if int(a1-(width)/2) <0 :
           start2 = 0
        else :
            start2=int(a1-width/2)
        if int(a1+(width)/2) >560 :
            end2 = 560
        else :
            end2=int(a1+width/2)
        
        for i in range(start1,end1):
            for l in range(start2,end2):
                if d_crop[i][l]<600:
                    #print(i,l)
                    flag +=1
                    
                    
        if flag == 0:
            
            retry = False
            print("retry = False")
        else :
            retry =True
            print("retry = True")
            print(flag)
    d_img=d_crop/(d_crop.max()/255.0)
    x2 = a1*(0.0011)-0.266
    y2 = b1*(-0.0011)+0.32
    print(x2,y2)
    print(start1,end1,start2,end2)

    d_img[start1:end1,start2:end2]*=0        
    cv2.imwrite("mj_setting/position_check.png",d_img)
  
    return x2,y2 
    

def place(x2,y2,z):
    print("place")
    data = 3
    with open('robot-cam.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(2)
    #robot.movel(bsr(0, y2, 0, 0, 0, 0), acc=0.1, vel=0.1, relative=True)  # move y2
    robot.movej([2.976461887359619, -0.7628591817668458, 2.0093711058246058, -6.113685747186178, -0.7992356459247034, 0.1757826805114746],acc=0.5,vel=0.5,relative=False)
    
    robot.movel(bsr(x2, y2, 0, 0, 0, 0), acc=0.5, vel=0.5, relative=True)  # move x2
    robot.movel(bsr(0, 0,-z, 0, 0, 0), acc=0.5, vel=0.5, relative=True)  # down
    gripper.gripper_action(0)
    time.sleep(2)
    robot.movel(bsr(0, 0, z, 0, 0, 0), acc=0.5, vel=0.5, relative=True)  # up
    gripper.close_gripper()
    time.sleep(2)
    





