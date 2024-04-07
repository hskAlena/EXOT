import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import imageio
#from PIL import Image
import matplotlib.image as mpimg
import pickle
import time
import os
import urx

robot = urx.Robot('192.168.1.66')

# Configure depth and color streams
#pipeline = rs.pipeline()
#config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#config.enable_record_to_file('object_detection.bag')

# Start streaming
#pipeline.start(config)

def main():
    img_list=[]
    depth_list=[]
    i=0
    
    while True:
        #data = 3
        time.sleep(0.05)

        with open('robot-cam.pkl', 'rb') as f:
             data = pickle.load(f)
        dataset = os.listdir("hs_setting/data_Depth")
        num_data=len(dataset)
        if data == 0:
            print("done here")
            t=0
            #dataset = os.listdir("hs_setting/data_RGB")
            #num_data=len(dataset)
            
            pipeline = rs.pipeline()
            config = rs.config()
            #dataset = os.listdir("hs_setting/data_RGB")
            #num_data=len(dataset)
            #num_data=i
            color_path = 'hs_setting/data_RGB/%d.avi'%num_data

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
            pipeline.start(config)
            time.sleep(3)
            
        if data == 1:
            print("pipe wait")
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            #depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_list.append(depth_image)
            #im = Image.fromarray(color_image)
            #im.save("before.png")

            colorwriter.write(color_image)
           
            image = color_image[:,:,::-1]
            
            a= robot.getl()
            #im2 = Image.fromarray(image)
            
            #img_list.append(im2)
            print("recording")
            
            
        elif data == 2:
            if t == 0 :
                print(i)

                #dataset = os.listdir("hs_setting/data_RGB")
                #num_data=len(dataset)
            #pipeline.stop()		
            #imageio.mimsave('hs_setting/data_RGB/%d.gif'%num_data, img_list, fps=30)
           
                colorwriter.release()
                with open('hs_setting/data_Depth/%d.pkl'%num_data, 'wb') as f:
                    pickle.dump(depth_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                img_list=[]
                print("done")
                pipeline.stop()
                t+=1
            i+=1
main()
