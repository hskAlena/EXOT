import pyrealsense2 as rs
import numpy as np
import cv2
import pickle

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#color_path = 'V00P00A00C00_rgb.avi'
#depth_path = 'V00P00A00C00_depth.avi'
#colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
#depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)

pipeline.start(config)


frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

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
        cv2.imwrite("mj_setting/depth_image.png",depth_image)
        cv2.imwrite("mj_setting/depth_colormap.png",depth_colormap)
        cv2.imwrite("mj_setting/color_image.png",color_image)
        
        data = []
        data.append(depth_image)
        data.append(depth_colormap)

        with open('robot-cam.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        break
