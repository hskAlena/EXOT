import cv2
from PIL import Image
def draw_box(img, box):
    [x, y, w, h] = box.split('\t')
    x = int(x)
    y = int(y)
    x2 = int(w)+x
    y2 = int(h[:-1])+y

    #image_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (x,y), (x2, y2), color=(0,0,255), thickness=2)
    image_BGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image_BGR

import glob
import imageio
images = glob.glob('/home/user/hsproj/Stark/data/TREK-150/P03-P03_09-4/img/*.jpg')
images.sort(key=lambda f: f[-14:-4])
print(images)
with open('~/hsproj/Stark/test/tracking_results/stark_st/baseline_got10k_only/P03-P03_P09-4.txt', 'r') as f:
    bboxes = f.readlines()

imgarray = []
for i in range(len(images)):
    img = cv2.imread(images[i])#'../robot-data/2/frame_0003.jpg')
    imgarray.append(Image.fromarray(draw_box(img, bboxes[i])))
imageio.mimsave('got_P03-P03_P09-4.gif',imgarray, fps=60)
