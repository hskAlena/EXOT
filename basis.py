import numpy as np

def bs(x,y,z,rx,ry,rz):#relative=False
	ar=np.array([[0,1,-1],[1,0,0],[0,-1,-1]])#e1,e2,e3
	arr=ar.T
	arr_xyz=np.array([[x],[y],[z]])#input_coordinate
	k=np.dot(arr,arr_xyz)
	return [0.4+float(k[0]),float(k[1]),0.5+float(k[2]),2+rx,1.5+ry,4+rz]

def bsr(x,y,z,rx,ry,rz):#relative=True
	ar=np.array([[0,1,-1],[1,0,0],[0,-1,-1]])#e1,e2,e3
	arr=ar.T
	arr_xyz=np.array([[x],[y],[z]])#input coordinate
	k=np.dot(arr,arr_xyz)
	return [float(k[0]),float(k[1]),float(k[2]),rx,ry,rz]

#start=robot.movel([0.3,0.1,0.4,2,1.5,4],acc=0.1,vel=0.1,relative=False)#(0,0,0)
#x=robot.movel([0.3,0.11,0.39,2,1.5,4],acc=0.1,vel=0.1,relative=False)#(1,0,0)
#y=robot.movel([0.31,0.1,0.4,2,1.5,4],acc=0.1,vel=0.1,relative=False)#(0,1,0)
#z=robot.movel([0.3,0.09,0.39,2,1.5,4],acc=0.1,vel=0.1,relative=False)#(0,0,1)
