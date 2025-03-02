import numpy as np

from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update
import matplotlib.pyplot as plt

# https://www.cs.utexas.edu/~pstone/Courses/393Rfall11/resources/RC09-Quinlan.pdf

# In[Generate static landmarks]

static_landmark_num = 50 # number of static landmarks
mapsize = 50
landmark_xy = mapsize*(np.random.rand(static_landmark_num,2)-0.5)
landmark_id = np.transpose([np.linspace(0,static_landmark_num-1,static_landmark_num,dtype='uint16')])
ls = np.append(landmark_xy,landmark_id,axis=1)

# In[Generate dynamic landmarks]

dynamic_landmark_num = 0 # number of dynamic landmarks
vm = 5 # velocity multiplier
landmark_xy = mapsize*(np.random.rand(dynamic_landmark_num,2)-0.5)
landmark_v = np.random.rand(dynamic_landmark_num,2)-0.5
landmark_id = np.transpose([np.linspace(static_landmark_num,static_landmark_num+dynamic_landmark_num-1,dynamic_landmark_num,dtype='uint16')])
ld = np.append(landmark_xy,landmark_id,axis=1)
ld = np.append(ld,landmark_v,axis=1)
fov = 80

Rt = 5*np.array([[0.1,0,0],
               [0,0.01,0],
               [0,0,0.01]])
Qt = np.array([[0.01,0],
               [0,0.01]])

x_init = [0,0,0.5*np.pi]

r1 = Robot(x_init, fov, Rt, Qt)
steps = 30
stepsize = 3
curviness = 0.5
x_true=x_init
obs=[]

# generate input sequence
u = np.zeros((steps,3))
u[:,0] = stepsize
u[4:12,1] = curviness
u[18:26,1] = curviness

# Generate random trajectory instead
#u = np.append(stepsize*np.ones((steps,1),dtype='uint8'),
#              curviness*np.random.randn(steps,2),
#              axis=1)

# generate dynamic landmark trajectories

ldt = ld
for j in range(1,steps):
    # update dynamic landmarks
    F = np.array([[1,0,0,vm,0],
                  [0,1,0,0,vm],
                  [0,0,1,0,0],
                  [0,0,0,1,0],
                  [0,0,0,0,1]])
    for i in range(len(ld)):
        ld[i,:] = F.dot(ld[i,:].T).T
    ldt = np.dstack((ldt,ld))

# generate robot states and observations
for movement, t in zip(u,range(steps)):
    landmarks = np.append(ls,ldt[:,:3,t],axis=0)
    
    # process robot movement
    x_true.append(r1.move(movement))
    obs.append(r1.sense(landmarks))

plotMap(ls,ldt,x_true,r1,mapsize)

# In[Estimation]

# Initialize state matrices
inf = 1e6

x_hat_updated=[[x_init]]
'''指的是x_k|k'''
x_hat_predicted=[[x_init]]
'''指的是x_k|k-1'''

cov = inf*np.eye(2*(static_landmark_num+dynamic_landmark_num)+3)
cov[:3,:3] = np.zeros((3,3))
cov_2 = cov.copy()

c_prob = 0.5*np.ones((static_landmark_num+dynamic_landmark_num,1))
c_prob_2=c_prob.copy()
