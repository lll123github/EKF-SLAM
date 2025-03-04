import numpy as np

from robot import Robot
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update
import matplotlib.pyplot as plt

# https://www.cs.utexas.edu/~pstone/Courses/393Rfall11/resources/RC09-Quinlan.pdf

# In[Generate static landmarks]

n = 50 # number of static landmarks
mapsize = 50
landmark_xy = mapsize*(np.random.rand(n,2)-0.5)
landmark_id = np.transpose([np.linspace(0,n-1,n,dtype='uint16')])
ls = np.append(landmark_xy,landmark_id,axis=1)

# In[Generate dynamic landmarks]

k = 0 # number of dynamic landmarks
vm = 5 # velocity multiplier
landmark_xy = mapsize*(np.random.rand(k,2)-0.5)
landmark_v = np.random.rand(k,2)-0.5
landmark_id = np.transpose([np.linspace(n,n+k-1,k,dtype='uint16')])
ld = np.append(landmark_xy,landmark_id,axis=1)
ld = np.append(ld,landmark_v,axis=1)


# In[Define and initialize robot parameters]

fov = 80

# Rt = 5*np.array([[0.1,0,0],
#                [0,0.01,0],
#                [0,0,0.01]])
# Qt = np.array([[0.01,0],
#                [0,0.01]])
Rt = 5*np.array([[0.01,0,0],
               [0,0.001,0],
               [0,0,0.01]])
Qt = np.array([[0.001,0],
               [0,0.001]])

x_init = [0,0,0.5*np.pi]

r1 = Robot(x_init, fov, Rt, Qt)

# In[Generate inputs and measurements]

steps = 30
stepsize = 3
curviness = 0.5

x_true = [x_init]
obs = []

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
x_true_last=[x_true[0]]+x_true
x_true_last.pop()

# In[Estimation]

# Initialize state matrices
inf = 1e6

mu = np.append(np.array([x_init]).T,np.zeros((2*(n+k),1)),axis=0)
mu_new = mu.copy()

mu_2 = np.append(np.array([x_init]).T,np.zeros((2*(n+k),1)),axis=0)
mu_new_2 = mu_2.copy()

mu_3 = np.append(np.array([x_init]).T,np.zeros((2*(n+k),1)),axis=0)
mu_new_3 = mu_3.copy()

cov = inf*np.eye(2*(n+k)+3)
cov[:3,:3] = np.zeros((3,3))
cov_2 = cov.copy()
cov_3 = cov.copy()

c_prob = 0.5*np.ones((n+k,1))
c_prob_2=c_prob.copy()
c_prob_3=c_prob.copy()

plotEstimate(mu, cov, r1, mapsize)

for movement, measurement,landmark in zip(u, obs,landmarks):

    mu_new, cov = predict(mu_new, cov, movement, Rt)
    mu = np.append(mu,mu_new,axis=1)
    plotEstimate(mu, cov, r1, mapsize)
    print('Measurements: {0:d}'.format(len(measurement)))
    mu_new, cov, c_prob_new = update(mu_new, cov, measurement, c_prob[:,-1].reshape(n+k,1), Qt)
    mu = np.append(mu,mu_new,axis=1)
    c_prob = np.append(c_prob, c_prob_new, axis=1)
    plotEstimate(mu, cov, r1, mapsize)
    plotMeasurement(mu_new, cov, measurement, n)
    plotError(mu,x_true[:len(mu[:,0::2])][:])
    print('----------')

plt.savefig('figs/ekf.png')

for movement,measurement,truth,truth_last,landmark in zip(u,obs,x_true,x_true_last,landmarks):
    mu_new_2,cov_2 = predict(mu_new_2, cov_2, movement, Rt,truth_last)
    mu_2 = np.append(mu_2,mu_new_2,axis=1)
    plotEstimate(mu_2, cov_2, r1, mapsize)
    print('Measurements: {0:d}'.format(len(measurement)))
    mu_new_2, cov_2, c_prob_new_2 = update(mu_new_2, cov_2, measurement, c_prob[:,-1].reshape(n+k,1), Qt,truth,landmark)
    mu_2 = np.append(mu_2,mu_new_2,axis=1)
    c_prob_2=np.append(c_prob_2,c_prob_new_2,axis=1)
    
    plotEstimate(mu_2, cov_2, r1, mapsize)
    plotMeasurement(mu_new_2, cov_2, measurement, n)
    plotError(mu_2,x_true[:len(mu_2[:,0::2])][:])
    print('----------')
plt.savefig('figs/ekf_truth_last.png')

for movement,measurement,truth,truth_last,landmark in zip(u,obs,x_true,x_true_last,landmarks):
    mu_new_3,cov_3 = predict(mu_new_3, cov_3, movement, Rt,truth)
    mu_3 = np.append(mu_3,mu_new_3,axis=1)
    plotEstimate(mu_3, cov_3, r1, mapsize)
    print('Measurements: {0:d}'.format(len(measurement)))
    mu_new_3, cov_3, c_prob_new_3 = update(mu_new_3, cov_3, measurement, c_prob[:,-1].reshape(n+k,1), Qt,truth,landmark)
    mu_3 = np.append(mu_3,mu_new_3,axis=1)
    c_prob_3=np.append(c_prob_3,c_prob_new_3,axis=1)
    
    plotEstimate(mu_3, cov_3, r1, mapsize)
    plotMeasurement(mu_new_3, cov_3, measurement, n)
    plotError(mu_3,x_true[:len(mu_3[:,0::2])][:])
    print('----------')
plt.savefig('figs/ekf_truth.png')