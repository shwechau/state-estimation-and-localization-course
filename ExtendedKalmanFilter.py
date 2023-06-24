#Python program for extended Klaman filter algorithm evaluating jacobuan matrices for measurement and motion model to evalauate position'
import numpy as np

def extended_kalmanfilter(x_prev,p_prev, u, y, deltaT, D,S):
    A = np.array([[1, deltaT], [0,1]]) #State transition matrix
    B = np.array([[0], [deltaT]]) #control input matrix
    Q = np.array([[0.01 , 0], [0, 0.1]]) #noise covariance
    Lk = np.array([[1,0],[0,1]]) #motion model noise jacobian
    Mk = np.array([[1]]) #measurement model noise jacobian
    wk = np.dot(0.01,Lk) #process noise
    x_pred = np.dot(A,x_prev) + np.dot(B,u)#state prediction
    np.set_printoptions(threshold=np.inf)
    p_pred = np.dot(np.dot(A,p_prev),A.T) +np.dot(np.dot(Lk,Q),Lk.T) #error covariance prediction

    H = np.array([[1,0]])
    Hk_calc = S/(((D-2.5)**2)+(S**2))
    Hk = np.array([[Hk_calc, 0]])
    R = np.array([[0.01]])
    x_resd = np.dot(Hk, (x_pred-x_prev))
    #yest = np.dot(Hk,x_pred) + x_resd + Mk
    inv_calc = (np.dot(Hk, np.dot(p_pred, Hk.T))) + (np.dot(Mk,np.dot(R,Mk.T)))
    K = np.dot(p_pred, np.dot(Hk.T,np.linalg.inv(inv_calc))) #kalman gain


    ypred = np.dot(Hk,x_pred)
    y_res = y-ypred
    x_correction = x_pred + np.dot(K, y_res)
    p_init = (1-np.dot(K,Hk))
    p_correction = np.dot(p_init,p_pred)

    return x_correction,p_correction


#Initial state and covariance
xo = np.array([[0],[5]]) #initial state mean
po = np.array([[0.01,  0],[0, 1]]) #initial state covariance

deltaT = 0.5 #time step
u0 = -2 #control input (m/s2)
y1 = np.pi/6 #measurement (rad/s)
D = 40 #measuremet noise (m)
S = 20 #Measurement distnace (m)

x_prev = xo
p_prev = po
u = u0
y = y1

x_est, p_est = extended_kalmanfilter(x_prev,p_prev, u, y, deltaT, D,S)
