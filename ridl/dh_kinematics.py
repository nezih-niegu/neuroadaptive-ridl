import math
import numpy as np
from scipy.optimize import least_squares

# Denavit–Hartenberg parameters
PSM_DH = [
    [-math.pi/2, 0.0, 0.0, 0.0, 'R'],
    [ math.pi/2, 0.0, 0.0, 0.0, 'R'],
    [      0.0,  0.0, 0.0, 0.0, 'P'],
    [-math.pi/2, 0.0, 0.0, 0.0, 'R'],
    [ math.pi/2, 0.0, 0.0, 0.0, 'R'],
    [-math.pi/2, 0.0, 0.0, 0.0, 'R'],
    [      0.0,  0.0, 0.0, 0.0, 'R'],
]

def dh(a, alpha, d, theta):
    return np.array([
        [ math.cos(theta), -math.sin(theta)*math.cos(alpha),
          math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
        [ math.sin(theta),  math.cos(theta)*math.cos(alpha),
         -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
        [           0.0,               math.sin(alpha),
                    math.cos(alpha),               d],
        [           0.0,                        0.0,
                           0.0,            1.0]
    ])

def fk(q):
    T = np.eye(4)
    for alpha,a,d0,th0,jt in PSM_DH:
        if jt=='R':
            th, d = th0+q[len(T)-1], d0
        else:
            th, d = (th0, d0+q[len(T)-1]) if jt=='P' else (th0,d0)
        T = T @ dh(a, alpha, d, th)
    R = T[:3,:3]
    x,y,z = T[:3,3]
    roll  = math.atan2(R[2,1], R[2,2])
    pitch = math.atan2(-R[2,0], math.hypot(R[0,0],R[1,0]))
    yaw   = math.atan2(R[1,0], R[0,0])
    return np.array([x,y,z,roll,pitch,yaw])

def ik_err(q, tgt):
    return fk(q) - tgt

def joint_positions(q):
    T = np.eye(4)
    pts = [(0,0,0)]
    for alpha,a,d0,th0,jt in PSM_DH:
        if jt=='R':
            th, d = th0+q[len(pts)-1], d0
        else:
            th, d = (th0, d0+q[len(pts)-1]) if jt=='P' else (th0,d0)
        T = T @ dh(a, alpha, d, th)
        pts.append(tuple(T[:3,3]))
    return np.array(pts)

def solve_kinematics(poses):
    n = poses.shape[0]
    qs = np.zeros((n,7))
    q0 = np.zeros(7)
    for i, tgt in enumerate(poses):
        sol = least_squares(ik_err, q0, args=(tgt,), xtol=1e-6, ftol=1e-6)
        qs[i], q0 = sol.x, sol.x
    jtraj = [joint_positions(q) for q in qs]
    return qs, jtraj
