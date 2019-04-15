import numpy as np


def rotX(theta):
    return np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])


def rotY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rotZ(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def euler_matrix(x, y, z):
    return rotX(x).dot(rotY(y)).dot(rotZ(z))


def vector_slerp(v1, v2, fraction):
    perp_v = np.cross(v1, v2)
    # perp_v /= np.linalg.norm(perp_v)
    angle = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) * fraction
    return rotation_matrix(angle, perp_v).dot(v1)


def unit_vector(v):
    return v/np.linalg.norm(v)


def rotation_matrix(angle, direction):
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = unit_vector(direction)
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                   [ direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
    return R
