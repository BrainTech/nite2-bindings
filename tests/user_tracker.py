#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
from nite2 import *

g_visibleUsers = 10*[False]
g_skeletonStates = 10*[SKELETON_NONE]

def update_user_state(user, ts):
    if user.isNew():
        print 'User:', user.id, ' - New'
    elif user.isVisible() and (not g_visibleUsers[user.id]):
        print 'User:', user.id, ' - Visible'
    elif (not user.isVisible()) and g_visibleUsers[user.id]:
        print 'User:', user.id, ' - Out of Scene'
    elif user.isLost():
        print 'User:', user.id, ' - Lost'

    g_visibleUsers[user.id] = user.isVisible()

    if g_skeletonStates[user.id] != user.skeleton.state:
        state = g_skeletonStates[user.id] = user.skeleton.state
        if state == SKELETON_NONE:
            print 'User:', user.id, ' - Stopped tracking'
        elif state == SKELETON_CALIBRATING:
            print 'User:', user.id, ' - Calibrating...'
        elif state == SKELETON_TRACKED:
            print 'User:', user.id, ' - Tracking!'
        elif state in [SKELETON_CALIBRATION_ERROR_NOT_IN_POSE, 
                       SKELETON_CALIBRATION_ERROR_HANDS,
                       SKELETON_CALIBRATION_ERROR_LEGS,
                       SKELETON_CALIBRATION_ERROR_HEAD,
                       SKELETON_CALIBRATION_ERROR_TORSO]:
            print 'User:', user.id, ' - Calibration Failed... :-|'

connections = ((JOINT_HEAD, JOINT_NECK),
	       (JOINT_NECK, JOINT_TORSO),
	       (JOINT_TORSO, JOINT_LEFT_SHOULDER),
	       (JOINT_TORSO, JOINT_RIGHT_SHOULDER),
	       (JOINT_LEFT_SHOULDER, JOINT_LEFT_ELBOW),
	       (JOINT_RIGHT_SHOULDER, JOINT_RIGHT_ELBOW),
	       (JOINT_LEFT_ELBOW, JOINT_LEFT_HAND),
	       (JOINT_RIGHT_ELBOW, JOINT_RIGHT_HAND),
	       (JOINT_TORSO, JOINT_LEFT_HIP),
	       (JOINT_TORSO, JOINT_RIGHT_HIP),
	       (JOINT_LEFT_HIP, JOINT_LEFT_KNEE),
	       (JOINT_RIGHT_HIP, JOINT_RIGHT_KNEE),
	       (JOINT_LEFT_KNEE, JOINT_LEFT_FOOT),
	       (JOINT_RIGHT_KNEE, JOINT_RIGHT_FOOT))

joints = (JOINT_HEAD,
	  JOINT_LEFT_ELBOW,
	  JOINT_LEFT_FOOT,
	  JOINT_LEFT_HAND,
	  JOINT_LEFT_HIP,
	  JOINT_LEFT_KNEE,
	  JOINT_LEFT_SHOULDER,
	  JOINT_NECK,
	  JOINT_RIGHT_ELBOW,
	  JOINT_RIGHT_FOOT,
	  JOINT_RIGHT_HAND,
	  JOINT_RIGHT_HIP,
	  JOINT_RIGHT_KNEE,
	  JOINT_RIGHT_SHOULDER,
	  JOINT_TORSO)

def draw_user(img, skeleton):
    for j in joints:
        joint = skeleton.getJoint(j)
        rc, x_new, y_new = ut.convertJointCoordinatesToDepth(joint.position)
        if joint.positionConfidence > 0.5:
            cv2.circle(img, (int(x_new), int(y_new)), 8, (0, 255, 0), -1)
        else:
            cv2.circle(img, (int(x_new), int(y_new)), 8, (0, 0, 255), -1)
    for c in connections:
        joint_1 = skeleton.getJoint(c[0])
        joint_2 = skeleton.getJoint(c[1])
        rc, x1, y1 = ut.convertJointCoordinatesToDepth(joint_1.position.x, joint_1.position.y, joint_1.position.z)
        rc, x2, y2 = ut.convertJointCoordinatesToDepth(joint_2.position.x, joint_2.position.y, joint_2.position.z)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255))


if __name__ == "__main__":
    
    rc = OpenNI.initialize()
    if rc != OPENNI_STATUS_OK:
        raise Exception('OpenNI initialize error: ' + str(rc))

    rc = NiTE.initialize()
    if rc != NITE_STATUS_OK:
        raise Exception('NiTE initialize error: ' + str(rc))

    dev = Device()
    rc = dev.open()
    if rc != OPENNI_STATUS_OK:
        raise Exception('device open error: ' + str(rc))

    if dev.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    col = VideoStream()
    dep = VideoStream()
    
    rc = col.create(dev, SENSOR_COLOR)
    if rc != OPENNI_STATUS_OK:
        raise Exception('color stream create error: ' + str(rc))

    rc = dep.create(dev, SENSOR_DEPTH)
    if rc != OPENNI_STATUS_OK:
        raise Exception('depth stream create error: ' + str(rc))

    rc = col.start()
    if rc != OPENNI_STATUS_OK:
        raise Exception('color stream start error: ' + str(rc))

    rc = dep.start()
    if rc != OPENNI_STATUS_OK:
        raise Exception('depth stream start error: ', + str(rc))
    
    ut = UserTracker()
    
    rc = ut.create()
    if rc != NITE_STATUS_OK:
        raise Exception('error creating user tracker: ' + str(rc))

    print 'Start moving around to get detected...'
    print '(PSI pose may be required for skeleton calibration, depending on the configuration)'

    try:
        while True:
            k =  cv2.waitKey(10)
            if k == 27:
                break
                
            rc, color = col.readFrame()
            if rc != OPENNI_STATUS_OK:
                raise Exception('error reading color frame: ' + str(rc))

            rc, depth = dep.readFrame()
            if rc != OPENNI_STATUS_OK:
                raise Exception('error reading depth frame: ' + str(rc))

            if color.isValid() and depth.isValid():
                data_rgb = color.data
                data_rgb = cv2.cvtColor(data_rgb, cv2.COLOR_BGR2RGB)
                data_depth = depth.data
                data_depth = np.float32(data_depth)
                data_depth = data_depth * (-255.0/3200.0) + (800.0*255.0/3200.0)
                data_depth = data_depth.astype('uint8')
                data_depth = cv2.cvtColor(data_depth, cv2.COLOR_GRAY2RGB)
            else:
                print '!'

            rc, frame = ut.readFrame()
            if rc != OPENNI_STATUS_OK:
                raise Exception('read frame error: ' + str(rc))
                
            users = frame.users
            for u in users:
                update_user_state(u, frame.timestamp)
                if u.isNew():
                    ut.startSkeletonTracking(u.id)
                elif u.skeleton.state == SKELETON_TRACKED:
                    draw_user(data_rgb, u.skeleton)
                    draw_user(data_depth, u.skeleton)
                    
            cv2.imshow('depth', data_rgb)
            cv2.imshow('color', data_depth)
                    
    finally:
        cv2.destroyAllWindows()
        ut.destroy()
        col.destroy()
        dep.destroy()
        dev.close()
        OpenNI.shutdown()
        NiTE.shutdown()
        print 'shutdown'
        
