#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from nite2 import *

rc = OpenNI.initialize()
if rc != OPENNI_STATUS_OK:
    raise Exception('initialization error')

dev = Device()

rc = dev.open()
if rc != OPENNI_STATUS_OK:
    raise Exception('device open error: ' + str(rc))

near_mode = dev.isNearModeSupported()
camera_elevation = dev.isCameraElevationSupported()
accelerometer = dev.isAccelerometerSupported()

print 'NearModeSupported:', near_mode
print 'CameraElevationSupported:', camera_elevation
print 'AccelerometerSupported: ', accelerometer

elev = [(  0, 0.5),
        (-30, 1.0),
        (-20, 0.5),
        (-10, 0.5),
        (  0, 0.5),
        ( 10, 0.5),
        ( 20, 0.5),
        ( 30, 0.5),
        (  0, 1.0)]

if camera_elevation:
    for e in elev:
        dev.setCameraElevation(e[0])
        time.sleep(e[1])
        if accelerometer:
            print dev.getAccelerometerReading()

dev.close()
OpenNI.shutdown()
