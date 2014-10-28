#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nite2 import *

rc = OpenNI.initialize()
if rc != OPENNI_STATUS_OK:
    raise Exception('OpennNI initialization error')

rc = NiTE.initialize()
if rc != NITE_STATUS_OK:
    raise Exception('NiTE initialization error')

dev = Device()

rc = dev.open()
# rc = dev.open(None) # should be the same as with no parameters
if rc != OPENNI_STATUS_OK:
    raise Exception('device open error: ' + str(rc))

ht = HandTracker()

rc = ht.create(dev)
if rc != NITE_STATUS_OK:
    raise Exception('error creating hand tracker')

ht.startGestureDetection(GESTURE_WAVE)
ht.startGestureDetection(GESTURE_CLICK)
ht.startGestureDetection(GESTURE_HAND_RAISE)
print 'Waiting for hands...'

try:
    while True:
        rc, frame = ht.readFrame()
        if rc != NITE_STATUS_OK:
            print '!'
            continue

        for g in frame.gestures:
            if g.isComplete():
                rc, new_id = ht.startHandTracking(g.currentPosition)
                if rc != NITE_STATUS_OK:
                    print 'start tracking failed'
                else:
                    print 'new hand:', new_id

        for h in frame.hands:
            if h.isTracking():
                print h.id, h.position
finally:
    ht.destroy()
    dev.close()
    NiTE.shutdown()
    OpenNI.shutdown()
    print 'shutdown'
