#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from nite2 import *

print 'start...'

rc = OpenNI.initialize()
if rc != OPENNI_STATUS_OK:
    raise Exception('initialization error')

print 'OpenNI initialized'

dev = Device()

rc = dev.open()
if rc != OPENNI_STATUS_OK:
    raise Exception('device open error: ' + str(rc))

print 'device openend'

col = VideoStream()
dep = VideoStream()
rec = Recorder()

print 'streams & recorder objects created'

rc = rec.create('rec.oni')
if rc != OPENNI_STATUS_OK:
    raise Exception('recorder create error: ' + str(rc))

print 'recorder created'

rc = col.create(dev, SENSOR_COLOR)
if rc != OPENNI_STATUS_OK:
    raise Exception('color stream create error: ' + str(rc))

print 'color stream created'
    
rc = dep.create(dev, SENSOR_DEPTH)
if rc != OPENNI_STATUS_OK:
    raise Exception('depth stream create error: ' + str(rc))

print 'depth stream created'

rc = rec.attach(col, True)
if rc != OPENNI_STATUS_OK:
    raise Exception('recorder attach error: ' + str(rc))

print 'attached color stream'

rc = rec.attach(dep, True)
if rc != OPENNI_STATUS_OK:
    raise Exception('recorder attach error: ' + str(rc))

print 'attached depth stream'

rc = col.start()
if rc != OPENNI_STATUS_OK:
    raise Exception('color stream start error: ' + str(rc))

print 'color stream start'

rc = dep.start()
if rc != OPENNI_STATUS_OK:
    raise Exception('depth stream start error: ' + str(rc))

print 'depth stream start'

rc = rec.start()
if rc != OPENNI_STATUS_OK:
    raise Exception('recorder start error: ' + str(rc))

print 'recorder start'

recording_time = 10
print 'Recording... Please wait ' + str(recording_time) + ' seconds...'
time.sleep(recording_time)
print 'Finished...'

rec.stop()
rec.destroy()
col.stop()
dep.stop()
col.destroy()
dep.destroy()
dev.close()
OpenNI.shutdown()
print 'shutdown'
