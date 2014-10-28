import os
import cv2
import time
import threading
import numpy as np

print 'PID:', os.getpid()

t = 10 # seconds
for _ in range(int(10*t)):
    time.sleep(0.1)

import nite2 as n

rgb_data = None
rgb_data_mutex = threading.RLock()

depth_data = None
depth_data_mutex = threading.RLock()

def new_rgb_frame_callback(video_stream):
    global rgb_data, rgb_data_mutex
    rgb_data_mutex.acquire()
    try:
        rc, frame  = video_stream.readFrame()
        if rc != n.OPENNI_STATUS_OK:
            print 'rgb frame error'
        rgb_data = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print 'Exception', e
    finally:
        rgb_data_mutex.release()

def new_depth_frame_callback(video_stream):
    global depth_data, depth_data_mutex
    depth_data_mutex.acquire()
    try:
        rc, frame  = video_stream.readFrame()
        if rc != n.OPENNI_STATUS_OK:
            print 'depth frame error'
        depth_data = np.float32(frame.data)
        depth_data = depth_data * (-255.0/3200.0) + (800.0*255.0/3200.0)
        depth_data = depth_data.astype('uint8')
    except Exception as e:
        print 'Exception', e
    finally:
        depth_data_mutex.release()

rc = n.OpenNI.initialize()
if rc != n.OPENNI_STATUS_OK:
    raise Exception('OpenNI initialization failed: ' + str(rc))

print 'init'

dev = n.Device()

rc = dev.open()
if rc != n.OPENNI_STATUS_OK:
    raise Exception('Device open error: ' + str(rc))

print 'open device'

col = n.VideoStream()
dep = n.VideoStream()

print 'creating'

col.create(dev, n.SENSOR_COLOR)
dep.create(dev, n.SENSOR_DEPTH)

print 'starting'

col.start()
dep.start()

print 'started'

rc = col.addNewFrameListener(new_rgb_frame_callback)
if rc != n.OPENNI_STATUS_OK:
    raise Exception('addNewFrameListener error: ' + str(rc))

rc = dep.addNewFrameListener(new_depth_frame_callback)
if rc != n.OPENNI_STATUS_OK:
    raise Exception('addNewFrameListener error: ' + str(rc))

print 'listeners registered'

try:
    while True:
        k = cv2.waitKey(10)
        if k == 27:
            break

        rgb_data_mutex.acquire()
        try:
            if rgb_data is not None:
                cv2.imshow('video', rgb_data)
        except:
            print 'rgb exception'
        finally:
            rgb_data_mutex.release()

        depth_data_mutex.acquire()
        try:
            if depth_data is not None:
                cv2.imshow('depth', depth_data)
        except:
            print 'depth exception'
        finally:
            depth_data_mutex.release()
except:
    print 'exception!!!'
finally:
    print 'the end'
    cv2.destroyAllWindows()
    print 'destroyAllWindows'
    col.removeNewFrameListener(new_rgb_frame_callback)
    dep.removeNewFrameListener(new_depth_frame_callback)
    print 'removeNewFrameListener'
    col.destroy()
    print 'col.destroy'
    dep.destroy()
    print 'dep.destroy'
    dev.close()
    print 'dev.close'
    n.OpenNI.shutdown()
    print 'shutdown'
