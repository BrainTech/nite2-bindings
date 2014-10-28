import os
import time
import numpy as np
import cv2

print 'PID:', os.getpid()

time.sleep(1)

def main():
    import nite2 as n

    def draw_hands(img, hands):
        for h in hands:
            if h.isTracking():
                rc, x_new, y_new = ht.convertHandCoordinatesToDepth(h.position)
                cv2.circle(img, (int(x_new), int(y_new)), 8, (0, 255, 0), -1)

    rc = n.OpenNI.initialize()
    if rc != n.OPENNI_STATUS_OK:
        raise Exception('OpenNI initialization failed')

    rc = n.NiTE.initialize()
    if rc != n.NITE_STATUS_OK:
        raise Exception('NiTE initialization failed')

    dev = n.Device()

    rc = dev.open()
    if rc != n.OPENNI_STATUS_OK:
        raise Exception('NiTE initialization failed: ' + str(rc))

    if dev.isImageRegistrationModeSupported(n.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
        dev.setImageRegistrationMode(n.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    else:
        print 'IMAGE_REGISTRATION_DEPTH_TO_COLOR mode is not available' 

    col = n.VideoStream()
    dep = n.VideoStream()
    
    rc = col.create(dev, n.SENSOR_COLOR)
    if rc != n.OPENNI_STATUS_OK:
        raise Exception('color stream create error: ' + str(rc))
    
    rc = dep.create(dev, n.SENSOR_DEPTH)
    if rc != n.OPENNI_STATUS_OK:
        raise Exception('depth stream create error: ' + str(rc))
    
    rc = col.start()
    if rc != n.OPENNI_STATUS_OK:
        raise Exception('color stream start error: ' + str(rc))
    
    rc = dep.start()
    if rc != n.OPENNI_STATUS_OK:
        raise Exception('depth stream start error: ', + str(rc))

    ht = n.HandTracker()

    rc = ht.create(dev)
    if rc != n.NITE_STATUS_OK:
        raise Exception('creating hand tracker failed')

    ht.setSmoothingFactor(0.1)

    ht.startGestureDetection(n.GESTURE_WAVE)
    ht.startGestureDetection(n.GESTURE_CLICK)
    ht.startGestureDetection(n.GESTURE_HAND_RAISE)
    print 'Move your hand to start tracking...'

    try:
        while True:
            k = cv2.waitKey(10)
            if k == 27:
                break

            rc, color = col.readFrame()
            if rc != n.OPENNI_STATUS_OK:
                print 'error reading color frame'

            rc, depth = dep.readFrame()
            if rc != n.OPENNI_STATUS_OK:
                print 'error reading depth frame'

            #print 'debug me'
            #time.sleep(20)
            
            if color.isValid():
                data_rgb = color.data
                #print data.shape
                data_rgb = cv2.cvtColor(data_rgb, cv2.COLOR_BGR2RGB)

            if depth.isValid():
                data_depth = depth.data
                #print data_depth.shape
                data_depth = np.float32(data_depth)
                data_depth = data_depth * (-255.0/3200.0) + (800.0*255.0/3200.0)
                data_depth = data_depth.astype('uint8')
                data_depth = cv2.cvtColor(data_depth, cv2.COLOR_GRAY2RGB)

            rc, frame = ht.readFrame()
            if rc != n.NITE_STATUS_OK:
                print 'error read frame', rc

            gestures = frame.gestures
            for g in gestures:
                if g.isComplete():
                    rc, newId = ht.startHandTracking(g.currentPosition)

            hands = frame.hands
            draw_hands(data_depth, hands)
            draw_hands(data_rgb, hands)

            cv2.imshow('depth', data_depth)
            cv2.imshow('color', data_rgb)
            
    finally:
        cv2.destroyAllWindows()
        ht.destroy()
        col.stop()
        dep.stop()
        col.destroy()
        dep.destroy()
        dev.close()
        n.NiTE.shutdown()
        n.OpenNI.shutdown()
        print 'shutdown'

if __name__ == '__main__':
    main()
