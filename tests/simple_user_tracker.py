
from nite2 import *

rc = OpenNI.initialize()
if rc != OPENNI_STATUS_OK:
    raise Exception('OpenNI initialization failed')

rc = NiTE.initialize()
if rc != NITE_STATUS_OK:
    raise Exception('NiTE initialization failed')

dev = Device()

rc = dev.open()
if rc != OPENNI_STATUS_OK:
    raise Exception('NiTE initialization failed: ' + str(rc))

ut = UserTracker()

rc = ut.create()
if rc != NITE_STATUS_OK:
    raise Exception('error creating user tracker')

print 'Start moving around to get detected...'
print '(PSI pose may be required for skeleton calibration, depending on the configuration)'

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


try:
    while True:
        rc, frame = ut.readFrame()
        if rc != NITE_STATUS_OK:
            print 'dupa'
            break
        users = frame.users
        for u in users:
            update_user_state(u, frame.timestamp)
            if u.isNew():
                ut.startSkeletonTracking(u.id)
            elif u.skeleton.state == SKELETON_TRACKED:
                skel = u.skeleton
                #head = skel.getJoint(JOINT_HEAD)
                head = skel[JOINT_HEAD]
                if head.positionConfidence > 0.5:
                    print u.id, head.position
finally:
    ut.destroy()
    dev.close()
    NiTE.shutdown()
    OpenNI.shutdown()
    print 'shutdown'
