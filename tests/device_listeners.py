#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import nite2 as n

def print_devices(devices):
    if len(devices) == 0:
        print 'No devices found'
    else:
        print 'Available devices (%d):\n' % len(devices)
        for d in devices:
            print 'URI           :', d.uri
            print 'Name          :', d.name
            print 'Vendor        :', d.vendor
            print 'USB Vendor Id :', d.usbVendorId
            print 'USB Product Id:', d.usbProductId
            print ''

def callback_connected(dev_info):
    print 'connected:     ' + dev_info.name + ' (' + dev_info.uri + ')'

def callback_disconnected(dev_info):
    print 'disconnected:  ' + dev_info.name + ' (' + dev_info.uri + ')'

def callback_state(dev_info, state):
    print 'state changed: ' + dev_info.name + ' (' + dev_info.uri + '), state: ' + str(state)

rc = n.OpenNI.initialize()
if rc != n.OPENNI_STATUS_OK:
    raise Exception('initialization error')

print_devices(n.OpenNI.enumerateDevices())

# should be called only once per event
n.OpenNI.addDeviceConnectedListener(callback_connected)
n.OpenNI.addDeviceConnectedListener(callback_connected)

n.OpenNI.addDeviceDisconnectedListener(callback_disconnected)
n.OpenNI.addDeviceStateChangedListener(callback_state)

try:
    while True:
        time.sleep(0.05)
finally:
    n.OpenNI.removeDeviceConnectedListener(callback_connected)
    n.OpenNI.removeDeviceDisconnectedListener(callback_disconnected)
    n.OpenNI.removeDeviceStateChangedListener(callback_state)
    n.OpenNI.shutdown()
    print 'shutdown'
