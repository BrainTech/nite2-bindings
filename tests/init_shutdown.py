#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nite2 as n

rc = n.OpenNI.initialize()
if rc != n.OPENNI_STATUS_OK:
    raise Exception('OpenNI initialize error')

rc = n.NiTE.initialize()
if rc != n.NITE_STATUS_OK:
    raise Exception('NiTE initialize error')

print 'Wrapper version:', n.__version__
print 'Wrapper version info:', n.__version_info__

v = n.NiTE.getVersion()
print 'NiTE version:', v

v = n.OpenNI.getVersion()
print 'OpenNI version:', v

n.NiTE.shutdown()
n.OpenNI.shutdown()
# print 'shutdown'
