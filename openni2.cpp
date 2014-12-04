
#define NO_IMPORT_ARRAY
#include "boost_python.h"
#include "openni2.h"

//-----------------------------------------------------------------------------------------------

class CameraSettingsWrapper : private openni::CameraSettings
{
public:
    inline CameraSettingsWrapper(const openni::CameraSettings * s)
        : openni::CameraSettings(*s)
    {
    }

    inline openni::Status setAutoExposureEnabled(bool enabled)
    {
        return openni::CameraSettings::setAutoExposureEnabled(enabled);
    }

    inline openni::Status setAutoWhiteBalanceEnabled(bool enabled)
    {
        return openni::CameraSettings::setAutoWhiteBalanceEnabled(enabled);
    }

    inline bool getAutoExposureEnabled() const
    {
        return openni::CameraSettings::getAutoExposureEnabled();
    }

    inline bool getAutoWhiteBalanceEnabled() const
    {
        return openni::CameraSettings::getAutoWhiteBalanceEnabled();
    }

    inline openni::Status setGain(int gain)
    {
        return openni::CameraSettings::setGain(gain);
    }

    inline openni::Status setExposure(int exposure)
    {
        return openni::CameraSettings::setExposure(exposure);
    }

    inline int getGain()
    {
        return openni::CameraSettings::getGain();
    }

    inline int getExposure()
    {
        return openni::CameraSettings::getExposure();
    }

    inline bool isValid() const
    {
        return openni::CameraSettings::isValid();
    }
};

//-----------------------------------------------------------------------------------------------

VideoFrameWrapper::VideoFrameWrapper(const openni::VideoFrameRef & f)
{
    m_isValid = f.isValid();

    if(m_isValid)
    {
        m_width = f.getWidth();
        m_height = f.getHeight();
        m_sensorType = f.getSensorType();
        m_timestamp = f.getTimestamp();
        m_frameIndex = f.getFrameIndex();
        m_croppingEnabled = f.getCroppingEnabled();
        m_cropOriginX = f.getCropOriginX();
        m_cropOriginY = f.getCropOriginY();
        m_videoMode = f.getVideoMode();

        const void * data = f.getData();

        if(!data)
            return;

        const openni::PixelFormat pf = m_videoMode.getPixelFormat();
        if(pf != openni::PIXEL_FORMAT_RGB888 &&
           pf != openni::PIXEL_FORMAT_DEPTH_1_MM &&
           pf != openni::PIXEL_FORMAT_DEPTH_100_UM)
            throw std::runtime_error("only RGB888, DEPTH_1_MM and DEPTH_100_UM pixel formats are supported");

        // openni::DepthPixel, openni::RGB888Pixel

        PyObject * obj = NULL;

        if(pf == openni::PIXEL_FORMAT_RGB888)
        {
            if(f.getStrideInBytes() != 3*getWidth())
                throw std::runtime_error("strange stride value ???");
            npy_intp dims[3] = { f.getHeight(), f.getWidth(), 3 };
            obj = PyArray_SimpleNewFromData(3, dims, NPY_UBYTE, (void*)data);
        }
        else if(pf == openni::PIXEL_FORMAT_DEPTH_1_MM || pf == openni::PIXEL_FORMAT_DEPTH_100_UM)
        {
            if(f.getStrideInBytes() != int(sizeof(openni::DepthPixel))*f.getWidth())
                throw std::runtime_error("strange stride value ???");
            npy_intp dims[2] = { f.getHeight(), f.getWidth() };
            obj = PyArray_SimpleNewFromData(2, dims, NPY_USHORT, (void*)data);
        }

        if(obj)
        {
            bp::handle<> handle(obj);
            bp::numeric::array arr(handle);

            // The problem of returning arr is twofold: firstly the user can modify
            // the data which will betray the const-correctness
            // Secondly the lifetime of the data is managed by the C++ API and not the lifetime
            // of the numpy array whatsoever. But we have a simply solution..

            m_data = arr.copy(); // copy the object. numpy owns the copy now.
        }
    }
    else
    {
        m_width = 0;
        m_height = 0;
        m_sensorType = (openni::SensorType)0;
        m_timestamp = 0;
        m_frameIndex = -1;
        m_croppingEnabled = false;
        m_cropOriginX = 0;
        m_cropOriginY = 0;
    }
}

//-----------------------------------------------------------------------------------------------

openni::Status DeviceWrapper::open(const bp::object & uri)
{
    if(uri.is_none())
    {
        ScopedGILRelease release_gil;
        return openni::Device::open(openni::ANY_DEVICE);
    }
    else
    {
        const std::string uri_str = bp::extract<std::string>(uri);
        ScopedGILRelease release_gil;
        return openni::Device::open(uri_str.c_str());
    }
}

//-----------------------------------------------------------------------------------------------

class VideoStreamWrapper : private openni::VideoStream, private openni::VideoStream::NewFrameListener
{
public:
    inline VideoStreamWrapper()
    {
    }

    inline ~VideoStreamWrapper()
    {
        if(!m_frameListeners.empty())
            openni::VideoStream::removeNewFrameListener(this);
    }

    inline bool isValid() const
    {
        return openni::VideoStream::isValid();
    }

    // defined later to break circular dependency
    inline openni::Status create(const DeviceWrapper & device, openni::SensorType sensorType);

    inline void destroy()
    {
        ScopedGILRelease release_gil;
        openni::VideoStream::destroy();
    }

    inline const SensorInfoWrapper getSensorInfo() const
    {
        return SensorInfoWrapper(&openni::VideoStream::getSensorInfo());
    }

    inline openni::Status start()
    {
        ScopedGILRelease release_gil;
        return openni::VideoStream::start();
    }

    inline void stop()
    {
        ScopedGILRelease release_gil;
        openni::VideoStream::stop();
    }

    inline bp::tuple readFrame()
    {
        openni::Status status;
        openni::VideoFrameRef frame;

        {
            ScopedGILRelease release_gil;
            status = openni::VideoStream::readFrame(&frame);
        }

        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object());
        else
            return bp::make_tuple(status, VideoFrameWrapper(frame));
    }

    inline openni::Status addNewFrameListener(const bp::object & new_frame_callback)
    {
        if(addListener(m_frameListeners, new_frame_callback))
            return openni::VideoStream::addNewFrameListener(this);
        else
            return openni::STATUS_OK;
    }

    inline void removeNewFrameListener(const bp::object & new_frame_callback)
    {
        if(removeListener(m_frameListeners, new_frame_callback))
            openni::VideoStream::removeNewFrameListener(this);
    }

    inline CameraSettingsWrapper getCameraSettings()
    {
        return CameraSettingsWrapper(openni::VideoStream::getCameraSettings());
    }

    inline VideoModeWrapper getVideoMode() const
    {
        return openni::VideoStream::getVideoMode();
    }

    inline openni::Status setVideoMode(const VideoModeWrapper & videoMode)
    {
        return openni::VideoStream::setVideoMode(videoMode.getVideoModeConstRef());
    }

    inline int getMaxPixelValue() const
    {
        return openni::VideoStream::getMaxPixelValue();
    }

    inline int getMinPixelValue() const
    {
        return openni::VideoStream::getMinPixelValue();
    }

    inline bool isCroppingSupported() const
    {
        return openni::VideoStream::isCroppingSupported();
    }

    inline bp::tuple getCropping() const
    {
        int pOriginX, pOriginY, pWidth, pHeight;
        bool cropping_enabled = openni::VideoStream::getCropping(&pOriginX, &pOriginY, &pWidth, &pHeight);
        return bp::make_tuple(cropping_enabled, pOriginX, pOriginY, pWidth, pHeight);
    }

    inline openni::Status setCropping(int originX, int originY, int width, int height)
    {
        return openni::VideoStream::setCropping(originX, originY, width, height);
    }

    inline openni::Status resetCropping()
    {
        return openni::VideoStream::resetCropping();
    }

    inline bool getMirroringEnabled() const
    {
        return openni::VideoStream::getMirroringEnabled();
    }

    inline openni::Status setMirroringEnabled(bool isEnabled)
    {
        return openni::VideoStream::setMirroringEnabled(isEnabled);
    }

    inline float getHorizontalFieldOfView() const
    {
        return openni::VideoStream::getHorizontalFieldOfView();
    }

    inline float getVerticalFieldOfView() const
    {
        return openni::VideoStream::getVerticalFieldOfView();
    }

    inline bool isPropertySupported(int propertyId) const
    {
        return openni::VideoStream::isPropertySupported(propertyId);
    }

    inline bool isCommandSupported(int commandId) const
    {
        return openni::VideoStream::isCommandSupported(commandId);
    }

protected:
    friend class OpenNIWrapper;
    friend class RecorderWrapper;
    friend class PlaybackControlWrapper;
    friend class CoordinateConverterWrapper;

    inline openni::VideoStream * getVideoStreamPtr()
    {
        return &static_cast<openni::VideoStream &>(*this);
    }

    inline openni::VideoStream & getVideoStreamRef()
    {
        return static_cast<openni::VideoStream &>(*this);
    }

    inline const openni::VideoStream & getVideoStreamConstRef() const
    {
        return static_cast<const openni::VideoStream &>(*this);
    }

private:
    virtual void onNewFrame(openni::VideoStream &) override
    {
        GILStateScopedEnsure gilEnsure;
        for(auto & obj : m_frameListeners)
            obj(bp::ptr(this));
    }

    std::vector<bp::object> m_frameListeners;

    // disable copy constructor
    VideoStreamWrapper(const VideoStreamWrapper &);
};

//-----------------------------------------------------------------------------------------------

inline openni::Status VideoStreamWrapper::create(const DeviceWrapper & device, openni::SensorType sensorType)
{
    ScopedGILRelease release_gil;
    return openni::VideoStream::create(device.getDeviceConstRef(), sensorType);
}

//-----------------------------------------------------------------------------------------------

inline openni::Status PlaybackControlWrapper::seek(const VideoStreamWrapper & stream, int frameIndex)
{
    return openni::PlaybackControl::seek(stream.getVideoStreamConstRef(), frameIndex);
}

inline int PlaybackControlWrapper::getNumberOfFrames(const VideoStreamWrapper & stream) const
{
    return openni::PlaybackControl::getNumberOfFrames(stream.getVideoStreamConstRef());
}

//-----------------------------------------------------------------------------------------------

class DeviceListenerImpl : private openni::OpenNI::DeviceConnectedListener,
                           private openni::OpenNI::DeviceDisconnectedListener,
                           private openni::OpenNI::DeviceStateChangedListener
{
public:
    inline DeviceListenerImpl()
    {
    }

    inline ~DeviceListenerImpl()
    {
        if(!deviceConnectedListeners.empty())
            openni::OpenNI::removeDeviceConnectedListener(this);
        if(!deviceDisconnectedListeners.empty())
            openni::OpenNI::removeDeviceDisconnectedListener(this);
        if(!deviceStateChangedListeners.empty())
            openni::OpenNI::removeDeviceStateChangedListener(this);
    }

    inline void addDeviceConnectedListener(const bp::object & callback)
    {
        if(addListener(deviceConnectedListeners, callback))
            openni::OpenNI::addDeviceConnectedListener(this);
    }

    inline void addDeviceDisconnectedListener(const bp::object & callback)
    {
        if(addListener(deviceDisconnectedListeners, callback))
            openni::OpenNI::addDeviceDisconnectedListener(this);
    }

    inline void addDeviceStateChangedListener(const bp::object & callback)
    {
        if(addListener(deviceStateChangedListeners, callback))
            openni::OpenNI::addDeviceStateChangedListener(this);
    }

    inline void removeDeviceConnectedListener(const bp::object & callback)
    {
        if(removeListener(deviceConnectedListeners, callback))
            openni::OpenNI::removeDeviceConnectedListener(this);
    }

    inline void removeDeviceDisconnectedListener(const bp::object & callback)
    {
        if(removeListener(deviceDisconnectedListeners, callback))
            openni::OpenNI::removeDeviceDisconnectedListener(this);
    }

    inline void removeDeviceStateChangedListener(const bp::object & callback)
    {
        if(removeListener(deviceStateChangedListeners, callback))
            openni::OpenNI::removeDeviceStateChangedListener(this);
    }

private:
    std::vector<bp::object> deviceConnectedListeners;
    std::vector<bp::object> deviceDisconnectedListeners;
    std::vector<bp::object> deviceStateChangedListeners;

    virtual void onDeviceConnected(const openni::DeviceInfo * deviceInfo) override
    {
        GILStateScopedEnsure gilEnsure;
        bp::object devInfoWrapper;
        if(deviceInfo)
            devInfoWrapper = bp::object(DeviceInfoWrapper(*deviceInfo));

        for(auto & obj : deviceConnectedListeners)
        {
            try
            {
                obj(devInfoWrapper);
            }
            catch(...)
            {
            }
        }
    }

    virtual void onDeviceDisconnected(const openni::DeviceInfo * deviceInfo) override
    {
        GILStateScopedEnsure gilEnsure;
        bp::object devInfoWrapper;
        if(deviceInfo)
            devInfoWrapper = bp::object(DeviceInfoWrapper(*deviceInfo));

        for(auto & obj : deviceDisconnectedListeners)
        {
            try
            {
                obj(devInfoWrapper);
            }
            catch(...)
            {
            }
        }
    }

    virtual void onDeviceStateChanged(const openni::DeviceInfo * deviceInfo, openni::DeviceState deviceState) override
    {
        GILStateScopedEnsure gilEnsure;
        bp::object devInfoWrapper;
        if(deviceInfo)
            devInfoWrapper = bp::object(DeviceInfoWrapper(*deviceInfo));

        for(auto & obj : deviceStateChangedListeners)
        {
            try
            {
                obj(devInfoWrapper, deviceState);
            }
            catch(...)
            {
            }
        }
    }
};

class OpenNIWrapper
{
private:
    OpenNIWrapper();

public:
    static inline openni::Status initialize()
    {
        ScopedGILRelease release_gil;
        return openni::OpenNI::initialize();
    }

    static inline void shutdown()
    {
        ScopedGILRelease release_gil;
        openni::OpenNI::shutdown();
    }

    static inline const VersionWrapper<openni::Version> getVersion()
    {
        return openni::OpenNI::getVersion();
    }

    static inline const std::string getExtendedError()
    {
        return openni::OpenNI::getExtendedError();
    }

    static bp::list enumerateDevices()
    {
        openni::Array<openni::DeviceInfo> deviceInfoList;

        {
            ScopedGILRelease release_gil;
            openni::OpenNI::enumerateDevices(&deviceInfoList);
        }

        bp::list deviceList;
        for(int i = 0; i < deviceInfoList.getSize(); i++)
            deviceList.append(DeviceInfoWrapper(deviceInfoList[i]));
        return deviceList;
    }

    static bp::tuple waitForAnyStream(const bp::list & streams, int timeout = openni::TIMEOUT_FOREVER)
    {
        const bp::ssize_t n = bp::len(streams);
        std::vector<openni::VideoStream*> streams_vec(n, nullptr);
        for(bp::ssize_t i = 0; i < n; i++)
            streams_vec[n] = bp::extract<VideoStreamWrapper&>(streams[i])().getVideoStreamPtr();

        openni::Status status;
        int pReadyStreamIndex;

        {
            ScopedGILRelease release_gil;
            status = openni::OpenNI::waitForAnyStream(streams_vec.data(), streams_vec.size(), &pReadyStreamIndex, timeout);
        }

        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object());
        else
            return bp::make_tuple(status, pReadyStreamIndex);
    }

    inline static openni::Status setLogOutputFolder(const std::string & strLogOutputFolder)
    {
        return openni::OpenNI::setLogOutputFolder(strLogOutputFolder.c_str());
    }

    static bp::tuple getLogFileName()
    {
        const unsigned int buf_size = 1024;
        char buf[buf_size];
        openni::Status status = openni::OpenNI::getLogFileName(buf, buf_size-1);
        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object());
        else
            return bp::make_tuple(status, std::string(buf));
    }

    inline static openni::Status setLogMinSeverity(int minSeverity)
    {
        return openni::OpenNI::setLogMinSeverity(minSeverity);
    }

    inline static openni::Status setLogConsoleOutput(bool consoleOutput)
    {
        return openni::OpenNI::setLogConsoleOutput(consoleOutput);
    }

    inline static openni::Status setLogFileOutput(bool fileOutput)
    {
        return openni::OpenNI::setLogFileOutput(fileOutput);
    }

    inline static void addDeviceConnectedListener(const bp::object & callback)
    {
        m_listener.addDeviceConnectedListener(callback);
    }

    inline static void addDeviceDisconnectedListener(const bp::object & callback)
    {
        m_listener.addDeviceDisconnectedListener(callback);
    }

    inline static void addDeviceStateChangedListener(const bp::object & callback)
    {
        m_listener.addDeviceStateChangedListener(callback);
    }

    inline static void removeDeviceConnectedListener(const bp::object & callback)
    {
        m_listener.removeDeviceConnectedListener(callback);
    }

    inline static void removeDeviceDisconnectedListener(const bp::object & callback)
    {
        m_listener.removeDeviceDisconnectedListener(callback);
    }

    inline static void removeDeviceStateChangedListener(const bp::object & callback)
    {
        m_listener.removeDeviceStateChangedListener(callback);
    }

private:
    static DeviceListenerImpl m_listener;
};

DeviceListenerImpl OpenNIWrapper::m_listener;

//-----------------------------------------------------------------------------------------------

class CoordinateConverterWrapper : private openni::CoordinateConverter
{
public:
    static bp::tuple convertWorldToDepth_i(const VideoStreamWrapper & depthStream,
                                           float worldX, float worldY, float worldZ)
    {
        int pDepthX, pDepthY;
        openni::DepthPixel pDepthZ;
        openni::Status status =
            openni::CoordinateConverter::convertWorldToDepth(
                depthStream.getVideoStreamConstRef(),
                worldX, worldY, worldZ,
                &pDepthX, &pDepthY, &pDepthZ);
        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object(), bp::object(), bp::object());
        else
            return bp::make_tuple(status, pDepthX, pDepthY, pDepthZ);
    }

    static bp::tuple convertWorldToDepth_f(const VideoStreamWrapper & depthStream,
                                           float worldX, float worldY, float worldZ)
    {
        float pDepthX, pDepthY, pDepthZ;
        openni::Status status =
            openni::CoordinateConverter::convertWorldToDepth(
                depthStream.getVideoStreamConstRef(),
                worldX, worldY, worldZ,
                &pDepthX, &pDepthY, &pDepthZ);
        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object(), bp::object(), bp::object());
        else
            return bp::make_tuple(status, pDepthX, pDepthY, pDepthZ);
    }

    static bp::tuple convertDepthToWorld_i(const VideoStreamWrapper & depthStream,
                                           int depthX, int depthY, openni::DepthPixel depthZ)
    {
        float pWorldX, pWorldY, pWorldZ;
        openni::Status status =
            openni::CoordinateConverter::convertDepthToWorld(
                depthStream.getVideoStreamConstRef(),
                depthX, depthY, depthZ,
                &pWorldX, &pWorldY, &pWorldZ);
        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object(), bp::object(), bp::object());
        else
            return bp::make_tuple(status, pWorldX, pWorldY, pWorldZ);
    }

    static bp::tuple convertDepthToWorld_f(const VideoStreamWrapper & depthStream,
                                           float depthX, float depthY, float depthZ)
    {
        float pWorldX, pWorldY, pWorldZ;
        openni::Status status =
            openni::CoordinateConverter::convertDepthToWorld(
                depthStream.getVideoStreamConstRef(),
                depthX, depthY, depthZ,
                &pWorldX, &pWorldY, &pWorldZ);
        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object(), bp::object(), bp::object());
        else
            return bp::make_tuple(status, pWorldX, pWorldY, pWorldZ);
    }

    static bp::tuple convertDepthToColor(const VideoStreamWrapper & depthStream,
                                         const VideoStreamWrapper & colorStream,
                                         int depthX, int depthY, openni::DepthPixel depthZ)
    {
        int pColorX, pColorY;
        openni::Status status =
            openni::CoordinateConverter::convertDepthToColor(
                depthStream.getVideoStreamConstRef(),
                colorStream.getVideoStreamConstRef(),
                depthX, depthY, depthZ,
                &pColorX, &pColorY);
        if(status != openni::STATUS_OK)
            return bp::make_tuple(status, bp::object(), bp::object());
        else
            return bp::make_tuple(status, pColorX, pColorY);
    }

private:
    CoordinateConverterWrapper();
};

//-----------------------------------------------------------------------------------------------

class RecorderWrapper : private openni::Recorder
{
public:
    inline openni::Status create(const std::string & fileName)
    {
        ScopedGILRelease release_gil;
        return openni::Recorder::create(fileName.c_str());
    }

    inline bool isValid() const
    {
        return openni::Recorder::isValid();
    }

    inline openni::Status attach(VideoStreamWrapper & stream, bool allowLossyCompression = false)
    {
        ScopedGILRelease release_gil;
        return openni::Recorder::attach(stream.getVideoStreamRef(), allowLossyCompression);
    }

    inline openni::Status start()
    {
        ScopedGILRelease release_gil;
        return openni::Recorder::start();
    }

    inline void stop()
    {
        ScopedGILRelease release_gil;
        openni::Recorder::stop();
    }

    inline void destroy()
    {
        ScopedGILRelease release_gil;
        openni::Recorder::destroy();
    }
};

//-----------------------------------------------------------------------------------------------

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(DeviceWrapper_open_overloads,
                                       DeviceWrapper::open,
                                       0, 1);

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(RecorderWrapper_attach_overloads,
                                       RecorderWrapper::attach,
                                       1, 2);

BOOST_PYTHON_FUNCTION_OVERLOADS(OpenNIWrapper_waitForAnyStream_overloads,
                                OpenNIWrapper::waitForAnyStream,
                                1, 2);

//-----------------------------------------------------------------------------------------------

void export_openni2()
{
    using namespace boost::python;

    scope().attr("TIMEOUT_NONE")    = int(openni::TIMEOUT_NONE);
    scope().attr("TIMEOUT_FOREVER") = int(openni::TIMEOUT_FOREVER);

    scope().attr("DEVICE_PROPERTY_FIRMWARE_VERSION")        = int(openni::DEVICE_PROPERTY_FIRMWARE_VERSION);
    scope().attr("DEVICE_PROPERTY_DRIVER_VERSION")          = int(openni::DEVICE_PROPERTY_DRIVER_VERSION);
    scope().attr("DEVICE_PROPERTY_HARDWARE_VERSION")        = int(openni::DEVICE_PROPERTY_HARDWARE_VERSION);
    scope().attr("DEVICE_PROPERTY_SERIAL_NUMBER")           = int(openni::DEVICE_PROPERTY_SERIAL_NUMBER);
    scope().attr("DEVICE_PROPERTY_ERROR_STATE")             = int(openni::DEVICE_PROPERTY_ERROR_STATE);
    scope().attr("DEVICE_PROPERTY_IMAGE_REGISTRATION")      = int(openni::DEVICE_PROPERTY_IMAGE_REGISTRATION);
    scope().attr("DEVICE_PROPERTY_PLAYBACK_SPEED")          = int(openni::DEVICE_PROPERTY_PLAYBACK_SPEED);
    scope().attr("DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED") = int(openni::DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED);

    scope().attr("STREAM_PROPERTY_CROPPING")           = int(openni::STREAM_PROPERTY_CROPPING);
    scope().attr("STREAM_PROPERTY_HORIZONTAL_FOV")     = int(openni::STREAM_PROPERTY_HORIZONTAL_FOV);
    scope().attr("STREAM_PROPERTY_VERTICAL_FOV")       = int(openni::STREAM_PROPERTY_VERTICAL_FOV);
    scope().attr("STREAM_PROPERTY_VIDEO_MODE")         = int(openni::STREAM_PROPERTY_VIDEO_MODE);
    scope().attr("STREAM_PROPERTY_MAX_VALUE")          = int(openni::STREAM_PROPERTY_MAX_VALUE);
    scope().attr("STREAM_PROPERTY_MIN_VALUE")          = int(openni::STREAM_PROPERTY_MIN_VALUE);
    scope().attr("STREAM_PROPERTY_STRIDE")             = int(openni::STREAM_PROPERTY_STRIDE);
    scope().attr("STREAM_PROPERTY_MIRRORING")          = int(openni::STREAM_PROPERTY_MIRRORING);
    scope().attr("STREAM_PROPERTY_NUMBER_OF_FRAMES")   = int(openni::STREAM_PROPERTY_NUMBER_OF_FRAMES);
    scope().attr("STREAM_PROPERTY_AUTO_WHITE_BALANCE") = int(openni::STREAM_PROPERTY_AUTO_WHITE_BALANCE);
    scope().attr("STREAM_PROPERTY_AUTO_EXPOSURE")      = int(openni::STREAM_PROPERTY_AUTO_EXPOSURE);
    scope().attr("STREAM_PROPERTY_EXPOSURE")           = int(openni::STREAM_PROPERTY_EXPOSURE);
    scope().attr("STREAM_PROPERTY_GAIN")               = int(openni::STREAM_PROPERTY_GAIN);

    scope().attr("DEVICE_COMMAND_SEEK") = int(openni::DEVICE_COMMAND_SEEK);

    //----------------------------------------------------------------------------------------------

    enum_<openni::Status>("OpenNiStatus", "Possible failure values.")
        .value("OPENNI_STATUS_OK",              openni::STATUS_OK)
        .value("OPENNI_STATUS_ERROR",           openni::STATUS_ERROR)
        .value("OPENNI_STATUS_NOT_IMPLEMENTED", openni::STATUS_NOT_IMPLEMENTED)
        .value("OPENNI_STATUS_NOT_SUPPORTED",   openni::STATUS_NOT_SUPPORTED)
        .value("OPENNI_STATUS_BAD_PARAMETER",   openni::STATUS_BAD_PARAMETER)
        .value("OPENNI_STATUS_OUT_OF_FLOW",     openni::STATUS_OUT_OF_FLOW)
        .value("OPENNI_STATUS_NO_DEVICE",       openni::STATUS_NO_DEVICE)
        .value("OPENNI_STATUS_TIME_OUT",        openni::STATUS_TIME_OUT)
        .export_values();
        ;

    enum_<openni::SensorType>("SensorType", "The source of the stream.")
        .value("SENSOR_IR",    openni::SENSOR_IR)
        .value("SENSOR_COLOR", openni::SENSOR_COLOR)
        .value("SENSOR_DEPTH", openni::SENSOR_DEPTH)
        .export_values();
        ;

    enum_<openni::PixelFormat>("PixelFormat", "All available formats of the output of a stream.")
        .value("PIXEL_FORMAT_DEPTH_1_MM",   openni::PIXEL_FORMAT_DEPTH_1_MM)
        .value("PIXEL_FORMAT_DEPTH_100_UM", openni::PIXEL_FORMAT_DEPTH_100_UM)
        .value("PIXEL_FORMAT_SHIFT_9_2",    openni::PIXEL_FORMAT_SHIFT_9_2)
        .value("PIXEL_FORMAT_SHIFT_9_3",    openni::PIXEL_FORMAT_SHIFT_9_3)
        .value("PIXEL_FORMAT_RGB888",       openni::PIXEL_FORMAT_RGB888)
        .value("PIXEL_FORMAT_YUV422",       openni::PIXEL_FORMAT_YUV422)
        .value("PIXEL_FORMAT_GRAY8",        openni::PIXEL_FORMAT_GRAY8)
        .value("PIXEL_FORMAT_GRAY16",       openni::PIXEL_FORMAT_GRAY16)
        .value("PIXEL_FORMAT_JPEG",         openni::PIXEL_FORMAT_JPEG)
        .value("PIXEL_FORMAT_YUYV",         openni::PIXEL_FORMAT_YUYV)
        .export_values()
        ;

    enum_<openni::DeviceState>("DeviceState")
        .value("DEVICE_STATE_OK",        openni::DEVICE_STATE_OK)
        .value("DEVICE_STATE_ERROR",     openni::DEVICE_STATE_ERROR)
        .value("DEVICE_STATE_NOT_READY", openni::DEVICE_STATE_NOT_READY)
        .value("DEVICE_STATE_EOF",       openni::DEVICE_STATE_EOF)
        .export_values()
        ;

    enum_<openni::ImageRegistrationMode>("ImageRegistrationMode", "Image registration is used to properly superimpose two images from cameras located at different points in space.")
        .value("IMAGE_REGISTRATION_OFF",            openni::IMAGE_REGISTRATION_OFF)
        .value("IMAGE_REGISTRATION_DEPTH_TO_COLOR", openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        .export_values()
        ;

    //----------------------------------------------------------------------------------------------

    class_<VersionWrapper<openni::Version>>("OpenNiVersion", "Holds an OpenNI version number, which consists of four separate numbers in the format: major.minor.maintenance.build. For example: 2.0.0.20.", no_init)
        .def(self_ns::str(self_ns::self))
        .add_property("major",       &VersionWrapper<openni::Version>::getMajor)
        .add_property("minor",       &VersionWrapper<openni::Version>::getMinor)
        .add_property("maintenance", &VersionWrapper<openni::Version>::getMaintenance)
        .add_property("build",       &VersionWrapper<openni::Version>::getBuild)
        ;

    class_<SensorInfoWrapper>("SensorInfo", no_init)
        .add_property("sensorType",          &SensorInfoWrapper::getSensorType)
        .add_property("supportedVideoModes", &SensorInfoWrapper::getSupportedVideoModes)
        ;

    class_<VideoStreamWrapper, boost::noncopyable>("VideoStream")
        .def("isValid",                     &VideoStreamWrapper::isValid)
        .def("create",                      &VideoStreamWrapper::create)
        .def("destroy",                     &VideoStreamWrapper::destroy)
        .def("getSensorInfo",               &VideoStreamWrapper::getSensorInfo)
        .def("start",                       &VideoStreamWrapper::start)
        .def("stop",                        &VideoStreamWrapper::stop)
        .def("readFrame",                   &VideoStreamWrapper::readFrame)
        .def("addNewFrameListener",         &VideoStreamWrapper::addNewFrameListener)
        .def("removeNewFrameListener",      &VideoStreamWrapper::removeNewFrameListener)
        .def("getCameraSettings",           &VideoStreamWrapper::getCameraSettings)
        .def("getVideoMode",                &VideoStreamWrapper::getVideoMode)
        .def("setVideoMode",                &VideoStreamWrapper::setVideoMode)
        .def("getMaxPixelValue",            &VideoStreamWrapper::getMaxPixelValue)
        .def("getMinPixelValue",            &VideoStreamWrapper::getMinPixelValue)
        .def("isCroppingSupported",         &VideoStreamWrapper::isCroppingSupported)
        .def("getCropping",                 &VideoStreamWrapper::getCropping)
        .def("setCropping",                 &VideoStreamWrapper::setCropping)
        .def("resetCropping",               &VideoStreamWrapper::resetCropping)
        .def("getMirroringEnabled",         &VideoStreamWrapper::getMirroringEnabled)
        .def("setMirroringEnabled",         &VideoStreamWrapper::setMirroringEnabled)
        .def("getHorizontalFieldOfView",    &VideoStreamWrapper::getHorizontalFieldOfView)
        .def("getVerticalFieldOfView",      &VideoStreamWrapper::getVerticalFieldOfView)
        .def("isPropertySupported",         &VideoStreamWrapper::isPropertySupported)
        .def("isCommandSupported",          &VideoStreamWrapper::isCommandSupported)
        //.def("getProperty",                 &VideoStreamWrapper::getProperty)
        //.def("setProperty",                 &VideoStreamWrapper::setProperty)
        //.def("invoke",                      &VideoStreamWrapper::invoke)
        ;

    class_<VideoFrameWrapper>("VideoFrame", no_init)
        .def("isValid",                   &VideoFrameWrapper::isValid)
        .add_property("data",             &VideoFrameWrapper::getData)
        .add_property("sensorType",       &VideoFrameWrapper::getSensorType)
        .add_property("videoMode",        &VideoFrameWrapper::getVideoMode)
        .add_property("timestamp",        &VideoFrameWrapper::getTimestamp)
        .add_property("frameIndex",       &VideoFrameWrapper::getFrameIndex)
        .add_property("croppingEnabled",  &VideoFrameWrapper::getCroppingEnabled)
        .add_property("cropOriginX",      &VideoFrameWrapper::getCropOriginX)
        .add_property("cropOriginY",      &VideoFrameWrapper::getCropOriginY)
        ;

    class_<DeviceWrapper, boost::noncopyable>("Device")
        .def("open",                             &DeviceWrapper::open, DeviceWrapper_open_overloads())
        .def("close",                            &DeviceWrapper::close)
        .def("getDeviceInfo",                    &DeviceWrapper::getDeviceInfo)
        .def("hasSensor",                        &DeviceWrapper::hasSensor)
        .def("getSensorInfo",                    &DeviceWrapper::getSensorInfo)
        .def("getPlaybackControl",               &DeviceWrapper::getPlaybackControl)
        .def("isImageRegistrationModeSupported", &DeviceWrapper::isImageRegistrationModeSupported)
        .def("getImageRegistrationMode",         &DeviceWrapper::getImageRegistrationMode)
        .def("setImageRegistrationMode",         &DeviceWrapper::setImageRegistrationMode)
        .def("isValid",                          &DeviceWrapper::isValid)
        .def("isFile",                           &DeviceWrapper::isFile)
        .def("setDepthColorSyncEnabled",         &DeviceWrapper::setDepthColorSyncEnabled)
        .def("getDepthColorSyncEnabled",         &DeviceWrapper::getDepthColorSyncEnabled)
        .def("isPropertySupported",              &DeviceWrapper::isPropertySupported)
        .def("isCommandSupported",               &DeviceWrapper::isCommandSupported)
        //.def("getProperty",                      &DeviceWrapper::getProperty)
        //.def("setProperty",                      &DeviceWrapper::setProperty)
        //.def("invoke",                           &DeviceWrapper::invoke)

        // Kinect specific functions
        .def("isNearModeSupported",              &DeviceWrapper::isNearModeSupported)
        .def("isCameraElevationSupported",       &DeviceWrapper::isCameraElevationSupported)
        .def("isAccelerometerSupported",         &DeviceWrapper::isAccelerometerSupported)
        .def("setNearMode",                      &DeviceWrapper::setNearMode)
        .def("getNearMode",                      &DeviceWrapper::getNearMode)
        .def("setCameraElevation",               &DeviceWrapper::setCameraElevation)
        .def("getCameraElevation",               &DeviceWrapper::getCameraElevation)
        .def("getAccelerometerReading",          &DeviceWrapper::getAccelerometerReading)
        ;

    class_<CameraSettingsWrapper>("CameraSettings", no_init)
        .def("setAutoExposureEnabled",          &CameraSettingsWrapper::setAutoExposureEnabled)
        .def("setAutoWhiteBalanceEnabled",      &CameraSettingsWrapper::setAutoWhiteBalanceEnabled)
        .def("getAutoExposureEnabled",          &CameraSettingsWrapper::getAutoExposureEnabled)
        .def("getAutoWhiteBalanceEnabled",      &CameraSettingsWrapper::getAutoWhiteBalanceEnabled)
        .def("setGain",                         &CameraSettingsWrapper::setGain)
        .def("setExposure",                     &CameraSettingsWrapper::setExposure)
        .def("getGain",                         &CameraSettingsWrapper::getGain)
        .def("getExposure",                     &CameraSettingsWrapper::getExposure)
        .def("isValid",                         &CameraSettingsWrapper::isValid)
        ;

    class_<PlaybackControlWrapper>("PlaybackControl", no_init)
        .def("getSpeed",            &PlaybackControlWrapper::getSpeed)
        .def("setSpeed",            &PlaybackControlWrapper::setSpeed)
        .def("getRepeatEnabled",    &PlaybackControlWrapper::getRepeatEnabled)
        .def("setRepeatEnabled",    &PlaybackControlWrapper::setRepeatEnabled)
        .def("seek",                &PlaybackControlWrapper::seek)
        .def("getNumberOfFrames",   &PlaybackControlWrapper::getNumberOfFrames)
        .def("isValid",             &PlaybackControlWrapper::isValid)
        ;

    class_<VideoModeWrapper>("VideoMode")
        .def("getPixelFormat", &VideoModeWrapper::getPixelFormat)
        .def("setPixelFormat", &VideoModeWrapper::setPixelFormat)
        .def("getResolution",  &VideoModeWrapper::getResolution)
        .def("setResolution",  &VideoModeWrapper::setResolution)
        .def("getFps",         &VideoModeWrapper::getFps)
        .def("setFps",         &VideoModeWrapper::setFps)
        ;

    class_<DeviceInfoWrapper>("DeviceInfo", no_init)
        .add_property("uri",          &DeviceInfoWrapper::getUri)
        .add_property("vendor",       &DeviceInfoWrapper::getVendor)
        .add_property("name",         &DeviceInfoWrapper::getName)
        .add_property("usbVendorId",  &DeviceInfoWrapper::getUsbVendorId)
        .add_property("usbProductId", &DeviceInfoWrapper::getUsbProductId)
        ;

    class_<CoordinateConverterWrapper, boost::noncopyable>("CoordinateConverter", no_init)
		.def("convertDepthToColor",   &CoordinateConverterWrapper::convertDepthToColor)  .staticmethod("convertDepthToColor")
        .def("convertDepthToWorld_i", &CoordinateConverterWrapper::convertDepthToWorld_i).staticmethod("convertDepthToWorld_i")
		.def("convertDepthToWorld_f", &CoordinateConverterWrapper::convertDepthToWorld_f).staticmethod("convertDepthToWorld_f")
		.def("convertWorldToDepth_i", &CoordinateConverterWrapper::convertWorldToDepth_i).staticmethod("convertWorldToDepth_i")
		.def("convertWorldToDepth_f", &CoordinateConverterWrapper::convertWorldToDepth_f).staticmethod("convertWorldToDepth_f")
		;

    class_<RecorderWrapper, boost::noncopyable>("Recorder")
        .def("create",  &RecorderWrapper::create)
        .def("isValid", &RecorderWrapper::isValid)
        .def("attach",  &RecorderWrapper::attach, RecorderWrapper_attach_overloads())
        .def("start",   &RecorderWrapper::start)
        .def("stop",    &RecorderWrapper::stop)
        .def("destroy", &RecorderWrapper::destroy)
        ;

    //----------------------------------------------------------------------------------------------

    class_<OpenNIWrapper, boost::noncopyable>("OpenNI", "The OpenNI class is a static entry point to the OpenNI library", no_init)
        .def("initialize",                       &OpenNIWrapper::initialize)                      .staticmethod("initialize")
        .def("shutdown",                         &OpenNIWrapper::shutdown)                        .staticmethod("shutdown")
        .def("getVersion",                       &OpenNIWrapper::getVersion)                      .staticmethod("getVersion")
        .def("getExtendedError",                 &OpenNIWrapper::getExtendedError)                .staticmethod("getExtendedError")
        .def("enumerateDevices",                 &OpenNIWrapper::enumerateDevices)                .staticmethod("enumerateDevices")
        .def("addDeviceConnectedListener",       &OpenNIWrapper::addDeviceConnectedListener)      .staticmethod("addDeviceConnectedListener")
        .def("addDeviceDisconnectedListener",    &OpenNIWrapper::addDeviceDisconnectedListener)   .staticmethod("addDeviceDisconnectedListener")
        .def("addDeviceStateChangedListener",    &OpenNIWrapper::addDeviceStateChangedListener)   .staticmethod("addDeviceStateChangedListener")
        .def("removeDeviceConnectedListener",    &OpenNIWrapper::removeDeviceConnectedListener)   .staticmethod("removeDeviceConnectedListener")
        .def("removeDeviceDisconnectedListener", &OpenNIWrapper::removeDeviceDisconnectedListener).staticmethod("removeDeviceDisconnectedListener")
        .def("removeDeviceStateChangedListener", &OpenNIWrapper::removeDeviceStateChangedListener).staticmethod("removeDeviceStateChangedListener")
        .def("setLogOutputFolder",               &OpenNIWrapper::setLogOutputFolder)              .staticmethod("setLogOutputFolder")
        .def("getLogFileName",                   &OpenNIWrapper::getLogFileName)                  .staticmethod("getLogFileName")
        .def("setLogMinSeverity",                &OpenNIWrapper::setLogMinSeverity)               .staticmethod("setLogMinSeverity")
        .def("setLogConsoleOutput",              &OpenNIWrapper::setLogConsoleOutput)             .staticmethod("setLogConsoleOutput")
        .def("setLogFileOutput",                 &OpenNIWrapper::setLogFileOutput)                .staticmethod("setLogFileOutput")
        .def("waitForAnyStream",                 &OpenNIWrapper::waitForAnyStream, OpenNIWrapper_waitForAnyStream_overloads()).staticmethod("waitForAnyStream")
        ;

}
