#ifndef OPENNI2_H
#define OPENNI2_H

#include "boost_python.h"
#include "enhanced_kinect/KinectProperty.h"

#include <OpenNI.h>

#include "wrapper_utils.h"

class VideoModeWrapper : private openni::VideoMode
{
public:
    inline VideoModeWrapper()
    {
        openni::VideoMode::setFps(0);
        openni::VideoMode::setResolution(0, 0);
        openni::VideoMode::setPixelFormat((openni::PixelFormat)0);
    }

    inline VideoModeWrapper(const VideoModeWrapper & p)
        : openni::VideoMode(p.getVideoModeConstRef())
    {
    }

    inline VideoModeWrapper(const openni::VideoMode & p)
        : openni::VideoMode(p)
    {
    }

    inline bp::tuple getResolution() const
    {
        return bp::make_tuple(openni::VideoMode::getResolutionX(),
                              openni::VideoMode::getResolutionY());
    }

    inline void setResolution(int x, int y)
    {
        openni::VideoMode::setResolution(x, y);
    }

    inline int getFps() const
    {
        return openni::VideoMode::getFps();
    }

    inline void setFps(int fps)
    {
        openni::VideoMode::setFps(fps);
    }

    inline openni::PixelFormat getPixelFormat() const
    {
        return openni::VideoMode::getPixelFormat();
    }

    inline void setPixelFormat(openni::PixelFormat format)
    {
        openni::VideoMode::setPixelFormat(format);
    }

protected:
    friend class VideoStreamWrapper;

    inline const openni::VideoMode & getVideoModeConstRef() const
    {
        return static_cast<const openni::VideoMode &>(*this);
    }
};

class VideoFrameWrapper
{
public:
    VideoFrameWrapper(const openni::VideoFrameRef & f = openni::VideoFrameRef());

    inline const bp::object getData() const
    {
        return m_data;
    }

    inline const VideoModeWrapper getVideoMode() const
    {
        return m_videoMode;
    }

    inline bool isValid() const
    {
        return m_isValid;
    }

    inline int getWidth() const
    {
        return m_width;
    }

    inline int getHeight() const
    {
        return m_height;
    }

    inline openni::SensorType getSensorType() const
    {
        return m_sensorType;
    }

    inline uint64_t getTimestamp() const
    {
        return m_timestamp;
    }

    inline int getFrameIndex() const
    {
        return m_frameIndex;
    }

    inline bool getCroppingEnabled() const
    {
        return m_croppingEnabled;
    }

    inline int getCropOriginX() const
    {
        return m_cropOriginX;
    }

    inline int getCropOriginY() const
    {
        return m_cropOriginY;
    }

private:
    bool m_isValid;
    bp::object m_data;
    VideoModeWrapper m_videoMode;
    int m_width;
    int m_height;
    openni::SensorType m_sensorType;
    uint64_t m_timestamp;
    int m_frameIndex;
    bool m_croppingEnabled;
    int m_cropOriginX;
    int m_cropOriginY;
};

class DeviceInfoWrapper
{
public:
    inline DeviceInfoWrapper(const openni::DeviceInfo & i)
        : m_uri(i.getUri())
        , m_vendor(i.getVendor())
        , m_name(i.getName())
        , m_usbVendorId(i.getUsbVendorId())
        , m_usbProductId(i.getUsbProductId())
    {
    }

    inline std::string getUri() const
    {
        return m_uri;
    }

    inline std::string getVendor() const
    {
        return m_vendor;
    }

    inline std::string getName() const
    {
        return m_name;
    }

    inline uint16_t getUsbVendorId() const
    {
        return m_usbVendorId;
    }

    inline uint16_t getUsbProductId() const
    {
        return m_usbProductId;
    }

private:
    const std::string m_uri;
    const std::string m_vendor;
    const std::string m_name;
    const uint16_t m_usbVendorId;
    const uint16_t m_usbProductId;
};

class SensorInfoWrapper
{
public:
    inline SensorInfoWrapper(const openni::SensorInfo * s)
    {
        if(s)
        {
            m_type = s->getSensorType();
            const auto & videoModes = s->getSupportedVideoModes();
            for(int i = 0; i < videoModes.getSize(); i++)
                m_videoModes.append(VideoModeWrapper(videoModes[i]));
        }
        else
            m_type = (openni::SensorType)-1;
    }

    inline openni::SensorType getSensorType() const
    {
        return m_type;
    }

    inline bp::list getSupportedVideoModes() const
    {
        return m_videoModes;
    }

private:
    openni::SensorType m_type;
    bp::list m_videoModes;
};

class VideoStreamWrapper;

class PlaybackControlWrapper : private openni::PlaybackControl
{
public:
    inline PlaybackControlWrapper(openni::PlaybackControl * p)
        : openni::PlaybackControl(*p)
    {
    }

    inline bool isValid() const
    {
        return openni::PlaybackControl::isValid();
    }

    inline float getSpeed() const
    {
        return openni::PlaybackControl::getSpeed();
    }

    inline openni::Status setSpeed(float speed)
    {
        return openni::PlaybackControl::setSpeed(speed);
    }

    inline bool getRepeatEnabled() const
    {
        return openni::PlaybackControl::getRepeatEnabled();
    }

    inline openni::Status setRepeatEnabled(bool repeat)
    {
        return openni::PlaybackControl::setRepeatEnabled(repeat);
    }

    openni::Status seek(const VideoStreamWrapper & stream, int frameIndex);

    int getNumberOfFrames(const VideoStreamWrapper & stream) const;
};

class DeviceWrapper : private openni::Device
{
public:
    inline DeviceWrapper()
    {
    }

    openni::Status open(const bp::object & uri = bp::object());

    inline void close()
    {
        ScopedGILRelease release_gil;
        openni::Device::close();
    }

    inline const DeviceInfoWrapper getDeviceInfo() const
    {
        return DeviceInfoWrapper(openni::Device::getDeviceInfo());
    }

    inline bool hasSensor(openni::SensorType sensorType)
    {
        return openni::Device::hasSensor(sensorType);
    }

    inline const bp::object getSensorInfo(openni::SensorType sensorType)
    {
        const openni::SensorInfo * si = openni::Device::getSensorInfo(sensorType);
        if(si)
            return bp::object(SensorInfoWrapper(si));
        else
            return bp::object();
    }

    inline bp::object getPlaybackControl()
    {
        openni::PlaybackControl * pc = openni::Device::getPlaybackControl();
        if(pc)
            return bp::object(PlaybackControlWrapper(pc));
        else
            return bp::object();
    }

    inline bool isImageRegistrationModeSupported(openni::ImageRegistrationMode mode) const
    {
        return openni::Device::isImageRegistrationModeSupported(mode);
    }

    inline openni::ImageRegistrationMode getImageRegistrationMode() const
    {
        return openni::Device::getImageRegistrationMode();
    }

    inline openni::Status setImageRegistrationMode(openni::ImageRegistrationMode mode)
    {
        return openni::Device::setImageRegistrationMode(mode);
    }

    inline bool isValid() const
    {
        return openni::Device::isValid();
    }

    inline bool isFile() const
    {
        return openni::Device::isFile();
    }

    inline openni::Status setDepthColorSyncEnabled(bool isEnabled)
    {
        return openni::Device::setDepthColorSyncEnabled(isEnabled);
    }

    inline bool getDepthColorSyncEnabled()
    {
        return openni::Device::getDepthColorSyncEnabled();
    }

    inline bool isCommandSupported(int commandId) const
    {
        return openni::Device::isCommandSupported(commandId);
    }

    inline bool isPropertySupported(int propertyId) const
    {
        return openni::Device::isPropertySupported(propertyId);
    }

public: // enhanced Kinect support
    inline bool isNearModeSupported() const
    {
        return openni::Device::isPropertySupported(KINECT_DEPTH_PROPERTY_NEAR_MODE);
    }

    inline bool isCameraElevationSupported() const
    {
        return openni::Device::isPropertySupported(KINECT_DEVICE_PROPERTY_CAMERA_ELEVATION);
    }

    inline bool isAccelerometerSupported() const
    {
        return openni::Device::isPropertySupported(KINECT_DEVICE_PROPERTY_ACCELEROMETER);
    }

    inline openni::Status setNearMode(bool nearMode)
    {
		ScopedGILRelease release_gil;
		return openni::Device::setProperty<bool>(KINECT_DEPTH_PROPERTY_NEAR_MODE, nearMode);
    }

    inline bp::object getNearMode()
    {
        bool data;
		openni::Status rc;

		{
			ScopedGILRelease release_gil;
			rc = openni::Device::getProperty<bool>(KINECT_DEPTH_PROPERTY_NEAR_MODE , &data);
		}
		
		if(rc != openni::STATUS_OK)
            return bp::object();
        else
            return bp::object(data);
    }

    inline openni::Status setCameraElevation(long angle)
    {
		ScopedGILRelease release_gil;
        return openni::Device::setProperty<long>(KINECT_DEVICE_PROPERTY_CAMERA_ELEVATION, angle);
    }

    inline bp::object getCameraElevation()
    {
        long data;
		openni::Status rc;

		{
			ScopedGILRelease release_gil;
	        rc = openni::Device::getProperty<long>(KINECT_DEVICE_PROPERTY_CAMERA_ELEVATION, &data);
		}
		
		if(rc != openni::STATUS_OK)
            return bp::object();
        else
            return bp::object(data);
    }

    // The accelerometer reading is returned as a 3-D vector pointing
    // in the direction of gravity (the floor on a non-accelerating
    // sensor). This 3-D vector is returned as a Vector4 (x, y, z, w)
    // with the w value always set to 0.0. The coordinate system is
    // centered on the sensor, and is a right-handed coordinate system
    // with the positive z in the direction the sensor is pointing at.
    // The vector is in gravity units (g), or 9.81m/s^2. The default
    // sensor rotation (horizontal, level placement) is represented by
    // the (x, y, z, w) vector whose value is (0, -1.0, 0, 0).
    inline bp::object getAccelerometerReading()
    {
        KVector4 data;
		openni::Status rc;

		{
			ScopedGILRelease release_gil;
			rc = openni::Device::getProperty<KVector4>(KINECT_DEVICE_PROPERTY_ACCELEROMETER, &data);
		}
			
		if(rc != openni::STATUS_OK)
            return bp::object();
        else
            return bp::make_tuple(data.x, data.y, data.z);
    }

protected:
    friend class VideoStreamWrapper;
    friend class HandTrackerWrapper;
    friend class UserTrackerWrapper;

    inline openni::Device * getDevicePointer()
    {
        return &static_cast<Device &>(*this);
    }

    inline const openni::Device & getDeviceConstRef() const
    {
        return static_cast<const Device &>(*this);
    }

private:
    DeviceWrapper(const DeviceWrapper &);
};

void export_openni2();

#endif // OPENNI2_H
