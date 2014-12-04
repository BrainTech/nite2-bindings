
#define NO_IMPORT_ARRAY
#include "boost_python.h"
#include "nite2.h"

#include <NiTE.h>

#include "wrapper_utils.h"

#include "openni2.h"

#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif __GNUC__
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

//-----------------------------------------------------------------------------------------------

class NiteWrapper : private nite::NiTE
{
public:
    static inline nite::Status initialize()
    {
        ScopedGILRelease release_gil;
        return nite::NiTE::initialize();
    }

    static inline void shutdown()
    {
        ScopedGILRelease release_gil;
        nite::NiTE::shutdown();
    }

    static inline VersionWrapper<nite::Version> getVersion()
    {
        return nite::NiTE::getVersion();
    }
};

//-----------------------------------------------------------------------------------------------

template<typename T>
static void extract_point3_from_py_obj(const bp::object & obj, T & x, T & y, T & z)
{
    T new_x, new_y, new_z;
    try
    {
        new_x = bp::extract<T>(obj[0]);
        new_y = bp::extract<T>(obj[1]);
        new_z = bp::extract<T>(obj[2]);
    }
    catch(...)
    {
        try
        {
            new_x = bp::extract<T>(obj.attr("x"));
            new_y = bp::extract<T>(obj.attr("y"));
            new_z = bp::extract<T>(obj.attr("z"));
        }
        catch(...)
        {
            throw std::runtime_error("cannot extract point3 from python object");
        }
    }
    x = new_x;
    y = new_y;
    z = new_z;
}

template<typename T>
static void extract_point4_from_py_obj(const bp::object & obj, T & w, T & x, T & y, T & z)
{
    T new_w, new_x, new_y, new_z;
    try
    {
        new_w = bp::extract<T>(obj[0]);
        new_x = bp::extract<T>(obj[1]);
        new_y = bp::extract<T>(obj[2]);
        new_z = bp::extract<T>(obj[3]);
    }
    catch(...)
    {
        try
        {
            new_w = bp::extract<T>(obj.attr("w"));
            new_x = bp::extract<T>(obj.attr("x"));
            new_y = bp::extract<T>(obj.attr("y"));
            new_z = bp::extract<T>(obj.attr("z"));
        }
        catch(...)
        {
            throw std::runtime_error("cannot extract point4 from python object");
        }
    }
    w = new_w;
    x = new_x;
    y = new_y;
    z = new_z;    
}

//-----------------------------------------------------------------------------------------------

class Point3fWrapper : private nite::Point3f
{
public:
    inline Point3fWrapper() 
        : nite::Point3f()
    {
    }

    inline Point3fWrapper(const NitePoint3f & p) 
        : nite::Point3f(p.x, p.y, p.z)
    {
    }
    
    inline Point3fWrapper(const Point3fWrapper & p)
        : nite::Point3f(p.x, p.y, p.z) 
    {
    }
    
    inline Point3fWrapper(float x, float y, float z) 
        : nite::Point3f(x, y, z) 
    {
    }
    
    inline Point3fWrapper(const bp::object & obj)
    {
        try
        {
            *this = bp::extract<Point3fWrapper>(obj);
        }
        catch(...)
        {
            try
            {
                extract_point3_from_py_obj(obj, x, y, z);
            }
            catch(...)
            {
                throw std::runtime_error("cannot create Point3fWrapper from python object");
            }
        }
    }

    inline float getX() const { return x; }
    inline float getY() const { return y; }
    inline float getZ() const { return z; }

    inline void setX(float x_) { x = x_; }
    inline void setY(float y_) { y = y_; }
    inline void setZ(float z_) { z = z_; }

    inline void set(float x_, float y_, float z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }

    inline int getTupleLength() const
    {
        return 3;
    }

    inline float getTupleElem(int n) const
    {
        switch(n)
        {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default:
                // translated by Boost.Python to IndexError (not RangeError)
                throw std::out_of_range("Point contains only 3 elements");
        }
    }

    inline void setTupleElem(int n, float val)
    {
        switch(n)
        {
            case 0: x = val; break;
            case 1: y = val; break;
            case 2: z = val; break;
            default:
                // translated by Boost.Python to IndexError (not RangeError)
                throw std::out_of_range("Point contains only 3 elements");
        }
    }

    inline const nite::Point3f & getPoint3fConstRef()
    {
        return static_cast<const nite::Point3f &>(*this);
    }
};

inline std::ostream & operator << (std::ostream & output, const Point3fWrapper & p)
{
    return output << "(x = " << p.getX() << ", y = " << p.getY() << ", z = " << p.getZ() << ')';
}

//-----------------------------------------------------------------------------------------------

class QuaternionWrapper : private nite::Quaternion
{
public:
    inline QuaternionWrapper() 
        : nite::Quaternion()
    {
    }
    
    inline QuaternionWrapper(const NiteQuaternion & p) 
        : nite::Quaternion(p.w, p.x, p.y, p.z)
    {
    }
    
    inline QuaternionWrapper(const QuaternionWrapper & p) 
        : nite::Quaternion(p.w, p.x, p.y, p.z) 
    {
    }
    
    inline QuaternionWrapper(float w, float x, float y, float z) 
        : nite::Quaternion(w, x, y, z) 
    {
    }
    
    inline QuaternionWrapper(const bp::object & obj)
    {
        try
        {
            *this = bp::extract<QuaternionWrapper>(obj);
        }
        catch(...)
        {
            try
            {
                extract_point4_from_py_obj(obj, w, x, y, z);
            }
            catch(...)
            {
                throw std::runtime_error("cannot create QuaternionWrapper from python object");
            }
        }
    }

    inline float getW() const { return w; }
    inline float getX() const { return x; }
    inline float getY() const { return y; }
    inline float getZ() const { return z; }

    inline void setW(float w_) { z = w_; }
    inline void setX(float x_) { x = x_; }
    inline void setY(float y_) { y = y_; }
    inline void setZ(float z_) { z = z_; }

    inline void set(float w_, float x_, float y_, float z_)
    {
        w = w_;
        x = x_;
        y = y_;
        z = z_;
    }

    inline int getTupleLength() const
    {
        return 4;
    }

    inline float getTupleElem(int n) const
    {
        switch(n)
        {
            case 0: return w;
            case 1: return x;
            case 2: return y;
            case 3: return z;
            default:
                // translated to by boost::python to IndexError
                throw std::out_of_range("Quaternion contains only 4 elements");
        }
    }

    inline void setTupleElem(int n, float val)
    {
        switch(n)
        {
            case 0: w = val; break;
            case 1: x = val; break;
            case 2: y = val; break;
            case 3: z = val; break;
            default:
                // translated to by boost::python to IndexError
                throw std::out_of_range("Quaternion contains only 4 elements");
        }
    }
};

inline std::ostream & operator << (std::ostream & output, const QuaternionWrapper & p)
{
    return output << "(w = " << p.getW() << ", x = " << p.getX() << ", y = " << p.getY() << ", z = " << p.getZ() << ')';
}

//-----------------------------------------------------------------------------------------------

class BoundingBoxWrapper
{
public:
    inline BoundingBoxWrapper(const nite::BoundingBox & b)
        : m_min(b.min)
        , m_max(b.max)
    {
    }

    inline const Point3fWrapper getMin() const
    {
        return m_min;
    }

    inline const Point3fWrapper getMax() const
    {
        return m_max;
    }

private:
    const Point3fWrapper m_min;
    const Point3fWrapper m_max;
};

//-----------------------------------------------------------------------------------------------

class SkeletonJointWrapper : private nite::SkeletonJoint
{
public:
    inline SkeletonJointWrapper(const nite::SkeletonJoint & j)
        : nite::SkeletonJoint(j)
    {
    }

    inline nite::JointType getType() const
    {
        return nite::SkeletonJoint::getType();
    }

    inline Point3fWrapper getPosition() const
    {
        return Point3fWrapper(nite::SkeletonJoint::getPosition());
    }

    inline float getPositionConfidence() const
    {
        return nite::SkeletonJoint::getPositionConfidence();
    }

    inline QuaternionWrapper getOrientation() const
    {
        return QuaternionWrapper(nite::SkeletonJoint::getOrientation());
    }

    inline float getOrientationConfidence() const
    {
        return nite::SkeletonJoint::getOrientationConfidence();
    }
};

//-----------------------------------------------------------------------------------------------

class PlaneWrapper
{
public:
    inline PlaneWrapper(const nite::Plane & p = nite::Plane())
        : m_point(p.point)
        , m_normal(p.normal)
    {
    }

    inline PlaneWrapper & operator = (const nite::Plane & p)
    {
        m_point = p.point;
        m_normal = p.normal;
        return *this;
    }

    inline const Point3fWrapper getPoint() const
    {
        return m_point;
    }

    inline const Point3fWrapper getNormal() const
    {
        return m_normal;
    }

private:
    Point3fWrapper m_point;
    Point3fWrapper m_normal;
};

//-----------------------------------------------------------------------------------------------

class SkeletonWrapper : private nite::Skeleton
{
public:
    inline SkeletonWrapper(const nite::Skeleton & s)
        : nite::Skeleton(s)
    {
    }

    inline const SkeletonJointWrapper getJoint(nite::JointType type) const
    {
        return SkeletonJointWrapper(nite::Skeleton::getJoint(type));
    }

    inline nite::SkeletonState getState() const
    {
        return nite::Skeleton::getState();
    }
};

//-----------------------------------------------------------------------------------------------

class UserDataWrapper : private nite::UserData
{
public:
    inline UserDataWrapper(const nite::UserData & ud)
        : nite::UserData(ud)
    {
    }

    inline nite::UserId getId() const
    {
        return nite::UserData::getId();
    }

    inline const BoundingBoxWrapper getBoundingBox() const
    {
        return BoundingBoxWrapper(nite::UserData::getBoundingBox());
    }

    inline const Point3fWrapper getCenterOfMass() const
    {
        return Point3fWrapper(nite::UserData::getCenterOfMass());
    }

    inline bool isNew() const
    {
        return nite::UserData::isNew();
    }

    inline bool isVisible() const
    {
        return nite::UserData::isVisible();
    }

    inline bool isLost() const
    {
        return nite::UserData::isLost();
    }

    inline const SkeletonWrapper getSkeleton() const
    {
        return SkeletonWrapper(nite::UserData::getSkeleton());
    }

    inline const nite::PoseData getPose(nite::PoseType type) const
    {
        return nite::PoseData(nite::UserData::getPose(type));
    }
};

//-----------------------------------------------------------------------------------------------

class UserTrackerFrameWrapper
{
public:
    inline UserTrackerFrameWrapper(const nite::UserTrackerFrameRef & f)
    {
        m_isValid = f.isValid();
        if(m_isValid)
        {
            m_floorConfidence = f.getFloorConfidence();
            m_floor = f.getFloor();
            m_timestamp = f.getTimestamp();
            m_frameIndex = f.getFrameIndex();

            const nite::Array<nite::UserData> & users = f.getUsers();
            const int n_users = users.getSize();
            m_user_ids.resize(n_users);
            for(int i = 0; i < n_users; i++)
            {
                const nite::UserData & user_data = users[i];
                m_users.append(UserDataWrapper(user_data));
                m_user_ids[i] = user_data.getId();
            }

            const nite::UserMap & m = f.getUserMap();
            const nite::UserId * pixels = m.getPixels();

            if(pixels)
            {
                if(m.getStride() != int(sizeof(nite::UserId)) * m.getWidth())
                    throw std::runtime_error("data format where stride != sizeof(UserId)*width is not currently supported");

                // size of array is equal to sizeof(UserId) * height * stride
                npy_intp dims[2] = { m.getHeight(), m.getWidth() };
                PyObject * py_obj = PyArray_SimpleNewFromData(2, dims, NPY_SHORT, (void*)pixels);

                bp::handle<> handle(py_obj);
                bp::numeric::array arr(handle);

                // The problem of returning arr is twofold: firstly the user can modify
                //  the data which will betray the const-correctness
                //  Secondly the lifetime of the data is managed by the C++ API and not the lifetime
                //  of the numpy array whatsoever. But we have a simply solution..

                m_userMap = arr.copy(); // copy the object. numpy owns the copy now.
            }

            // TODO...
            //m_depthFrame = VideoFrameWrapper(f.getDepthFrame());
        }
        else
        {
            m_floorConfidence = 0.0;
            m_timestamp = 0;
            m_frameIndex = -1;
        }
    }

    inline bool isValid() const
    {
        return m_isValid;
    }

    inline bp::object getUserById(nite::UserId id) const
    {
        for(unsigned int i = 0; i < m_user_ids.size(); i++)
            if(m_user_ids[i] == id)
                 return m_users[i];
        return bp::object();
    }

    inline const bp::list getUsers() const
    {
        return m_users;
    }

    inline float getFloorConfidence() const
    {
        return m_floorConfidence;
    }

    inline const PlaneWrapper getFloor() const
    {
        return m_floor;
    }

    inline const VideoFrameWrapper getDepthFrame()
    {
        return m_depthFrame;
    }

    const bp::object getUserMap() const
    {
        return m_userMap;
    }

    inline uint64_t getTimestamp() const
    {
        return m_timestamp;
    }

    inline int getFrameIndex() const
    {
        return m_frameIndex;
    }

private:
    bool m_isValid;
    float m_floorConfidence;
    PlaneWrapper m_floor;
    bp::list m_users;
    std::vector<nite::UserId> m_user_ids;
    bp::object m_userMap;
    VideoFrameWrapper m_depthFrame;
    uint64_t m_timestamp;
    int m_frameIndex;
};

//-----------------------------------------------------------------------------------------------

class HandDataWrapper
{
public:
    inline HandDataWrapper(const nite::HandData & d)
        : m_id(d.getId())
        , m_position(d.getPosition())
        , m_flags(d.isNew(), d.isLost(), d.isTracking(), d.isTouchingFov())
    {}

    inline nite::HandId getId() const
    {
        return m_id;
    }

    inline const Point3fWrapper getPosition() const
    {
        return m_position;
    }

    inline bool isNew() const
    {
        return m_flags.isNew;
    }

    inline bool isLost() const
    {
        return m_flags.isLost;
    }

    inline bool isTracking() const
    {
        return m_flags.isTracking;
    }

    inline bool isTouchingFov() const
    {
        return m_flags.isTouchingFov;
    }

private:
    const nite::HandId m_id;
    const Point3fWrapper m_position;

    const struct Flags
    {
        inline Flags(bool isNew_, bool isLost_, bool isTracking_, bool isTouchingFov_)
            : isNew(isNew_)
            , isLost(isLost_)
            , isTracking(isTracking_)
            , isTouchingFov(isTouchingFov_)
        {}

        unsigned char isNew : 1;
        unsigned char isLost : 1;
        unsigned char isTracking : 1;
        unsigned char isTouchingFov : 1;
    } m_flags;
};

//-----------------------------------------------------------------------------------------------

class GestureDataWrapper
{
public:
    inline GestureDataWrapper(const nite::GestureData & d)
        : m_type(d.getType())
        , m_currentPosition(d.getCurrentPosition())
        , m_flags(d.isComplete(), d.isInProgress())
    {
    }

    inline nite::GestureType getType() const
    {
        return m_type;
    }

    inline const Point3fWrapper getCurrentPosition() const
    {
        return m_currentPosition;
    }

    inline bool isComplete() const
    {
        return m_flags.isComplete;
    }

    inline bool isInProgress() const
    {
        return m_flags.isInProgress;
    }

private:
    const nite::GestureType m_type;
    const Point3fWrapper m_currentPosition;

    const struct Flags
    {
        inline Flags(bool isComplete_, bool isInProgress_)
            : isComplete(isComplete_)
            , isInProgress(isInProgress_)
        {}

        unsigned char isComplete : 1;
        unsigned char isInProgress : 1;
    } m_flags;
};

//-----------------------------------------------------------------------------------------------

class HandTrackerFrameWrapper
{
public:
    inline HandTrackerFrameWrapper(const nite::HandTrackerFrameRef & f)
    {
        m_isValid = f.isValid();
        if(m_isValid)
        {
            m_timetamp = f.getTimestamp();
            m_frameIndex = f.getFrameIndex();

            const nite::Array<nite::HandData> & hd = f.getHands();
            for(int i = 0; i < hd.getSize(); i++)
                m_hands.append(HandDataWrapper(hd[i]));

            const nite::Array<nite::GestureData> & gd = f.getGestures();
            for(int i = 0; i < gd.getSize(); i++)
                m_gestures.append(GestureDataWrapper(gd[i]));

            m_depthFrame = VideoFrameWrapper(f.getDepthFrame());
        }
        else
        {
            m_timetamp = 0;
            m_frameIndex = -1;
        }
    }

    inline bool isValid() const
    {
        return m_isValid;
    }

    inline const bp::list getHands() const
    {
        return m_hands;
    }

    inline const bp::list getGestures() const
    {
        return m_gestures;
    }

    inline VideoFrameWrapper getDepthFrame() const
    {
        return m_depthFrame;
    }

    inline uint64_t getTimestamp() const
    {
        return m_timetamp;
    }

    inline int getFrameIndex() const
    {
        return m_frameIndex;
    }
    
private:
    bool m_isValid;
    bp::list m_hands;
    bp::list m_gestures;
    VideoFrameWrapper m_depthFrame;
    uint64_t m_timetamp;
    int m_frameIndex;
};

//-----------------------------------------------------------------------------------------------

class UserTrackerWrapper : private nite::UserTracker,
                           private nite::UserTracker::NewFrameListener
{
public:
    inline UserTrackerWrapper()
    {
    }

    inline ~UserTrackerWrapper()
    {
        if(!m_frameListeners.empty())
            nite::UserTracker::removeNewFrameListener(this);
    }

    inline bool isValid() const
    {
        return nite::UserTracker::isValid();
    }

    inline nite::Status create(const bp::object & device = bp::object())
    {
        if(device.is_none())
        {
            ScopedGILRelease release_gil;
            return nite::UserTracker::create();
        }
        else
        {
            DeviceWrapper & dev = bp::extract<DeviceWrapper&>(device);
            ScopedGILRelease release_gil;
            return nite::UserTracker::create(dev.getDevicePointer());
        }
    }

    inline void destroy()
    {
        ScopedGILRelease release_gil;
        nite::UserTracker::destroy();
    }

    inline bp::tuple readFrame()
    {
        nite::Status status;
        nite::UserTrackerFrameRef frameRef;

        {
            ScopedGILRelease release_gil;
            status = nite::UserTracker::readFrame(&frameRef);
        }

        if(status != nite::STATUS_OK)
            return bp::make_tuple(status, bp::object());
        else
            return bp::make_tuple(status, UserTrackerFrameWrapper(frameRef));
    }

    inline nite::Status setSkeletonSmoothingFactor(float factor)
    {
        return nite::UserTracker::setSkeletonSmoothingFactor(factor);
    }

    inline float getSkeletonSmoothingFactor() const
    {
        return nite::UserTracker::getSkeletonSmoothingFactor();
    }

    inline nite::Status startSkeletonTracking(nite::UserId id)
    {
        return nite::UserTracker::startSkeletonTracking(id);
    }

    inline void stopSkeletonTracking(nite::UserId id)
    {
        nite::UserTracker::stopSkeletonTracking(id);
    }

    inline nite::Status startPoseDetection(nite::UserId user, nite::PoseType type)
    {
        return nite::UserTracker::startPoseDetection(user, type);
    }

    inline void stopPoseDetection(nite::UserId user, nite::PoseType type)
    {
        nite::UserTracker::stopPoseDetection(user, type);
    }

    inline bp::tuple convertJointCoordinatesToDepth(float x, float y, float z) const
    {
        float outX, outY;
        nite::Status status = nite::UserTracker::convertJointCoordinatesToDepth(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline bp::tuple convertJointCoordinatesToDepth_obj(const bp::object & obj) const
    {
        float x, y, z;
        float outX, outY;
        extract_point3_from_py_obj(obj, x, y, z);
        nite::Status status = nite::UserTracker::convertJointCoordinatesToDepth(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline bp::tuple convertDepthCoordinatesToJoint(int x, int y, int z) const
    {
        float outX, outY;
        nite::Status status = nite::UserTracker::convertDepthCoordinatesToJoint(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline bp::tuple convertDepthCoordinatesToJoint_obj(const bp::object & obj) const
    {
        int x, y, z;
        float outX, outY;
        extract_point3_from_py_obj(obj, x, y, z);
        nite::Status status = nite::UserTracker::convertDepthCoordinatesToJoint(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline void addNewFrameListener(const bp::object & new_frame_callback)
    {
        if(addListener(m_frameListeners, new_frame_callback))
            nite::UserTracker::addNewFrameListener(this);
    }

    inline void removeNewFrameListener(const bp::object & new_frame_callback)
    {
        if(removeListener(m_frameListeners, new_frame_callback))
            nite::UserTracker::removeNewFrameListener(this);
    }

private:
    virtual void onNewFrame(nite::UserTracker &) override
    {
        GILStateScopedEnsure gilEnsure;
        for(auto & obj : m_frameListeners)
        {
            try
            {
                obj();
            }
            catch(...)
            {
            }
        }
    }

    std::vector<bp::object> m_frameListeners;
};

//-----------------------------------------------------------------------------------------------

class HandTrackerWrapper : private nite::HandTracker,
                           private nite::HandTracker::NewFrameListener
{
public:
    inline HandTrackerWrapper()
    {
    }

    inline ~HandTrackerWrapper()
    {
        if(!m_frameListeners.empty())
            nite::HandTracker::removeNewFrameListener(this);
    }

    inline bool isValid() const
    {
        return nite::HandTracker::isValid();
    }

    inline nite::Status create(const bp::object & device = bp::object())
    {
        if(device.is_none())
        {
            ScopedGILRelease release_gil;
            return nite::HandTracker::create();
        }
        else
        {
            DeviceWrapper & dev = bp::extract<DeviceWrapper&>(device);
            ScopedGILRelease release_gil;
            return nite::HandTracker::create(dev.getDevicePointer());
        }
    }

    inline void destroy()
    {
        ScopedGILRelease release_gil;
        nite::HandTracker::destroy();
    }

    inline bp::tuple readFrame()
    {
        nite::Status status;
        nite::HandTrackerFrameRef frame;

        {
            ScopedGILRelease release_gil;
            status = nite::HandTracker::readFrame(&frame);
        }

        if(status != nite::STATUS_OK)
            return bp::make_tuple(status, bp::object());
        else
            return bp::make_tuple(status, HandTrackerFrameWrapper(frame));
    }

    inline nite::Status setSmoothingFactor(float factor)
    {
        return nite::HandTracker::setSmoothingFactor(factor);
    }

    inline float getSmoothingFactor() const
    {
        return nite::HandTracker::getSmoothingFactor();
    }

    inline bp::tuple startHandTracking(const bp::object & position)
    {
        nite::HandId newHandId;
        nite::Status status = nite::HandTracker::startHandTracking(Point3fWrapper(position).getPoint3fConstRef(), &newHandId);
        if(status != nite::STATUS_OK)
            return bp::make_tuple(status, bp::object());
        else
            return bp::make_tuple(status, newHandId);
    }

    inline void stopHandTracking(nite::HandId id)
    {
        nite::HandTracker::stopHandTracking(id);
    }

    inline nite::Status startGestureDetection(nite::GestureType type)
    {
        return nite::HandTracker::startGestureDetection(type);
    }

    inline void stopGestureDetection(nite::GestureType type)
    {
        nite::HandTracker::stopGestureDetection(type);
    }

    inline bp::tuple convertHandCoordinatesToDepth(float x, float y, float z) const
    {
        float outX, outY;
        nite::Status status = nite::HandTracker::convertHandCoordinatesToDepth(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline bp::tuple convertHandCoordinatesToDepth_obj(const bp::object & obj) const
    {
        float x, y, z;
        extract_point3_from_py_obj(obj, x, y, z);
        float outX, outY;
        nite::Status status = nite::HandTracker::convertHandCoordinatesToDepth(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline bp::tuple convertDepthCoordinatesToHand(int x, int y, int z) const
    {
        float outX, outY;
        nite::Status status = nite::HandTracker::convertDepthCoordinatesToHand(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline bp::tuple convertDepthCoordinatesToHand_obj(const bp::object & obj) const
    {
        int x, y, z;
        extract_point3_from_py_obj(obj, x, y, z);
        float outX, outY;
        nite::Status status = nite::HandTracker::convertDepthCoordinatesToHand(x, y, z, &outX, &outY);
        return bp::make_tuple(status, outX, outY);
    }

    inline void addNewFrameListener(const bp::object & new_frame_callback)
    {
        if(addListener(m_frameListeners, new_frame_callback))
            nite::HandTracker::addNewFrameListener(this);
    }

    inline void removeNewFrameListener(const bp::object & new_frame_callback)
    {
        if(removeListener(m_frameListeners, new_frame_callback))
            nite::HandTracker::removeNewFrameListener(this);
    }

private:
    virtual void onNewFrame(nite::HandTracker &) override
    {
        GILStateScopedEnsure gilEnsure;
        for(auto & obj : m_frameListeners)
        {
            try
            {
                obj();
            }
            catch(...)
            {
            }
        }
    }

    std::vector<bp::object> m_frameListeners;
};

//-----------------------------------------------------------------------------------------------

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(HandTrackerWrapper_create_overloads,
                                       HandTrackerWrapper::create,
                                       0, 1);

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(UserTrackerWrapper_create_overloads,
                                       UserTrackerWrapper::create,
                                       0, 1);

//-----------------------------------------------------------------------------------------------

void export_nite2()
{
    using namespace boost::python;

    enum_<nite::JointType>("JointType", "Available joints in skeleton")
        .value("JOINT_HEAD",           nite::JOINT_HEAD)
        .value("JOINT_NECK",           nite::JOINT_NECK)
        .value("JOINT_LEFT_SHOULDER",  nite::JOINT_LEFT_SHOULDER)
        .value("JOINT_RIGHT_SHOULDER", nite::JOINT_RIGHT_SHOULDER)
        .value("JOINT_LEFT_ELBOW",     nite::JOINT_LEFT_ELBOW)
        .value("JOINT_RIGHT_ELBOW",    nite::JOINT_RIGHT_ELBOW)
        .value("JOINT_LEFT_HAND",      nite::JOINT_LEFT_HAND)
        .value("JOINT_RIGHT_HAND",     nite::JOINT_RIGHT_HAND)
        .value("JOINT_TORSO",          nite::JOINT_TORSO)
        .value("JOINT_LEFT_HIP",       nite::JOINT_LEFT_HIP)
        .value("JOINT_RIGHT_HIP",      nite::JOINT_RIGHT_HIP)
        .value("JOINT_LEFT_KNEE",      nite::JOINT_LEFT_KNEE)
        .value("JOINT_RIGHT_KNEE",     nite::JOINT_RIGHT_KNEE)
        .value("JOINT_LEFT_FOOT",      nite::JOINT_LEFT_FOOT)
        .value("JOINT_RIGHT_FOOT",     nite::JOINT_RIGHT_FOOT)
        .export_values();

    enum_<nite::SkeletonState>("SkeletonState", "Possible states of the skeleton")
        .value("SKELETON_NONE",                          nite::SKELETON_NONE)
        .value("SKELETON_CALIBRATING",                   nite::SKELETON_CALIBRATING)
        .value("SKELETON_TRACKED",                       nite::SKELETON_TRACKED)
        .value("SKELETON_CALIBRATION_ERROR_NOT_IN_POSE", nite::SKELETON_CALIBRATION_ERROR_NOT_IN_POSE)
        .value("SKELETON_CALIBRATION_ERROR_HANDS",       nite::SKELETON_CALIBRATION_ERROR_HANDS)
        .value("SKELETON_CALIBRATION_ERROR_HEAD",        nite::SKELETON_CALIBRATION_ERROR_HEAD)
        .value("SKELETON_CALIBRATION_ERROR_LEGS",        nite::SKELETON_CALIBRATION_ERROR_LEGS)
        .value("SKELETON_CALIBRATION_ERROR_TORSO",       nite::SKELETON_CALIBRATION_ERROR_TORSO)
        .export_values();
        ;

    enum_<nite::Status>("NiteStatus", "Possible failure values")
        .value("NITE_STATUS_OK",          nite::STATUS_OK)
        .value("NITE_STATUS_ERROR",       nite::STATUS_ERROR)
        .value("NITE_STATUS_BAD_USER_ID", nite::STATUS_BAD_USER_ID)
        .value("NITE_STATUS_OUT_OF_FLOW", nite::STATUS_OUT_OF_FLOW)
        .export_values();
        ;

    enum_<nite::PoseType>("PoseType")
        .value("POSE_PSI",           nite::POSE_PSI)
        .value("POSE_CROSSED_HANDS", nite::POSE_CROSSED_HANDS)
        .export_values();
        ;

    enum_<nite::GestureType>("GestureType", "Available gestures types")
        .value("GESTURE_WAVE",       nite::GESTURE_WAVE)
        .value("GESTURE_CLICK",      nite::GESTURE_CLICK)
        .value("GESTURE_HAND_RAISE", nite::GESTURE_HAND_RAISE)
        .export_values();
        ;

    //----------------------------------------------------------------------------------------------

    class_<VersionWrapper<nite::Version>>("NiteVersion", no_init)
        .def(self_ns::str(self_ns::self))
        .add_property("major",       &VersionWrapper<nite::Version>::getMajor)
        .add_property("minor",       &VersionWrapper<nite::Version>::getMinor)
        .add_property("maintenance", &VersionWrapper<nite::Version>::getMaintenance)
        .add_property("build",       &VersionWrapper<nite::Version>::getBuild)
        ;

    class_<Point3fWrapper>("Point3f", "Encapsulates a single point in 3D space, storing the x/y/z coordinates as floating point numbers.")
        .def(init<Point3fWrapper>())
        .def(init<float, float, float>())
        .def(self_ns::str(self_ns::self))
        .add_property("x", &Point3fWrapper::getX, &Point3fWrapper::setX)
        .add_property("y", &Point3fWrapper::getY, &Point3fWrapper::setY)
        .add_property("z", &Point3fWrapper::getZ, &Point3fWrapper::setZ)
        .def("set",         &Point3fWrapper::set)
        .def("__len__",     &Point3fWrapper::getTupleLength)
        .def("__getitem__", &Point3fWrapper::getTupleElem)
        .def("__setitem__", &Point3fWrapper::setTupleElem)
        ;

    class_<QuaternionWrapper>("Quaternion", "Represents a Quaternion.  The Quaternion is stored as four floating point numbers.  (The quaternions are a number system that extends the complex number system from two dimensions to four.)")
        .def(init<QuaternionWrapper>())
        .def(init<float, float, float, float>())
        .def(self_ns::str(self_ns::self))
        .add_property("x", &QuaternionWrapper::getX, &QuaternionWrapper::setX)
        .add_property("y", &QuaternionWrapper::getY, &QuaternionWrapper::setY)
        .add_property("z", &QuaternionWrapper::getZ, &QuaternionWrapper::setZ)
        .add_property("w", &QuaternionWrapper::getW, &QuaternionWrapper::setW)
        .def("set",         &QuaternionWrapper::set)
        .def("__len__",     &QuaternionWrapper::getTupleLength)
        .def("__getitem__", &QuaternionWrapper::getTupleElem)
        .def("__setitem__", &QuaternionWrapper::setTupleElem)
        ;

    class_<PlaneWrapper>("Plane", "Enapsulates a geometrical plane", no_init)
        .add_property("point",  &PlaneWrapper::getPoint)
        .add_property("normal", &PlaneWrapper::getNormal)
        ;

    class_<BoundingBoxWrapper>("BoundingBox", no_init)
        .add_property("min", &BoundingBoxWrapper::getMin)
        .add_property("max", &BoundingBoxWrapper::getMax)
        ;

    class_<nite::PoseData>("PoseData", no_init)
        .add_property("type", &nite::PoseData::getType)
        .def("isHeld",        &nite::PoseData::isHeld)
        .def("isEntered",     &nite::PoseData::isEntered)
        .def("isExited",      &nite::PoseData::isExited)
        ;

    class_<SkeletonJointWrapper>("SkeletonJoint", no_init)
        .add_property("type",                  &SkeletonJointWrapper::getType)
        .add_property("position",              &SkeletonJointWrapper::getPosition)
        .add_property("positionConfidence",    &SkeletonJointWrapper::getPositionConfidence)
        .add_property("orientation",           &SkeletonJointWrapper::getOrientation)
        .add_property("orientationConfidence", &SkeletonJointWrapper::getOrientationConfidence)
        ;

    class_<SkeletonWrapper>("Skeleton", no_init)
        .add_property("state", &SkeletonWrapper::getState)
        .def("getJoint",       &SkeletonWrapper::getJoint)
        .def("__getitem__",    &SkeletonWrapper::getJoint)
        ;

    class_<UserDataWrapper>("UserData", no_init)
        .add_property("id",           &UserDataWrapper::getId)
        .add_property("boundingBox",  &UserDataWrapper::getBoundingBox)
        .add_property("centerOfMass", &UserDataWrapper::getCenterOfMass)
        .add_property("skeleton",     &UserDataWrapper::getSkeleton)
        .add_property("pose",         &UserDataWrapper::getPose)
        .def("isNew",                 &UserDataWrapper::isNew)
        .def("isVisible",             &UserDataWrapper::isVisible)
        .def("isLost",                &UserDataWrapper::isLost)
        ;

    class_<GestureDataWrapper>("GestureData", no_init)
        .add_property("type",            &GestureDataWrapper::getType)
        .add_property("currentPosition", &GestureDataWrapper::getCurrentPosition)
        .def("isComplete",               &GestureDataWrapper::isComplete)
        .def("isInProgress",             &GestureDataWrapper::isInProgress)
        ;

    class_<HandTrackerFrameWrapper>("HandTrackerFrame", no_init)
        .add_property("hands",      &HandTrackerFrameWrapper::getHands)
        .add_property("gestures",   &HandTrackerFrameWrapper::getGestures)
        .add_property("depthFrame", &HandTrackerFrameWrapper::getDepthFrame)
        .add_property("timestamp",  &HandTrackerFrameWrapper::getTimestamp)
        .add_property("frameIndex", &HandTrackerFrameWrapper::getFrameIndex)
        .def("isValid",             &HandTrackerFrameWrapper::isValid)
        ;

    class_<HandDataWrapper>("HandData", no_init)
        .add_property("id",         &HandDataWrapper::getId)
        .add_property("position",   &HandDataWrapper::getPosition)
        .def("isNew",               &HandDataWrapper::isNew)
        .def("isLost",              &HandDataWrapper::isLost)
        .def("isTracking",          &HandDataWrapper::isTracking)
        .def("isTouchingFov",       &HandDataWrapper::isTouchingFov)
        ;

    class_<UserTrackerFrameWrapper>("UserTrackerFrame", no_init)
        .add_property("users",           &UserTrackerFrameWrapper::getUsers)
        .add_property("floorConfidence", &UserTrackerFrameWrapper::getFloorConfidence)
        .add_property("floor",           &UserTrackerFrameWrapper::getFloor)
        .add_property("depthFrame",      &UserTrackerFrameWrapper::getDepthFrame)
        .add_property("userMap",         &UserTrackerFrameWrapper::getUserMap)
        .add_property("timestamp",       &UserTrackerFrameWrapper::getTimestamp)
        .add_property("frameIndex",      &UserTrackerFrameWrapper::getFrameIndex)
        .def("isValid",                  &UserTrackerFrameWrapper::isValid)
        .def("getUserById",              &UserTrackerFrameWrapper::getUserById)
        ;

    class_<UserTrackerWrapper, boost::noncopyable>("UserTracker")
        .def("create",                         &UserTrackerWrapper::create, UserTrackerWrapper_create_overloads())
        .def("destroy",                        &UserTrackerWrapper::destroy)
        .def("readFrame",                      &UserTrackerWrapper::readFrame)
        .def("isValid",                        &UserTrackerWrapper::isValid)
        .def("setSkeletonSmoothingFactor",     &UserTrackerWrapper::setSkeletonSmoothingFactor)
        .def("getSkeletonSmoothingFactor",     &UserTrackerWrapper::getSkeletonSmoothingFactor)
        .def("startSkeletonTracking",          &UserTrackerWrapper::startSkeletonTracking)
        .def("stopSkeletonTracking",           &UserTrackerWrapper::stopSkeletonTracking)
        .def("startPoseDetection",             &UserTrackerWrapper::startPoseDetection)
        .def("stopPoseDetection",              &UserTrackerWrapper::stopPoseDetection)
        .def("addNewFrameListener",            &UserTrackerWrapper::addNewFrameListener)
        .def("removeNewFrameListener",         &UserTrackerWrapper::removeNewFrameListener)
        .def("convertJointCoordinatesToDepth", &UserTrackerWrapper::convertJointCoordinatesToDepth)
        .def("convertDepthCoordinatesToJoint", &UserTrackerWrapper::convertDepthCoordinatesToJoint)
        .def("convertJointCoordinatesToDepth", &UserTrackerWrapper::convertJointCoordinatesToDepth_obj)
        .def("convertDepthCoordinatesToJoint", &UserTrackerWrapper::convertDepthCoordinatesToJoint_obj)
        ;

    class_<HandTrackerWrapper, boost::noncopyable>("HandTracker", "HandTracker provides access to all algorithms relates to tracking individual hands, as well as detecting gestures in the depthmap.")
        .def("create",                        &HandTrackerWrapper::create, HandTrackerWrapper_create_overloads())
        .def("destroy",                       &HandTrackerWrapper::destroy)
        .def("readFrame",                     &HandTrackerWrapper::readFrame)
        .def("isValid",                       &HandTrackerWrapper::isValid)
        .def("getSmoothingFactor",            &HandTrackerWrapper::getSmoothingFactor)
        .def("setSmoothingFactor",            &HandTrackerWrapper::setSmoothingFactor)
        .def("startHandTracking",             &HandTrackerWrapper::startHandTracking)
        .def("stopHandTracking",              &HandTrackerWrapper::stopHandTracking)
        .def("addNewFrameListener",           &HandTrackerWrapper::addNewFrameListener)
        .def("removeNewFrameListener",        &HandTrackerWrapper::removeNewFrameListener)
        .def("startGestureDetection",         &HandTrackerWrapper::startGestureDetection)
        .def("stopGestureDetection",          &HandTrackerWrapper::stopGestureDetection)
        .def("convertHandCoordinatesToDepth", &HandTrackerWrapper::convertHandCoordinatesToDepth)
        .def("convertDepthCoordinatesToHand", &HandTrackerWrapper::convertDepthCoordinatesToHand)
        .def("convertHandCoordinatesToDepth", &HandTrackerWrapper::convertHandCoordinatesToDepth_obj)
        .def("convertDepthCoordinatesToHand", &HandTrackerWrapper::convertDepthCoordinatesToHand_obj)
        ;

    //----------------------------------------------------------------------------------------------

    class_<NiteWrapper, boost::noncopyable>("NiTE", "The NiTE class is a static entry point to the library", no_init)
        .def("initialize", &NiteWrapper::initialize).staticmethod("initialize")
        .def("shutdown",   &NiteWrapper::shutdown)  .staticmethod("shutdown")
        .def("getVersion", &NiteWrapper::getVersion).staticmethod("getVersion")
        ;

}
