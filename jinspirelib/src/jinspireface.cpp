#include "jinspireface.hpp"

static HFLogLevel LOG = HF_LOG_INFO;

inline HResult CVImageToImageStream(const inspirecv::Image &image, HFImageStream &handle, HFImageFormat format = HF_STREAM_BGR,
                                    HFRotation rot = HF_CAMERA_ROTATION_0)
{

    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: Converting: %d x %d", image.Width(), image.Height());
    }

    if (image.Empty())
    {
        return -1;
    }
    HFImageData imageData = {0};
    imageData.data = (uint8_t *)image.Data();
    imageData.height = image.Height();
    imageData.width = image.Width();
    imageData.format = format;
    imageData.rotation = rot;

    auto ret = HFCreateImageStream(&imageData, &handle);

    return ret;
}

HFSession setupSession(std::string resourcePath, HInt32 detectPixelLevel, HOption option, HInt32 maxDetectNum)
{

    // Initialization at the beginning of the program
    const char *path = resourcePath.c_str();

    // HFLogPrint(HF_LOG_INFO, "Creating Session %s", path);
    HResult ret = HFReloadInspireFace(path);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Failed to launch InspireFace: %d", ret);
        return NULL;
    }

    // HF_ENABLE_DETECT_MODE_LANDMARK
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;

    // Handle of the current face SDK algorithm context
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Create FaceContext error: %d", ret);
        return NULL;
    }
    return session;
}

extern "C" HFSession *createSession(const char *packPath, HInt32 detectPixelLevel, HOption option, HInt32 maxDetectNum)
{
    HFLogPrint(HF_LOG_ERROR, "Lib: Session create");
    HFSession session = setupSession(packPath, detectPixelLevel, option, maxDetectNum);
    HFSession *sessionPtr = new HFSession(session);
    HFLogPrint(HF_LOG_ERROR, "Lib: Session created");
    return sessionPtr;
}

HFImageStream ConvertCVImage(cv::Mat &cvimage)
{
    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: Converting: %d x %d", cvimage.cols, cvimage.rows);
    }

    HFImageData imageData = {0};
    imageData.data = cvimage.data;
    imageData.height = cvimage.rows;
    imageData.width = cvimage.cols;
    imageData.rotation = HF_CAMERA_ROTATION_0;
    imageData.format = HF_STREAM_BGR;

    HFImageStream imageSteamHandle;
    HResult ret = HFCreateImageStream(&imageData, &imageSteamHandle);
    if (ret == HSUCCEED)
    {
        if (LOG == HF_LOG_DEBUG)
        {
            HFLogPrint(HF_LOG_DEBUG, "Lib: Image handle: %ld", (long)imageSteamHandle);
        }
    }

    return imageSteamHandle;
}

HFMultipleFaceData detectFaces(HFSession *sessionPtr, HFImageStream imageStream, cv::Mat *imagePtr, bool drawBoundingBoxes)
{
    cv::Mat image = *imagePtr;

    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: Detecting...");
    }
    HFSession session = *sessionPtr;

    // Execute HF_FaceContextRunFaceTrack captures face information in an image
    HFMultipleFaceData multipleFaceData = {0};
    HResult ret = HFExecuteFaceTrack(session, imageStream, &multipleFaceData);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Execute HFExecuteFaceTrack error: %d", ret);
        return multipleFaceData;
    }

    // Print the number of faces detected
    auto faceNum = multipleFaceData.detectedNum;
    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: Num of face: %d", faceNum);
    }

    if (drawBoundingBoxes)
    {
        // cv::Mat outImage = image.clone();

        for (int index = 0; index < faceNum; ++index)
        {
            HFaceRect faceRect = multipleFaceData.rects[index];

            cv::Rect rect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0));
        }
    }

    if (LOG == HF_LOG_DEBUG)
    {
        for (int index = 0; index < faceNum; ++index)
        {
            HFaceRect faceRect = multipleFaceData.rects[index];
            HFLogPrint(HF_LOG_DEBUG, "Lib: Face: %d - [x%d:y%d:w%d:h%d]", index, faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            HFLogPrint(HF_LOG_DEBUG, "Lib: FaceID: %d - Conf: %.2f", multipleFaceData.trackIds[index], multipleFaceData.detConfidence[index]);

            //  Print FaceID, In IMAGE-MODE it is changing, in VIDEO-MODE it is fixed, but it may be lost
            // Print Head euler angle, It can often be used to judge the quality of a face by the Angle
            // of the head
            // HFLogPrint(HF_LOG_INFO, "Roll: %f, Yaw: %f, Pitch: %f", multipleFaceData.angles.roll[index], multipleFaceData.angles.yaw[index],       multipleFaceData.angles.pitch[index]);
        }
    }

    /*
        ret = HFReleaseImageStream(imageStream);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Lib: Release image stream error: %d", ret);
        }
    */
    return multipleFaceData;
}

extern "C" int releaseSession(HFSession *session)
{

    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: Session releasing");
    }
    HResult ret = HFReleaseInspireFaceSession(*session);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Release session error: %d", ret);
        return ret;
    }

    // When you don't need it (you can ignore it)
    HFTerminateInspireFace();
    return 0;
}

HResult getFaceEmbedding(HFSession *sessionPtr, HFFaceFeature *feature, HFMultipleFaceData multipleFaceData, HFImageStream imageStream, int faceNr)
{

    HFSession session = *sessionPtr;

    // TODO assert faceNr

    auto faceNum = multipleFaceData.detectedNum;
    HFLogPrint(HF_LOG_INFO, "Lib: Embedding Num of face: %d", faceNum);

    // Extract facial features from the first detected face, an interface that uses copy features in a comparison scenario
    HResult ret = HFCreateFaceFeature(feature);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Create face embedding error: %d", ret);
        return ret;
    }

    ret = HFFaceFeatureExtractCpy(session, imageStream, multipleFaceData.tokens[faceNr], (*feature).data);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Extract embedding error: %d", ret);
        return ret;
    }
    int size = (*feature).size;
    HFLogPrint(HF_LOG_INFO, "Lib: Extract feature size: %d", size);

    if (LOG == HF_LOG_DEBUG)
    {

        for (int i = 0; i < size; i++)
        {
            HFLogPrint(HF_LOG_DEBUG, "Vector[%d]: %.2f", i, (*feature).data[i]);
        }
    }

    return 0;
}

HResult getFaceAttributes(HFSession *sessionPtr, HFFaceAttributeResult *faceAttrPtr, HFMultipleFaceData multipleFaceData, HFImageStream imageStream)
{
    HFSession session = *sessionPtr;

    HResult ret = HFMultipleFacePipelineProcessOptional(session, imageStream, &multipleFaceData, HF_ENABLE_FACE_ATTRIBUTE);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Failed to run pipeline: %d", ret);
        return 10;
    }
    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: Loading face attributes.");
    }

    ret = HFGetFaceAttributeResult(session, faceAttrPtr);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Create face attr error: %d", ret);
        return ret;
    }
    return 0;
}

extern "C" HFFaceFeature *faceEmbeddings(HFSession *sessionPtr, HFMultipleFaceData *multipleFaceDataPtr, cv::Mat *imagePtr, int faceNr)
{
    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: faceEmbeddings");
    }

    HFMultipleFaceData multipleFaceData = *multipleFaceDataPtr;
    cv::Mat image = *imagePtr;
    HFImageStream imageStream = ConvertCVImage(image);

    HFFaceFeature *feature = new HFFaceFeature();
    HResult ret = getFaceEmbedding(sessionPtr, feature, multipleFaceData, imageStream, faceNr);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Failed to run pipeline: %d", ret);
    }

    return feature;
}

extern "C" HFFaceAttributeResult *faceAttributes(HFSession *sessionPtr, HFMultipleFaceData *multipleFaceDataPtr, cv::Mat *imagePtr)
{

    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: factAttributes");
    }
    HFFaceAttributeResult data = {};
    HFMultipleFaceData multipleFaceData = *multipleFaceDataPtr;
    cv::Mat image = *imagePtr;
    HFImageStream imageStream = ConvertCVImage(image);
    HResult ret = getFaceAttributes(sessionPtr, &data, multipleFaceData, imageStream);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Failed to run pipeline: %d", ret);
    }

    if (LOG == HF_LOG_DEBUG)
    {
        if (data.num <= 0)
        {
            HFLogPrint(HF_LOG_DEBUG, "Lib: No attr found");
        }
        else
        {
            HFLogPrint(HF_LOG_DEBUG, "Lib: Race: %d", data.race[0]);
            HFLogPrint(HF_LOG_DEBUG, "Lib: Gender: %d", data.gender[0]);
            HFLogPrint(HF_LOG_DEBUG, "Lib: AgeBracket: %d", data.ageBracket[0]);
        }
    }
    HFFaceAttributeResult *faceAttr = new HFFaceAttributeResult(data);
    return faceAttr;
}

extern "C" HFMultipleFaceData *detect(HFSession *sessionPtr, cv::Mat *imagePtr, bool drawBoundingBoxes)
{
    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: detect()");
    }
    cv::Mat image = *imagePtr;
    HFImageStream imageStream = ConvertCVImage(image);
    HFMultipleFaceData data = detectFaces(sessionPtr, imageStream, imagePtr, drawBoundingBoxes);
    if (LOG == HF_LOG_DEBUG)
    {
        if (data.detectedNum != 0)
        {
            int faceNum = data.detectedNum;
            HFLogPrint(HF_LOG_DEBUG, "Lib: Detected: %d", faceNum);
        }
        else
        {
            HFLogPrint(HF_LOG_DEBUG, "Lib: No face data");
        }
    }

    HResult ret = HFReleaseImageStream(imageStream);
    if (ret != HSUCCEED && LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Release image stream error: %d", ret);
    }

    HFMultipleFaceData *multipleFaceData = new HFMultipleFaceData(data);
    return multipleFaceData;
}

extern "C" void releaseFaceFeature(PHFFaceFeature *feature)
{
    HFReleaseFaceFeature(*feature);
}

extern "C" void logLevel(HFLogLevel level)
{
    HFLogPrint(HF_LOG_INFO, "Lib: Setting log level");
    HFSetLogLevel(level);
    LOG = level;
    if (LOG == HF_LOG_DEBUG)
    {
        HFLogPrint(HF_LOG_DEBUG, "Lib: DEBUG TEST");
    }
}