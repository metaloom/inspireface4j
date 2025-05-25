#include "jinspireface.hpp"

static bool initialized = false;

static std::unique_ptr<HFSession> globalSession;
// static std::string sourcePathStr = "test_res/data/RD/d3.jpeg";

inline HResult CVImageToImageStream(const inspirecv::Image &image, HFImageStream &handle, HFImageFormat format = HF_STREAM_BGR,
                                    HFRotation rot = HF_CAMERA_ROTATION_0)
{

    HFLogPrint(HF_LOG_INFO, "Converting: %d x %d", image.Width(), image.Height());

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

HFSession setupSession(std::string resourcePath)
{
    // Initialization at the beginning of the program
    const char *path = resourcePath.c_str();

    // HFLogPrint(HF_LOG_INFO, "Creating Session %s", path);
    HResult ret = HFReloadInspireFace(path);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to launch InspireFace: %d", ret);
        return NULL;
    }

    HOption option = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS | HF_ENABLE_DETECT_MODE_LANDMARK | HF_ENABLE_FACE_ATTRIBUTE;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 20;
    // Face detection image input level
    HInt32 detectPixelLevel = 640;
    // Handle of the current face SDK algorithm context
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return NULL;
    }
    return session;
}

extern "C" void initializeSession(const char *packPath)
{
    if (!initialized)
    {
        HFSession session = setupSession(packPath);
        globalSession = std::make_unique<HFSession>(std::move(session));
        initialized = true;
    }
    else
    {
        HFLogPrint(HF_LOG_ERROR, "Session already initialized");
    }
}

HFImageStream loadImage(std::string sourcePathStr)
{
    HFImageBitmap image;
    HPath sourcePath = sourcePathStr.c_str();
    HResult ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return NULL;
    }
    // Prepare an image parameter structure for configuration
    HFImageStream stream = {0};
    ret = HFCreateImageStreamFromImageBitmap(image, HF_CAMERA_ROTATION_0, &stream);
    if (ret != HSUCCEED)
    {
        HFReleaseImageStream(stream);
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return NULL;
    }

    return stream;
}

HFImageStream ConvertCVImage(cv::Mat &cvimage)
{
    HFLogPrint(HF_LOG_INFO, "Converting: %d x %d", cvimage.cols, cvimage.rows);

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
        HFLogPrint(HF_LOG_INFO, "image handle: %ld", (long)imageSteamHandle);
    }

    return imageSteamHandle;
}

HFMultipleFaceData detectFaces(HFImageStream imageStream, cv::Mat *imagePtr, bool drawBoundingBoxes)
{
    cv::Mat image = *imagePtr;

    // HFLogPrint(HF_LOG_INFO, "Detecting...");
    HFSession session = *globalSession.get();

    // Execute HF_FaceContextRunFaceTrack captures face information in an image
    HFMultipleFaceData multipleFaceData = {0};
    HResult ret = HFExecuteFaceTrack(session, imageStream, &multipleFaceData);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
        return multipleFaceData;
    }

    // Print the number of faces detected
    auto faceNum = multipleFaceData.detectedNum;
    HFLogPrint(HF_LOG_INFO, "Num of face: %d", faceNum);

    if (drawBoundingBoxes)
    {
        // cv::Mat outImage = image.clone();

        for (int index = 0; index < faceNum; ++index)
        {
            HFaceRect faceRect = multipleFaceData.rects[index];
            HFLogPrint(HF_LOG_INFO, "Face: %d - [x%d:y%d:w%d:h%d]", index, faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            cv::Rect rect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0));

            // HFImageBitmapDrawRect(drawImage, multipleFaceData.rects[index], {0, 100, 255}, 4);
            //  Print FaceID, In IMAGE-MODE it is changing, in VIDEO-MODE it is fixed, but it may be lost
            HFLogPrint(HF_LOG_INFO, "FaceID: %d - Conf: %.2f", multipleFaceData.trackIds[index], multipleFaceData.detConfidence[index]);
            // Print Head euler angle, It can often be used to judge the quality of a face by the Angle
            // of the head
            // HFLogPrint(HF_LOG_INFO, "Roll: %f, Yaw: %f, Pitch: %f", multipleFaceData.angles.roll[index], multipleFaceData.angles.yaw[index],       multipleFaceData.angles.pitch[index]);
        }
        // std::string outputFile = "draw_detected.jpg";
        // HFImageBitmapWriteToFile(drawImage, );
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

int tearDownSession(HFSession session)
{
    HResult ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: Release session error: %d", ret);
        return ret;
    }

    // When you don't need it (you can ignore it)
    HFTerminateInspireFace();
    return 0;
}

extern "C" void releaseSession()
{
    if (initialized)
    {
        HFSession session = *globalSession.get();
        tearDownSession(session);
        initialized = false;
    }
    else
    {
        HFLogPrint(HF_LOG_ERROR, "Session not yet initialized");
    }
}

int getFaceEmbedding(HFMultipleFaceData multipleFaceData, HFImageStream imageStream)
{

    HFSession session = *globalSession.get();

    /*
        // Execute face tracking on the image
        HFMultipleFaceData multipleFaceData = {0};
        HResult ret = HFExecuteFaceTrack(session, stream, &multipleFaceData); // Track faces in the image
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Run face track error: %d", ret);
            return ret;
        }
        if (multipleFaceData.detectedNum == 0)
        { // Check if any faces were detected
            HFLogPrint(HF_LOG_ERROR, "No face was detected");
            return ret;
        }
        */

    // Extract facial features from the first detected face, an interface that uses copy features in a comparison scenario
    HFFaceFeature feature;
    HResult ret = HFCreateFaceFeature(&feature);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Create face feature error: %d", ret);
        return ret;
    }

    ret = HFFaceFeatureExtractCpy(session, imageStream, multipleFaceData.tokens[0], feature.data);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Extract feature error: %d", ret);
        return ret;
    }
    int size = feature.size;
    HFLogPrint(HF_LOG_INFO, "Extract feature size: %d", size);
    /*
    for (int i = 0; i < size; i++)
    {
        HFLogPrint(HF_LOG_INFO, "Vector[%d]: %.2f", i, feature.data[i]);
    }
    */

    // Not in use need to release
    HFReleaseFaceFeature(&feature);

    return 0;
}

HResult getFaceAttributes(HFFaceAttributeResult faceAttr, HFMultipleFaceData multipleFaceData, HFImageStream imageStream)
{
    HFSession session = *globalSession.get();

    HResult ret = HFMultipleFacePipelineProcessOptional(session, imageStream, &multipleFaceData, HF_ENABLE_FACE_ATTRIBUTE);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to run pipeline: %d", ret);
        return 10;
    }

    HFLogPrint(HF_LOG_INFO, "Loading face attributes.");

    ret = HFGetFaceAttributeResult(session, &faceAttr);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Create face attr error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Loaded face attributes.");
    if (faceAttr.num <= 0)
    {
        HFLogPrint(HF_LOG_ERROR, "No attr found");
    }
    else
    {
        HFLogPrint(HF_LOG_INFO, "Race: %d", faceAttr.race[0]);
        HFLogPrint(HF_LOG_INFO, "Gender: %d", faceAttr.gender[0]);
        HFLogPrint(HF_LOG_INFO, "AgeBracket: %d", faceAttr.ageBracket[0]);
    }

    return 0;
}

extern "C" HFFaceAttributeResult *faceAttributes(HFMultipleFaceData multipleFaceData, cv::Mat *imagePtr)
{
    HFLogPrint(HF_LOG_INFO, "Lib: factAttributes");
    cv::Mat image = *imagePtr;
    HFImageStream imageStream = ConvertCVImage(image);
    HFFaceAttributeResult data = {};
    HResult ret = getFaceAttributes(data, multipleFaceData, imageStream);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to run pipeline: %d", ret);
    }

    HFFaceAttributeResult *faceAttr = new HFFaceAttributeResult(data);
    return faceAttr;
}

extern "C" HFMultipleFaceData *detect(cv::Mat *imagePtr, bool drawBoundingBoxes)
{
    HFLogPrint(HF_LOG_INFO, "Lib: detect");
    cv::Mat image = *imagePtr;
    HFImageStream imageStream = ConvertCVImage(image);
    HFMultipleFaceData data = detectFaces(imageStream, imagePtr, drawBoundingBoxes);
    // HFLogPrint(HF_LOG_INFO, "Lib: detected %d", data.detectedNum);
    if (data.detectedNum != 0)
    {
        int faceNum = data.detectedNum;
        HFLogPrint(HF_LOG_INFO, "Lib: Detected: %d", faceNum);
        // getFaceEmbedding(data, imageStream);
        HFFaceAttributeResult attrData = {};

        HResult ret = getFaceAttributes(attrData, data, imageStream);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Lib: Failed to run pipeline: %d", ret);
        }
    }
    else
    {
        HFLogPrint(HF_LOG_ERROR, "Lib: No face data");
    }

    /*
        HResult ret = HFReleaseImageStream(imageStream);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
        }
        */

    HFMultipleFaceData *multipleFaceData = new HFMultipleFaceData(data);
    return multipleFaceData;
}
