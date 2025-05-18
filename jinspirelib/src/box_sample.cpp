#include <iostream>
#include <memory>
#include <inspireface.h>
#include <inspirecv/inspirecv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static bool initialized = false;

static std::unique_ptr<HFSession> globalSession;

inline HResult CVImageToImageStream(const inspirecv::Image &image, HFImageStream &handle, HFImageFormat format = HF_STREAM_BGR,
                                    HFRotation rot = HF_CAMERA_ROTATION_0)
{
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

HFSession setupSession()
{
    // Initialization at the beginning of the program
    std::string resourcePath = "test_res/pack/Pikachu";
    HResult ret = HFReloadInspireFace(resourcePath.c_str());
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

extern "C" void initializeSession()
{
    if (!initialized)
    {
        HFSession session = setupSession();
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
        //        HFReleaseImageBitmap(imageBitmap);
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return NULL;
    }

    return stream;
}

HFImageStream loadCVImage(cv::Mat *imagePtr)
{
    cv::Mat cvimage = *imagePtr;
    inspirecv::Image image(cvimage.cols, cvimage.rows, cvimage.channels(), cvimage.data);

    HFImageStream imgHandle;
    HResult ret = CVImageToImageStream(image, imgHandle);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to convert cv image: %d", ret);
        return NULL;
    }

    return imgHandle;
}

HFMultipleFaceData detectFaces(HFSession session, HFImageStream imageHandle, cv::Mat image)
{

    // Execute HF_FaceContextRunFaceTrack captures face information in an image
    HFMultipleFaceData multipleFaceData = {0};
    HResult ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
        return multipleFaceData;
    }

    // Print the number of faces detected
    auto faceNum = multipleFaceData.detectedNum;
    HFLogPrint(HF_LOG_INFO, "Num of face: %d", faceNum);

    /*
        // Copy a new image to draw
        HFImageBitmap drawImage = {0};
        ret = HFImageBitmapCopy(image, &drawImage);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Copy ImageBitmap error: %d", ret);
            return multipleFaceData;
        }
        HFImageBitmapData data;
        ret = HFImageBitmapGetData(drawImage, &data);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Get ImageBitmap data error: %d", ret);
            return multipleFaceData;
        }
    */
    cv::Mat outImage = image.clone();

    for (int index = 0; index < faceNum; ++index)
    {
        HFaceRect faceRect = multipleFaceData.rects[index];
        cv::Rect rect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
        cv::rectangle(outImage, rect, cv::Scalar(0, 255, 0));

        // HFImageBitmapDrawRect(drawImage, multipleFaceData.rects[index], {0, 100, 255}, 4);
        //  Print FaceID, In IMAGE-MODE it is changing, in VIDEO-MODE it is fixed, but it may be lost
        HFLogPrint(HF_LOG_INFO, "FaceID: %d - Conf: %.2f", multipleFaceData.trackIds[index], multipleFaceData.detConfidence[index]);
        // Print Head euler angle, It can often be used to judge the quality of a face by the Angle
        // of the head
        // HFLogPrint(HF_LOG_INFO, "Roll: %f, Yaw: %f, Pitch: %f", multipleFaceData.angles.roll[index], multipleFaceData.angles.yaw[index],       multipleFaceData.angles.pitch[index]);
    }
    // std::string outputFile = "draw_detected.jpg";
    // HFImageBitmapWriteToFile(drawImage, );

    bool check = imwrite("draw_detected.jpg", outImage);

    HFLogPrint(HF_LOG_WARN, "Write to file success: %s", "draw_detected.jpg");

    /*
        ret = HFReleaseImageStream(imageHandle);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
        }
        */
    return multipleFaceData;
}

int tearDownSession(HFSession session)
{
    HResult ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
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



int getFaceEmbedding(HFSession session, HFMultipleFaceData multipleFaceData, HFImageStream imageStream)
{

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

int getFaceAttributes(HFSession session, HFMultipleFaceData multipleFaceData, HFImageStream imageStream)
{

    HResult ret = HFMultipleFacePipelineProcessOptional(session, imageStream, &multipleFaceData, HF_ENABLE_FACE_ATTRIBUTE);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to run pipeline: %d", ret);
        return 10;
    }

    HFLogPrint(HF_LOG_INFO, "Loading face attributes.");
    HFFaceAttributeResult faceAttr = {};
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

HFImageStream loadImageStream(std::string sourcePathStr)
{
    // Load a image
    HFImageBitmap image;
    HPath sourcePath = sourcePathStr.c_str();
    HResult ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return NULL;
    }

    HFImageStream imageStream = loadImage(sourcePathStr);
    return imageStream;
}

int main()
{
    initializeSession();
    HFSession session = *globalSession.get();
    /*
        HFSession session = setupSession();
        if (session == NULL)
        {
            return 10;
        }
    */

    std::string sourcePathStr = "test_res/data/RD/d3.jpeg";
    //    std::string sourcePathStr = "test_res/data/bulk/pedestrian.png";
    cv::Mat img = cv::imread(sourcePathStr, cv::IMREAD_COLOR);

    HFImageStream imageStream = loadImageStream(sourcePathStr);

    HFLogPrint(HF_LOG_INFO, "Detecting...");
    HFMultipleFaceData multipleFaceData = detectFaces(session, imageStream, img);
    // HFMultipleFaceData multipleFaceData = *multipleFaceDataPtr;

    if (multipleFaceData.detectedNum != 0)
    {
        int faceNum = multipleFaceData.detectedNum;
        HFLogPrint(HF_LOG_INFO, "Detected: %d", faceNum);
        getFaceEmbedding(session, multipleFaceData, imageStream);
        getFaceAttributes(session, multipleFaceData, imageStream);
    }
    else
    {
        HFLogPrint(HF_LOG_ERROR, "No face data");
    }

    HResult ret = HFReleaseImageStream(imageStream);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
    }

    // The memory must be freed at the end of the program
    // return tearDownSession(session);
    releaseSession();

    return 0;
}