#include <iostream>
#include <memory>
#include <inspireface.h>
#include <inspirecv/inspirecv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static bool initialized = false;

static std::unique_ptr<HFSession> globalSession;
static std::string sourcePathStr = "test_res/pexels-olly-3812743_1k.jpg";

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

HFSession setupSession()
{
    // Initialization at the beginning of the program
    std::string resourcePath = "packs/Pikachu";
    HResult ret = HFReloadInspireFace(resourcePath.c_str());
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to launch InspireFace: %d", ret);
        return NULL;
    }

    HOption option = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_QUALITY | HF_ENABLE_LIVENESS | HF_ENABLE_FACE_ATTRIBUTE;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 40;
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

HFImageStream ConvertCVImage(cv::Mat &cvimage)
{
    HFLogPrint(HF_LOG_INFO, "Converting: %d x %d", cvimage.cols, cvimage.rows);

    // cv::Mat cvimage = *imagePtr;
    // inspirecv::Image image(cvimage.cols, cvimage.rows, cvimage.channels(), cvimage.data);
    // auto image = inspirecv::Image::Create("test_res/data/RD/d3.jpeg");

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

    /*
        HFImageStream imgHandle;
        HResult ret = CVImageToImageStream(image, imgHandle);
        if (ret != HSUCCEED)
        {
            HFLogPrint(HF_LOG_ERROR, "Failed to convert cv image: %d", ret);
            return NULL;
        }
    */
    return imageSteamHandle;
}

HFMultipleFaceData detectFaces(HFImageStream imageHandle, cv::Mat image, bool drawBoundingBoxes)
{
    HFLogPrint(HF_LOG_INFO, "Detecting...");
    HFSession session = *globalSession.get();

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
    if (drawBoundingBoxes)
    {
        // cv::Mat outImage = image.clone();

        for (int index = 0; index < faceNum; ++index)
        {
            HFaceRect faceRect = multipleFaceData.rects[index];
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

int getFaceAttributes(HFMultipleFaceData multipleFaceData, HFImageStream imageStream)
{
    HFSession session = *globalSession.get();

    HOption pipelineOption = HF_ENABLE_QUALITY | HF_ENABLE_FACE_ATTRIBUTE;

    /* In this loop, all faces are processed */
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

int getDenseLandmarkFromFace(HFMultipleFaceData multipleFaceData, cv::Mat image, int faceNr, bool drawPoints, HPoint2f *denseLandmarkPoints)
{
    HInt32 numOfLmk;

    /* Get the number of dense landmark points */
    HFGetNumOfFaceDenseLandmark(&numOfLmk);
    denseLandmarkPoints = (HPoint2f *)malloc(sizeof(HPoint2f) * numOfLmk);
    if (denseLandmarkPoints == NULL)
    {
        HFLogPrint(HF_LOG_ERROR, "Memory allocation failed!");
        return -1;
    }

    HResult ret = HFGetFaceDenseLandmarkFromFaceToken(multipleFaceData.tokens[faceNr], denseLandmarkPoints, numOfLmk);
    if (ret != HSUCCEED)
    {
        free(denseLandmarkPoints);
        HFLogPrint(HF_LOG_ERROR, "HFGetFaceDenseLandmarkFromFaceToken error!");
        return -1;
    }

    if (drawPoints)
    {

        /* Draw dense landmark points */
        for (int i = 0; i < numOfLmk; i++)
        {

            // cv::Point point{denseLandmarkPoints[i].x, denseLandmarkPoints[i].y};
            cv::Point point(denseLandmarkPoints[i].x, denseLandmarkPoints[i].y);
            cv::circle(image, point, 2, cv::Scalar(255, 0, 255), 0, 8, 1);

            // TODO change this to CV
            /*
            HFImageBitmapDrawCircleF(drawImage,
                                     (HPoint2f){denseLandmarkPoints[i].x, denseLandmarkPoints[i].y},
                                     0,
                                     (HColor){100, 100, 0},
                                     2);
            */
        }
        free(denseLandmarkPoints);
    }
    return 0;
}

void detect(cv::Mat *imagePtr, bool drawBoundingBoxes)
{
    cv::Mat image = *imagePtr;

    // HFImageStream imageStream = loadImageStream(sourcePathStr);
    HFImageStream imageStream = ConvertCVImage(image);

    HFMultipleFaceData multipleFaceData = detectFaces(imageStream, image, drawBoundingBoxes);
    // HFMultipleFaceData multipleFaceData = *multipleFaceDataPtr;

    if (multipleFaceData.detectedNum != 0)
    {
        int faceNum = multipleFaceData.detectedNum;
        HFLogPrint(HF_LOG_INFO, "Detected: %d", faceNum);
        getFaceEmbedding(multipleFaceData, imageStream);
        getFaceAttributes(multipleFaceData, imageStream);
        for (int i = 0; i < faceNum; i++)
        {
            HPoint2f *denseLandmarkPoints;
            getDenseLandmarkFromFace(multipleFaceData, image, i, true, denseLandmarkPoints);
        }
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
}

void writeImage(cv::Mat image)
{
    bool check = imwrite("draw_detected.jpg", image);
    if (check)
    {
        HFLogPrint(HF_LOG_WARN, "Write to file success: %s", "draw_detected.jpg");
    }
    else
    {
        HFLogPrint(HF_LOG_ERROR, "Write to file failed: %s", "draw_detected.jpg");
    }
}

int main()
{
    //    std::string sourcePathStr = "test_res/data/bulk/pedestrian.png";
    initializeSession();
    cv::Mat image = cv::imread(sourcePathStr, cv::IMREAD_COLOR);
    detect(&image, true);
    writeImage(image);
    releaseSession();
    return 0;
}