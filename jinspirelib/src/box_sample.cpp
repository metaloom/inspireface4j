#include <iostream>
#include <inspireface.h>

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

HFMultipleFaceData detectFaces(HFSession session, HFImageStream imageHandle, HFImageBitmap image)
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
    for (int index = 0; index < faceNum; ++index)
    {
        HFImageBitmapDrawRect(drawImage, multipleFaceData.rects[index], {0, 100, 255}, 4);
        // Print FaceID, In IMAGE-MODE it is changing, in VIDEO-MODE it is fixed, but it may be lost
        HFLogPrint(HF_LOG_INFO, "FaceID: %d - Conf: %.2f", multipleFaceData.trackIds[index], multipleFaceData.detConfidence[index]);
        // Print Head euler angle, It can often be used to judge the quality of a face by the Angle
        // of the head
        // HFLogPrint(HF_LOG_INFO, "Roll: %f, Yaw: %f, Pitch: %f", multipleFaceData.angles.roll[index], multipleFaceData.angles.yaw[index],       multipleFaceData.angles.pitch[index]);
    }
    // std::string outputFile = "draw_detected.jpg";
    HFImageBitmapWriteToFile(drawImage, "draw_detected.jpg");
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

void loadCVImage()
{
    /*
        HFImageStream imgHandle;
        auto img = inspirecv::Image::Create(GET_DATA("data/attribute/1423.jpg"));
        REQUIRE(!img.Empty());
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);
        */
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

int main()
{
    HFSession session = setupSession();
    if (session == NULL)
    {
        return 10;
    }

    // Load a image
    //    std::string sourcePathStr = "test_res/data/bulk/pedestrian.png";
    std::string sourcePathStr = "test_res/data/RD/d3.jpeg";
    HFImageBitmap image;
    HPath sourcePath = sourcePathStr.c_str();
    HResult ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return 10;
    }

    HFImageStream imageStream = loadImage(sourcePathStr);

    HFLogPrint(HF_LOG_INFO, "Detecting...");
    HFMultipleFaceData multipleFaceData = detectFaces(session, imageStream, image);
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

    // The memory must be freed at the end of the program
    return tearDownSession(session);
    // return 10;
}