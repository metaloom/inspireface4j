#include <string>
#include <string.h>
#include "jinspireface.hpp"

// For Test code
#include <iostream>
#include <string>
#include <memory>
/*
#include "../insightface/challenges/inspireface/initialization_module/launch.h"
#include <inspireface/middleware/inspirecv_image_process.h>
#include "inspireface/track_module/landmark/face_landmark_adapt.h"
*/
extern "C" void test_str(const char *name, const char *name2)
{
    printf("Test %s - %s\n", name, name2);
    fflush(stdout);
}

extern "C" void initialize(const char *labelsPath, const char *modelPath, bool useGPU)
{
    printf("Initializing YoloLib using model %s - labels %s - Using GPU: %s\n", modelPath, labelsPath, useGPU ? "true" : "false");
    fflush(stdout);
}

/**
extern "C" int test(cv::Mat *imagePtr) {
    std::string expansion_path = "";
    INSPIRE_LAUNCH->Load("test_res/pack/Pikachu-t4");
    auto archive = INSPIRE_LAUNCH->getMArchive();

    inspire::InspireModel lmkModel;
    auto ret = archive.LoadModel("landmark", lmkModel);
    if (ret != 0) {
        INSPIRE_LOGE("Load %s error: %d", "landmark", ret);
        return -1;
    }

    inspire::FaceLandmarkAdapt lmk;
    lmk.loadData(lmkModel, lmkModel.modelType);

    
    //auto data = image.Resize(112, 112);
    cv::Mat image = *imagePtr;
    auto lmk_out = lmk(image);
    std::vector<inspirecv::Point2i> landmarks_output(inspire::FaceLandmarkAdapt::NUM_OF_LANDMARK);
    for (int i = 0; i < inspire::FaceLandmarkAdapt::NUM_OF_LANDMARK; ++i) {
        float x = lmk_out[i * 2 + 0] * image.Width();
        float y = lmk_out[i * 2 + 1] * image.Height();
        landmarks_output[i] = inspirecv::Point<int>(x, y);
    }

    for (int i = 0; i < landmarks_output.size(); ++i) {
        image.DrawCircle(landmarks_output[i], 5, {0, 0, 255});
    }
    image.Write("crop_lmk.png");
}
*/