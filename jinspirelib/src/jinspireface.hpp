#ifndef JINSPIREFACE_HPP
#define JINSPIREFACE_HPP

#include <iostream>
#include <memory>
#include <inspireface.h>
#include <inspirecv/inspirecv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct HFLandMarkData
{
    HInt32 size;      ///< Size of the landmark data
    HPoint2f *points; ///< Pointer to the landmark points
} HFLandMarkData, *PHFLandMarkData;

#endif // JINSPIREFACE_HPP