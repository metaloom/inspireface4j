#include "handlers.hpp"

FaceDetectorHandler::FaceDetectorHandler()
{
    face_detector = dlib::get_frontal_face_detector();
}
dlib::frontal_face_detector FaceDetectorHandler::getFaceDetector()
{
    return face_detector;
}

FaceDetectorHandler *FaceDetectorHandler::getFaceDetectorHandler()
{
    static FaceDetectorHandler *face_detector_handler;
    if (!face_detector_handler) {
        face_detector_handler = new FaceDetectorHandler;
    }
    return face_detector_handler;
}

ShapePredictorHandler::ShapePredictorHandler(const std::string &model_path) : model_path(model_path)
{
    dlib::deserialize(model_path) >> shape_predictor_model;
}

dlib::shape_predictor ShapePredictorHandler::getShapePredictorModel()
{
    return shape_predictor_model;
}

ShapePredictorHandler *ShapePredictorHandler::getShapePredictorHandler(const std::string &model_path)
{
    static ShapePredictorHandler *shape_predictor_handler;
    if (!shape_predictor_handler) {
        shape_predictor_handler = new ShapePredictorHandler(model_path);
    }
    return shape_predictor_handler;
}

// CNN Face Detector Handler

CNNFaceDetectorHandler::CNNFaceDetectorHandler(const std::string &model_path) : model_path(model_path)
{
    dlib::deserialize(model_path) >> cnn_face_model;
}

net_type CNNFaceDetectorHandler::getCNNFaceDetectorModel()
{
    return cnn_face_model;
}

CNNFaceDetectorHandler *CNNFaceDetectorHandler::getCNNFaceDetectorHandler(const std::string &model_path)
{
    static CNNFaceDetectorHandler *cnn_face_detector_handler;
    if (!cnn_face_detector_handler) {
        cnn_face_detector_handler = new CNNFaceDetectorHandler(model_path);
    }
    return cnn_face_detector_handler;
}


// Face EmbeddingHandler

FaceEmbeddingHandler::FaceEmbeddingHandler(const std::string &model_path) : model_path(model_path)
{
    dlib::deserialize(model_path) >> face_embedding_model;
}

anet_type FaceEmbeddingHandler::getFaceEmbeddingModel()
{
    return face_embedding_model;
}

FaceEmbeddingHandler *FaceEmbeddingHandler::getFaceEmbeddingHandler(const std::string &model_path)
{
    static FaceEmbeddingHandler *face_embedding_handler;
    if (!face_embedding_handler) {
        face_embedding_handler = new FaceEmbeddingHandler(model_path);
    }
    return face_embedding_handler;
}
