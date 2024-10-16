import os

class Config:
    # Dataset paths
    DATASETS_PATH = "datasets"
    
    KITTIRoad_TRAINING_IMAGES_PATH = os.path.join(DATASETS_PATH, "KITTIRoad/training/image_2")
    KITTIRoad_TRAINING_MASKS_PATH = os.path.join(DATASETS_PATH, "KITTIRoad/training/gt_image_2")
    
    RoadAnomaly_IMAGES_PATH = os.path.join(DATASETS_PATH, "RoadAnomaly/images")
    RoadAnomaly_MASKS_PATH = os.path.join(DATASETS_PATH, "RoadAnomaly/masks")
    
    LostAndFound_TRAIN_IMAGES_PATH = os.path.join(DATASETS_PATH, "LostAndFound/images/train")
    LostAndFound_TRAIN_MASKS_PATH = os.path.join(DATASETS_PATH, "LostAndFound/masks/train")
    LostAndFound_TEST_IMAGES_PATH = os.path.join(DATASETS_PATH, "LostAndFound/images/test")
    LostAndFound_TEST_MASKS_PATH = os.path.join(DATASETS_PATH, "LostAndFound/masks/test")
    
    VKITTI_IMAGES_PATH= os.path.join(DATASETS_PATH, "vkitti/images")
    VKITTI_MASKS_PATH= os.path.join(DATASETS_PATH, "vkitti/masks")

    # Model paths
    WEIGHTS_PATH = "weights"
    SAMPLES_PATH = "samples"
    
    # Training settings
    ROAD_BATCH_SIZE = 16
    ANOMALY_BATCH_SIZE = ROAD_BATCH_SIZE
    ROAD_LEARNING_RATE = 0.0001
    ANOMALY_LEARNING_RATE = 0.0005
    ROAD_EPOCHS = 1000
    ANOMALY_EPOCHS = ROAD_EPOCHS
    DEBUG = False
    CONTINUE_TRAINING = True
    ROAD_CHECKPOINT_PATH = os.path.join(WEIGHTS_PATH, 'best_road_model.pth')
    ANOMALY_CHECKPOINT_PATH = os.path.join(WEIGHTS_PATH, 'best_anomaly_model.pth')
    USE_BOTH_DATASETS = True

    # DataLoader settings
    PIN_MEMORY = True

    # Proxy settings
    PROXY = "http://x98.local:1092"
    os.environ['http_proxy'] = PROXY
    os.environ['https_proxy'] = PROXY
