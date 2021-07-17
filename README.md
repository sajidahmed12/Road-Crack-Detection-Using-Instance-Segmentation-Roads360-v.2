# Road-Crack-Detection-Using-Instance-Segmentation-Model-YOLACT (Roads360 v.2)

Codes and Model Credit Goes to 
 - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)

## My Previous work on Semantic Segmentaion [Roads360](https://github.com/sajidahmed12/Roads360-Image-Segmentation-With-Kittiseg-Extended-Model) 

---

## Objective
Bangladesh has been fighting the issue of ensuring a safer road management
system since its inception. Yet Bangladesh is suffering from an incapable and unattended
road communication system. Building good-quality roads with sustainable materials
covers only a part of maintaining and preserving a safe road system. But maintaining
real-time observation and constant data collection about the routes is a must to
ensure safe roads for safe commutation. Automation in real-time data collection about
the road surface condition and providing analyzed feedback to the government and people
can effectively reduce road accidents across the country. In this project, we
present a smart system for road health monitoring. The two main branches of this project
are Large-scale crowd-sourced data collection of road-surface condition through user application
and Real-time feedback based on image segmentation to detect road-cracks
and anomalies. We believe that such a crowd-sourcing-based platform will be highly effective
in ensuring a safer commutation experience for citizens of Bangladesh. In autonomous driving in Bangladesh, given a front camera view, the car needs to know where the road is and Crackers on the road surface. In this project, we trained a neural network to label the pixels of a road in images by using a Model YOLACT and trained with our Own collected images for road and Crack segmentation.


### Watch the Demo video : [here](https://youtu.be/y2mP3m7w1gY)
![][16] 

# 499 Capstone Project Showcase Poster 
![][img]

[img]: ./poster/poster499.JPG



# Installation
 - Clone this repository:
   ```Shell
   git clone https://github.com/sajidahmed12/Road-Crack-Detection-Using-Instance-Segmentation-Model-YOLACT.git
   cd Road-Crack-Detection-Using-Instance-Segmentation-Model-YOLACT
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```

## 1.2 Dependencies & my environment

Anaconda is used for managing the environment.

* Python 3.7.*+ , Pytorch v1.1, CUDA 10.0 CuDnn 7.5.0, Numpy, SciPy,glob,tdqm
* OS: Windows 10 Pro
* CPU: Intel® Core™ i7-7700K CPU @ 4.00-4.60 GHz 4 core × 8 Threads
* GPU: NVidia GeForce GTX 1070 (8GB GDDR5X VRAM)
* RAM: 16GB DDR4 2400 MHz


## Yolact Weights        

YOLACT++ models default weights:

| Image Size | Backbone |   |  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Resnet50-FPN  |  |  | [yolact_plus_resnet50_54_800000.pth](https://drive.google.com/file/d/1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EcJAtMiEFlhAnVsDf00yWRIBUC4m8iE9NEEiV05XwtEoGw) |
| 550        | Resnet101-FPN |  |  | [yolact_plus_base_54_800000.pth](https://drive.google.com/file/d/15id0Qq5eqRbkD-N3ZjDZXdCvRyIaHpFB/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EVQ62sF0SrJPrl_68onyHF8BpG7c05A8PavV4a849sZgEA)

To evalute the model with the `Yolact base default` config , put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).

## Our Trained Model Weights        

Model  weights:

| Image Size | Backbone |   |  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Yolact++ with Resnet101-FPN  |  |  | [yolact_base_23688_1066000.pth](https://drive.google.com/file/d/1z9jg4JxYAHup7Jg6UlA3akLscURQS33q/view)  | [Mirror](https://drive.google.com/file/d/1z9jg4JxYAHup7Jg6UlA3akLscURQS33q/view) |
|

To evalute the model with our trained Model for `Road/Crack Segmentation`, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).


## Testing the Model (with image)
```Shell
# Results on the specified image.
python eval.py --trained_model=weights/yolact_base_23688_1066000.pth --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_23688_1066000.pth --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_23688_1066000.pth --images=path/to/input/folder:path/to/output/folder
```
## Testing the Model (with video)
```Shell

# Display a video in real-time. Author's Suggests that "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_23688_1066000.pth --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time.
python eval.py --trained_model=weights/yolact_base_23688_1066000.pth  --video_multiframe=4 --video=0

# Process a video and save it to another file.
python eval.py --trained_model=weights/yolact_base_23688_1066000.pth --video_multiframe=4 --video=input_video.mp4:output_video.mp4
```
for more try the eval.help
```Shell
python eval.py --help
```


# Training
We trained on with out own dataset. Make sure to download the entire dataset from the link `here`.
 - To train, grab an pretrained model and put it in `./weights` folder.
   - Download `yolact_base_23688_1066000.pth` from [here](https://drive.google.com/file/d/1z9jg4JxYAHup7Jg6UlA3akLscURQS33q/view)

 - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 .
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

## Custom Datasets
We have used this method for training our own dataset by following these steps:

 - We have created a definition for our dataset under `dataset_base` in `data/config.py`
```Python
ROAD_CLASSES = ("road", "crack", "surroundings")

road_dataset = dataset_base.copy({
    'name': 'Road Dataset',

    'train_images': 'data/data_road/training',
    'train_info':   'data/data_road/training/annotations.json',

    'valid_images': 'data/data_road/training',
    'valid_info':   'data/data_road/training/annotations.json',

    'has_gt': True,
    'class_names': ROAD_CLASSES
})
```



## SOME RESULTS 

![][0]
![][1]
![][3]
![][4]
![][5]
![][6]
![][7]
![][8]
![][10]
![][11]
![][12]
![][13]
![][14]
![][15]
![][16]
![][17]
![][18]
![][19]
![][20]


[//]: # (Results outputs)
[0]: ./results/outputs_1.png
[1]: ./results/outputs_3.png
[3]: ./results/outputs_5.png
[4]: ./results/outputs_26.png
[5]: ./results/outputs_22.png
[6]: ./results/outputs_23.png
[7]: ./results/outputs_12.png
[8]: ./results/outputs_11.png
[10]: ./results/outputs_13.png
[11]: ./results/outputs_17.png
[12]: ./results/outputs_21.png
[13]: ./results/outputs_25.png
[14]: ./results/outputs_31.png
[15]: ./results/outputs_41.png
[16]: ./results/outputs_6.png
[17]: ./results/outputs_10.png
[18]: ./results/outputs_20.png
[19]: ./results/outputs_25.png
[20]: ./results/outputs_21.png

## if any issues questions please please contact [Md Sajid Ahmed ](mailto:sajid.ahmed1@northsouth.edu)
