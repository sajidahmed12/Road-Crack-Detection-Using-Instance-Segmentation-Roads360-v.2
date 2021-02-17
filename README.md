# Road-Crack-Detection-Using-Instance-Segmentation-Model-YOLACT

Previous work on Semantic Segmentaion ## Roads360- https://github.com/sajidahmed12/Roads360-Image-Segmentation-With-Kittiseg-Extended-Model 

### Objective
In the case of the autonomous driving in Bangladesh , given an front camera view, the car needs to know where is the road and where are the Crackes on the road surface . In this project, we trained a neural network to label the pixels of a road in images, by using a Model YOLACT and trained with our Own collected images for road and Crack segmentation.

### New Model/ State-Of-Art for Instance Segmentation 
**Will be Working with This Model Furture in future **
**Training Testing Evaluation as well** 

#### 1.2 Dependencies & my environment

Anaconda is used for managing the environment.

* Python 3.7.3 , Pytorch v1.1, CUDA 10.0 CuDnn 7.5.0, Numpy, SciPy,glob,tdqm
* OS: Windows 10 Pro
* CPU: Intel® Core™ i7-7700K CPU @ 4.000 GHz 4 core × 8 Threads
* GPU: NVidia GeForce GTX 1070 (8GB GDDR5X VRAM)
* Memory: 16GB DDR4 2400 MHz


### Command to Test/Eval Model From Pre-train Weights in ./Weights Folder

***link for Downloading Weights***  

## Weights                                                                                                              

yolact_base_54_800000.pth] (https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing) 

[Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg)

our Trained Model Weights: https://drive.google.com/file/d/1z9jg4JxYAHup7Jg6UlA3akLscURQS33q/view?usp=sharing

Put this Weight file under the ./weights folder

# For Video
## Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.

## use argument **"--display_fps"** to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video3.mp4

## Process a video and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4

# For image
## Display results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

## Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

## Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder

# Training
## Trains using the base config with a batch size of 8 ( default).
python train.py --config=yolact_base_config

## Resume training yolact_base with a Pretrain/specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# SOME RESULTS 

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
