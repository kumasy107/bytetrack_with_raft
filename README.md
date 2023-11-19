# bytetrack_with_raft

## Abstract

https://github.com/kumasy107/bytetrack_with_raft/assets/64134440/4444a4f2-644c-4d37-8bdf-6dfd68341092

ByteTrack is very strong Multi Object Tracker during tracking small objects like people etc., but it isn't good at tracking big objects like cars because ByteTrack is tracking with only IoU and vulnerable to occlusion.
To solve this problem, we propose an occlusion-resistant tracker that can track cars by using optical flow.
As a detector, tracker and optical flow predictor, we use YOLOX, ByteTrack and RAFT respectively.

+ In this study, we track cars at intersections because many occlusions happen there.
+ We installed a camera as the same height as traffic signal at an intersection.
+ When occlusion occurred, ours can continue to track cars more then 90% although ByteTrack tracks about 40%.
+ In the case of tracking a car at an intersection, both MOTA and IDF1 are improved over ByteTrack's.
+ No real time.

## Usage
This repository is based on [ByteTrack](https://github.com/ifzhang/ByteTrack).

1. clone this repository
   ```
   git clone https://github.com/kumasy107/bytetrack_with_raft
   cd bytetrack_with_raft
   ```
2. Install libraries
   ```
   pip install -r requirements.txt
   ```
3. Set pretrained weight below ./pretrained
   You can get YOLOX pretrained model from [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0).
4. put your video below ./videos and predict by running 
   ```
   python3 sample_entire_raft.py video --path ./videos/hoge.avi -n yolox-x -c pretrained/yolox_x.pth --match_thresh 0.8 --track_thresh 0.7 --fp16 --fuse --save_result --raft model RAFT/models/raft-sintel.pth --nms 0.45 --conf 0.1
   ```
