# bytetrack_with_raft

## Abstract



https://github.com/kumasy107/bytetrack_with_raft/assets/64134440/1096f289-2e1f-42de-b84b-ef3d241a012e



ByteTrack is very strong Multi Object Tracker during tracking small objects like people etc., but it isn't good at tracking big objects like cars because ByteTrack is tracking with only using IoU and vulnerable to occlusion.
For another reason, the shape of the bounding box is deformed just before the occlusion starts.

To solve this problem, we propose an occlusion-resistant tracker that can track cars by using optical flow, and we also propose the robust way to save bounding box's shape just before the occlusion starts.

As a detector, a tracker and an optical flow predictor, we use YOLOX, ByteTrack and RAFT respectively.

In the fig below, we show the way to compare predicted to detected. OF stands for optical flow. We predict OF at frame t with Kalman filter, and compare it to OF at frame t obtained with RAFT. We consider IoU as well as OF in our comparisons.

<img width="400" alt="predict_en" src="https://github.com/kumasy107/bytetrack_with_raft/assets/64134440/61856f66-1ec0-4a8d-b3c9-7a751fa651fd">


+ In this study, we track cars at intersections because many occlusions happen there.
+ We installed a camera as the same height as traffic signal at an intersection.
+ When occlusion occurred, ours can continue to track cars precisely more than 90% although ByteTrack tracks about 40%.
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
