# bytetrack_with_raft

## Abstract
ByteTrackはIoUのみでID対応付けを行うため、隣接するフレーム間での対応付けはめっぽう強いものの、他の物体による遮蔽（オクルージョン）に対して弱い。
その対策として、オプティカルフロー（画面内での物体の動きベクトル）をIoUと紐付けることで、オクルージョンに対応することを試みた。
物体検出ではYOLOX、複数物体追跡ではByteTrack、オプティカルフローの予測にはRAFTを用いている。

+ GPUを必要とする。
+ 交差点を走行する車両に対して、信号の高さと同じ位置にカメラを設置し、本手法に適用したところ、ByteTrackのみではオクルージョンに対して4割程度しか対応できていなかったが、9割程度対応が可能になった。
+ ByteTrackのみの追跡と比較して、MOTA、IDF1がともに向上した。
+ 30フレームの処理に20秒ほどかかる。そのため、現状ではリアルタイム性がない。RAFTを別の手法に変えると、実行速度の向上が期待できる。

## Usage
sample_entire_raft.pyと同じ階層で以下の通りに実行。
```
python3 sample entire raft.py video –path ./videos/20200914 2 cam1.avi -n yolox-x -cpretrained/yolox x.pth –match thresh 0.8 –track thresh 0.7 –fp16 –fuse –save result –raft model RAFT/models/raft-sintel.pth
–nms 0.45 –conf 0.1
```

実行に必要なライブラリ等は後にまとめる予定。
