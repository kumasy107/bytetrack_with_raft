#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("RAFT/core")

from raft import RAFT
from utils.utils import InputPadder
import torch

DEVICE = "cuda"

import time
import argparse

import cv2
import numpy as np
import os

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import warnings

warnings.filterwarnings("ignore")

# 注目するオブジェクトを指定
attention = []

# 車両を追跡中か否かの判定に利用する
check_list = []

# 引数の解析
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path",
        default="./videos/20200914_2_cam1.avi",
        help="path to images or video",
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument(
        "--track_thresh", type=float, default=0.7, help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.8,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--min-box-area", type=float, default=10, help="filter out tiny boxes"
    )
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )

    # RAFT
    parser.add_argument("--raft_model", help="restore checkpoint")
    # parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    return parser


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# RAFTで用いる関数
def load_params(params_path):
    state_dict = torch.load(params_path, map_location=torch.device("cpu"))
    return state_dict


# RAFTで用いる関数
def load_image(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# trackingの結果をファイルに書き込んで出力
def write_results(filename, results):
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    s=round(score, 2),
                )
                f.write(line)
        f.close()


class Predictor(object):
    def __init__(
        self, model, exp, trt_file=None, decoder=None, device="cpu", fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes  # 80
        self.confthre = exp.test_conf  # クラス確率のしきい値（多分）。0.01
        self.nmsthre = exp.nmsthre  # IoUがこのしきい値より大きいBBは同じ車両を検知しているとみなす（多分）。0.65。
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    # YOLOXの動作は基本ここで行われる
    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # 前処理
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            if timer is not None:
                timer.tic()
            # 検出の実行
            outputs = self.model(img)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            # self.nmsthre:0.65(default)
            # 後処理
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


def imageflow_demo(predictor, args):
    # 引数解析 #################################################################
    # カメラ
    cap = cv2.VideoCapture(args.path)
    cap_width = 1920  # 動画の幅
    cap_height = 1080  # 動画の高さ
    # capに幅と高さを設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    # 動画からfpsを調べ、capに設定する
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    # trackingの結果を出力するファイルの名前
    os.makedirs("YOLOX_outputs/yolox_x/track_result", exist_ok=True)
    result_filename = "YOLOX_outputs/yolox_x/track_result/{}_matchthresh{}_trackthresh{}.txt".format(
        args.path.split("/")[-1].split(".")[0], args.match_thresh, args.track_thresh
    )

    # バウンディングボックスを書き込んだ動画の名前
    os.makedirs("multitracking_result", exist_ok=True)
    save_path = "multitracking_result/{}_matchthresh{}_trackthresh{}.mp4".format(args.path.split("/")[-1].split(".")[0], args.match_thresh, args.track_thresh)
    logger.info(f"video save_path is {save_path}")
    # 上記の動画を出力するのに用いる
    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), cap_fps, (cap_width, cap_height)
    )

    # 交差点外で停止している車両を追跡の対象から外すためのマスク。
    # Mask_cam = cv2.imread('out_mask.png') #マスク画像なぜか逆
    # Mask_cam = cv2.imread("out_mask_2_1.png")
    Mask_cam = np.full((1080, 1920, 3), 255)  # フィルタアウトしないとき

    tracker = BYTETracker(args, frame_rate=30)  # ByteTrackerクラスのインスタンス生成

    timer = Timer()
    frame_id = 0  # フレーム番号
    results = []

    # RAFT

    hajime = time.time()

    print("start")

    # 動画のフレーム数分だけ繰り返す。（実験データは9400フレーム）
    while True:

        # カメラキャプチャ ################################################
        ret, frame = cap.read()  # frameに動画のフレームが入る
        if not ret:
            break

        # RAFT
        # フレームの各ピクセルについてオプティカルフローを計算する
        raft_model = RAFT(args)

        state_dict = load_params(args.raft_model)

        from collections import OrderedDict

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if "module" in k:
                k = k.replace("module.", "")
            new_state_dict[k] = v

        raft_model.load_state_dict(new_state_dict)

        raft_model.to(DEVICE)
        raft_model.eval()

        # メモリの都合上，（オプティカルフローを求めるための画像の）高さと幅を1/2する（つまり総ピクセル数は1/4になる）
        raft_frame = cv2.resize(frame, (int(cap_width / 2), int(cap_height / 2)))

        # 最初のフレームではオプティカルフローを計算できないのでflo=Noneとする
        if frame_id == 0:
            image1 = load_image(raft_frame)
            flo = None
        else:
            # image1（前フレーム）とimage2（現フレーム）からオプティカルフローを計算
            image2 = load_image(raft_frame)
            # おまじないのようなものと思ってよい
            padder = InputPadder(image2.shape)
            pad_image1, pad_image2 = padder.pad(image1, image2)
            flow_low, flow_up = raft_model(
                pad_image1, pad_image2, iters=10, test_mode=True
            )
            flo = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
            # 高さと幅を1/2にしていたため、元に戻す。
            flo = cv2.resize(flo, (1920, 1080))
            # 高さと幅を2倍したため、オプティカルフローも2倍する
            flo = flo * 2.0
            # 現フレームの画像をimage1にする
            image1 = image2

        # 推論実施 ########################################################
        # Object Detection
        outputs, img_info = predictor.inference(frame, timer)  # outputsに物体検出の結果が入力される
        if outputs[0] is not None:
            class_list = []
            for i in range(len(outputs[0])):
                class_list.append(outputs[0][i][6].item())
            # online_targetsに追跡結果が出力される。lost_maskはオプティカルフローを可視化したいときなどに使っていたので追跡結果に影響はない
            online_targets, lost_mask = tracker.update(
                outputs[0],
                [img_info["height"], img_info["width"]],
                exp.test_size,
                Mask_cam,
                class_list,
                frame,
                flo,
                predictor,
            )
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for online_target in online_targets:
                tlwh = online_target.tlwh  # 追跡した車両の座標
                track_id = online_target.track_id  # 追跡した車両のID
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    # BBが非常に小さい場合は追跡しない
                    online_tlwhs.append(tlwh)
                    online_ids.append(track_id)
                    online_scores.append(online_target.score)

            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            # BBなどの描画
            online_im = plot_tracking(
                img_info["raw_img"],
                online_tlwhs,
                online_ids,
                frame_id=frame_id + 1,
                fps=1.0 / timer.average_time,
            )
        else:
            timer.toc()
            online_im = img_info["raw_img"]

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            online_im = cv2.add(online_im, lost_mask)  # lost_maskを出力する動画に描き加える
            writer.write(online_im)  # 動画の出力
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        frame_id += 1
        # 30フレームとごとに進捗を表示
        if frame_id % 30 == 0:
            tochu = time.time()
            fps_now = frame_id / (tochu - hajime)
            print(frame_id, "frames・・・fps=", fps_now)

    if args.save_result:
        write_results(result_filename, results)
    cap.release()
    writer.release()
    syuuryou = time.time()
    print("time=", syuuryou - hajime)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
