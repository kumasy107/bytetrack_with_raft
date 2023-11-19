import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

#from pytorch_memlab import profile


def load_params(params_path):
   state_dict = torch.load(params_path, map_location=torch.device('cpu'))
   return state_dict

def load_image(img):
    #img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(flo):
    #img = img[0].permute(1,2,0).cpu().detach().numpy()
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #img_flo = np.concatenate([img, flo], axis=0)
    img_flo = flo.astype('uint8')

    return img_flo[:, :, [2,1,0]]
    
    #cv2.imwrite(f'result/image_{i+1}.png', img_flo[:, :, [2,1,0]])


def demo(args):

    #model = torch.nn.DataParallel(RAFT(args))
    model = RAFT(args)
    
    state_dict = load_params(args.model)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    #model.load_state_dict(load_params(args.model))

    #model = model.module
    model.to(DEVICE)
    model.eval()
   
    frame_count = 0

    # カメラ準備 ###############################################################
    cap = cv2.VideoCapture(args.movie)
    #幅と高さは自分で決める必要がある(960*540ならこれらしい)
    cap_width = 960
    cap_height = 544
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力ファイル
    if args.movie is not None:
        outpath = os.path.splitext(args.movie)[0] + "_dense_flow.mp4"
    else:
        outpath = "camera.mp4"

    writer = cv2.VideoWriter(
        outpath,
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap_fps, (cap_width, cap_height)
    )

    print("start")

    while True:
        frame_count += 1
        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        #optical flow
        if frame_count == 1:
            image1 = load_image(frame)
            continue
        image2 = load_image(frame)
        padder = InputPadder(image2.shape)
        pad_image1, pad_image2 = padder.pad(image1, image2)
        flow_low, flow_up = model(pad_image1, pad_image2, iters=10, test_mode=True)
        #flow_frame = viz(pad_image1, flow_up)
        flow_frame = viz(flow_up)
        writer.write(flow_frame)
        image1 = image2
        #30フレームとごとに進捗を表示
        if(frame_count % 30 == 0):
            print(frame_count,"frames")
    cap.release()
    writer.release()
   
    """
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        # 画像を保存するときに順番に保存されるようにenumerateでindexを追加します。
        images = sorted(images)
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # 引数を追加
            viz(image1, flow_up, i)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument("--movie", type=str, default=None)
    args = parser.parse_args()
    os.makedirs("result", exist_ok=True)
    demo(args)
