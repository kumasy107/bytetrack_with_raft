#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""hist matching."""

import cv2
import numpy as np


# 減色処理を行う
# カラーヒストグラムの比較でしか用いないので、0,1,2,3に減色処理する。
def decreaseColor(img):
    dst = img.copy()

    idx = np.where((0 <= img) & (64 > img))
    dst[idx] = 0
    idx = np.where((64 <= img) & (128 > img))
    dst[idx] = 1
    idx = np.where((128 <= img) & (192 > img))
    dst[idx] = 2
    idx = np.where((192 <= img) & (256 > img))
    dst[idx] = 3

    return dst


# カラーヒストグラムを比較して類似度を計算
def disandcol_sim(tracka, trackb_list):
    if len(trackb_list) < 1:
        return None
    imga = tracka.crop_image
    IMG_SIZE = (240, 240)  # 画像サイズを統一
    target_img = cv2.resize(imga, IMG_SIZE)
    target_img = target_img[20:220, 20:220]  # BBの枠付近は背景を含んでることが多いので無視する
    # 減数処理
    de_target_img = decreaseColor(target_img)
    de_map = (
        de_target_img[:, :, 0]
        + 4 * de_target_img[:, :, 1]
        + 16 * de_target_img[:, :, 2]
    )
    target_count = [0] * 64
    for i in range(64):
        c_target = np.count_nonzero(de_map == i)
        target_count[i] = c_target

    ranking = {}
    for trackb in trackb_list:
        imgb = trackb.crop_image
        comparing_img = cv2.resize(imgb, IMG_SIZE)
        comparing_img = comparing_img[20:220, 20:220]
        de_comparing_img = decreaseColor(comparing_img)
        de_map = (
            de_comparing_img[:, :, 0]
            + 4 * de_comparing_img[:, :, 1]
            + 16 * de_comparing_img[:, :, 2]
        )
        sum_log = 0
        # カラーヒストグラムの比較の実行
        for i in range(64):
            kazu = min(target_count[i], np.count_nonzero(de_map == i))
            sum_log += kazu
        ret = sum_log / de_map.size
        ranking[trackb] = ret
    # 類似度の大きい順にソート
    ranking2 = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    # 一番大きい画像類似度がしきい値より大きければ、その画像類似度となった車両を似ている車両と判定する。
    if ranking2[0][1] > 0.6:
        return ranking2[0][0]
    else:
        return None
