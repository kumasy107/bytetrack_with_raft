import math

import lap
import numpy as np
import scipy
from tracker import kalman_filter
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist

exp_list = []
iou_list = []


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


# 線形割り当て
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    # 割当てコスト(cost)、どの列にどの行が当てられるかをしめす配列(x)、その逆(y)
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs, alost):
    # IoUを計算
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float),
    )
    for i in range(len(ious)):
        ious[i] *= alost[i]
        ious[i] = alost[i] - ious[i]
    return ious


def iou_distance(atracks, btracks, frame_id):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    alost = []
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        # lostした時間が長いほど，IoUの重みを小さくする
        for track in atracks:
            # 追跡中はIoUの重みを0.8に固定。
            if track.is_activated:
                alost.append(0.8)
            elif track.frame_out:
                alost.append(0.8)
            # 追跡が途絶えているときは、追跡が途絶える時間が長いほど重みを小さくする
            else:
                alost.append((90.0 - frame_id + track.end_frame) / 180.0)
        btlbrs = [track.tlbr for track in btracks]
    # IoUの計算
    alost = np.array(alost, dtype="float64")
    _ious = ious(atlbrs, btlbrs, alost)
    cost_matrix = _ious
    return cost_matrix


def flow_sim(aflows, bflows):
    # OFを用いた類似度の計算
    if len(aflows) == 0 or len(bflows) == 0:
        return [[0.0]]
    aflows = np.array(aflows)
    bflows = np.array(bflows)
    cost_matrix = []

    for deg, x, y, lost in aflows:
        # プール車両の予測される速度と，検出車両の予測される速度の類似度
        len_sim = np.sqrt(
            pow(bflows[:, 1] - x, 2) + pow(bflows[:, 2] - y, 2)
        )  # ユークリッド距離
        len_sim = np.exp(-len_sim / 200)  # 正規化
        arg_sim = []
        # プール車両の予測される偏角と，検出車両の予測される偏角の類似度
        for bdeg, bdif in zip(bflows[:, 0], bflows[:, 3]):
            if bdif < 1.0:
                # 車が止まっているなら偏角の値が参考にならないので、0.8に固定
                arg_sim.append(0.8)
            else:
                # 動いているならcos類似度を計算
                arg_sim.append(abs(math.cos(math.radians(abs(deg - bdeg) / 2.0))))
        arg_sim = np.array(arg_sim, dtype="float64")
        # 追跡が途絶えているフレーム数が長くなるほどOF類似度の重みを大きくする。
        cost_matrix.append(((lost + 90.0) / 180.0) * (1.0 - len_sim * arg_sim))
    cost_matrix = np.array(cost_matrix, dtype="float64")
    # 小さい=類似度が高いとしたいらしい
    return cost_matrix


def flow_distance(atracks, btracks, frame_id):
    """
    Compute cost based on OpticalFlow
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    aflows = []
    bflows = []
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        # いつ発生すんの
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.flow_pred for track in atracks]
        btlbrs = [track.flow_pred for track in btracks]
        # プール車両の予測される座標位置と偏角,追跡中か否かの情報を取得
        for track in atracks:
            x_flow = track.flow_pred[0]  # x方向のOF
            y_flow = track.flow_pred[1]  # y方向のOF
            diff = np.sqrt(pow(x_flow, 2) + pow(y_flow, 2))  # ベクトルの大きさ
            degree = np.degrees(np.arctan2(-y_flow, x_flow))  # ベクトルの偏角
            tc_x = track.tlwh[0] + track.tlwh[2] / 2.0  # BBの中点のx座標
            tc_y = track.tlwh[1]  # BBの上辺のy座標
            # lostした時間が長いほど，OFの重みを大きくする
            # 追跡中の車両の重みは0.2（になるようにlostを-54に固定）
            if track.is_activated:
                lost_frame = -54
            elif track.frame_out:
                lost_frame = -54
            # 追跡が途絶えている場合は途絶えているフレーム数を保持する
            else:
                lost_frame = frame_id - track.end_frame
            aflows.append([degree, tc_x, tc_y, lost_frame])
        # 新たに検出した車両の予測される座標位置と偏角を取得
        for track in btracks:
            x_flow = track.flow_pred[0]
            y_flow = track.flow_pred[1]
            degree = np.degrees(np.arctan2(-y_flow, x_flow))
            diff = np.sqrt(pow(x_flow, 2) + pow(y_flow, 2))
            tc_x = track.tlwh[0] + track.tlwh[2] / 2.0
            tc_y = track.tlwh[1]
            bflows.append([degree, tc_x, tc_y, diff])
    # 類似度を計算
    cost_matrix = flow_sim(aflows, bflows)
    cost_matrix = np.array(cost_matrix, dtype="float64")
    return cost_matrix


def similarity(atracks, btracks, frame_id, mot20):
    # プールに属する車両と新たに検出した車両のすべての組み合わせで、IoUを計算する
    # 返り値は、len(atracks)行len(btracks)列の配列
    coor_dists = iou_distance(atracks, btracks, frame_id)
    # プールに属する車両と新たに検出した車両のすべての組み合わせで、OFを用いた類似度を計算する
    # 返り値は、len(atracks)行len(btracks)列の配列
    flow_dists = flow_distance(atracks, btracks, frame_id)
    dists = coor_dists + flow_dists
    if not mot20:
        dists = fuse_score(dists, btracks)
    return dists


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(
        0.0, cdist(track_features, det_features, metric)
    )  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric="maha",
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
