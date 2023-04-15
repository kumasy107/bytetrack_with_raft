import copy

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from tracker import matching

from .basetrack import BaseTrack, TrackState
from .disandcol import disandcol_sim
from .kalman_filter import KalmanFilter
# YOLOXの特徴量から類似度を計算しようとしたものの、良い結果は得られなかった。
# from .sim_by_yolox import yolox_sim
from .kalman_for_flow import kalman_of

model = LinearRegression()  # 線形回帰
flag_start = 0

"""
下記のように記述することで出力する動画に長方形を書き込んだりできる。
lost中に予測しているBBの座標を出力したりするのに用いた。
mask = cv2.rectangle(
                mask,
                (int(atrack.tlwh[0]), int(atrack.tlwh[1])),
                (int(atrack.tlwh[0]+atrack.tlwh[2]), int(atrack.tlwh[1]+atrack.tlwh[3])),
                (255,255,255),
                thickness=2,
                )
"""


def queue(src, a, num):
    # 直近10フレームのOF値をキューを用いて記録する
    dst = np.roll(src, -num)
    dst[-1] = a
    return dst


def area_pre(d, img):
    # [ymin,xmin,ymax,xmax]
    # 交差点外の車両を追跡の対象から除くために用いる。
    if d[0] < 0:
        d[0] = 0
    if d[2] >= img.shape[0]:
        d[2] = img.shape[0] - 1
    if d[1] < 0:
        d[1] = 0
    if d[3] >= img.shape[1]:
        d[3] = img.shape[1] - 1
    if (
        int(img[int(d[2]), int(d[3]), 0]) != 0
        and int(img[int(d[2]), int(d[3]), 1]) != 0
        and int(img[int(d[2]), int(d[3]), 2]) != 0
    ):
        return True
    if (
        int(img[int(d[2]), int(d[1]), 0]) != 0
        and int(img[int(d[2]), int(d[1]), 1]) != 0
        and int(img[int(d[2]), int(d[1]), 2]) != 0
    ):
        return True
    if (
        int(img[int(d[2]), int((int(d[1]) + int(d[3])) / 2), 0]) != 0
        and int(img[int(d[2]), int((int(d[1]) + int(d[3])) / 2), 1]) != 0
        and int(img[int(d[2]), int((int(d[1]) + int(d[3])) / 2), 2]) != 0
    ):
        return True
    else:
        return False


def area_dup(atlwh, btlwh):
    # 2つのバウンディングボックスが重複しているかどうかを判定する。
    # 重複している場合は面積を、していない場合は0を返す。
    # また、その重複がオクルージョンを引き起こすような重複である場合はflagをTrueとしている。
    # オクルージョンを引き起こすような重複については熊本の卒業論文を参照。
    dx = min(atlwh[0] + atlwh[2], btlwh[0] + btlwh[2]) - max(atlwh[0], btlwh[0])
    dy = min(atlwh[1] + atlwh[3], btlwh[1] + btlwh[3]) - max(atlwh[1], btlwh[1])

    a_y1 = atlwh[1]
    a_y2 = atlwh[1] + atlwh[3]
    b_y1 = btlwh[1]
    b_y2 = btlwh[1] + btlwh[3]
    flag = False
    if a_y2 < b_y2 and (a_y1 + a_y2) / 2 > b_y1:
        flag = True
    if (dx > 0) and (dy > 0):
        return dx * dy, flag
    else:
        return 0, flag


def frame_out(tlwh, flow):
    # フレームに写っている車両がフレームの外に出ようとしているときTrueを返す。
    # 例えばフレームの左端に写っている車両が左向きに走行していれば、フレームの外に出ようとしていると考えられる。
    diff = np.sqrt(pow(flow[0], 2) + pow(flow[1], 2))  # 速さ
    argument = np.degrees(np.arctan2(-flow[1], flow[0]))  # ベクトルの偏角
    if diff < 1:
        # 車両がほとんど停止している場合はフレームの外に出ようとしていないと判定する。
        return False
    if tlwh[0] < 5:
        if argument < -150 or argument > 150:
            return True
    if tlwh[0] + tlwh[2] > 1915:
        if -30 < argument < 30:
            return True
    if tlwh[1] < 5:
        if 60 < argument < 120:
            return True
    if tlwh[1] + tlwh[3] > 1075:
        if 60 < argument < 120:
            return True
    return False


def res_flow(bboxes):
    # BBの座標から、オプティカルフローを求める。
    # 引数は複数のBB
    bbox = copy.deepcopy(bboxes)
    # BBの情報を格納
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    # なぜか3倍する必要があった
    x1 *= 3
    y1 *= 3
    x2 *= 3
    y2 *= 3
    # はみ出ている部分の形を整えるために使う
    x1_limit = np.zeros(len(x1))
    x2_limit = np.full(len(x2), 1920)
    y2_limit = np.full(len(y2), 1080)
    # フレームからはみ出ている部分を整える
    x1 = np.maximum(x1, x1_limit)
    y1 = np.maximum(y1, x1_limit)  # 0より小さい値は0にする
    x2 = np.minimum(x2, x2_limit)
    y2 = np.minimum(y2, y2_limit)
    # BBの内側5割のオプティカルフローのみを使う
    x1_temp = copy.deepcopy(x1)
    y1_temp = copy.deepcopy(y1)
    x1 = (3 * x1 + x2) / 4.0
    x2 = (x1_temp + 3 * x2) / 4.0
    y1 = (3 * y1 + y2) / 4.0
    y2 = (y1_temp + 3 * y2) / 4.0
    # int型に直す
    x1 = np.array(x1, dtype="int64")
    y1 = np.array(y1, dtype="int64")
    x2 = np.array(x2, dtype="int64")
    y2 = np.array(y2, dtype="int64")

    return x1, y1, x2, y2


def resize_img(tlwh):
    # 入力されたtlwhから、フレーム内に収まるよう整形されたtlbrを返す。
    x1 = max(int(tlwh[0]), 0)
    x2 = min(int(tlwh[0] + tlwh[2]), 1920)
    y1 = max(int(tlwh[1]), 0)
    y2 = min(int(tlwh[1] + tlwh[3]), 1080)
    return x1, x2, y1, y2


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, flow):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score  # クラス確率
        self.tracklet_len = 0  # 追跡しているフレーム数
        # self.flow_log = np.zeros((10,2))
        # self.flow_log[:,:] = np.nan
        self.flow_new = flow  # さいしんのflow
        self.flow_pred = flow  # 予想するflow
        self.stop = False  # オクルージョンを検出したらTrue
        self.flow_pred_list = np.array([flow])
        self.area = self.tlwh[2] * self.tlwh[3]  # BBの面積
        self.frame_out = False  # フレームアウトを検知したらTrue
        self.keep_mean = self.tlwh_to_xyah(self._tlwh)  # stopした時点でのBBを保持する

        self.dup = -1  # オクルージョンを受けていると判定した場合、オクルージョンの原因となっている車両を保持する

    def predict(self):
        # 本プログラムでは使っていない
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        # t-1フレームでのBBの情報からtフレームでのBBを予測する。
        if len(stracks) > 0:
            # BBの情報(xyah,vx,vy,va,vh)
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            # 共分散とか
            multi_covariance = np.asarray([st.covariance for st in stracks])
            # 各車両のステータス(追跡中かどうかなど)
            multi_state = np.asarray([st.state for st in stracks])
            # カルマンフィルタを用いて予測
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            # 予測結果を各車両のmean,covなどに格納する
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_by_flow(stracks):
        # オクルージョンを検出した場合はこっちでtフレームのBBを予測
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            multi_state = np.asarray([st.state for st in stracks])
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # stracks[i].mean = mean
                # BBの形は信頼できないので更新しない
                # オプティカルフロー（OF）の値を用いてBBの座標を移動させる。
                stracks[i].mean[0] += stracks[i].flow_pred[0]
                stracks[i].mean[1] += stracks[i].flow_pred[1]
                if stracks[i].state < 2:
                    # 追跡中の場合はkeep_meanの座標もOFを用いて移動させる
                    stracks[i].keep_mean[0] += stracks[i].flow_pred[0]
                    stracks[i].keep_mean[1] += stracks[i].flow_pred[1]
                # stracks[i].covariance = cov

    # predict by flow
    def flow_predict(stracks, frame_id):
        if len(stracks) > 0:
            for st in stracks:
                if st.state == TrackState.Lost:
                    # 追跡できていない場合は，予測値で更新
                    st.flow_pred_list = queue(st.flow_pred_list, st.flow_pred, 2)
                pred_x, pred_y = kalman_of(st.flow_pred_list)
                st.flow_pred = [pred_x, pred_y]

    # predict by flow
    def flow_predict_before(stracks, frame_id):
        if len(stracks) > 0:
            # 回帰分析に用いる
            f_list = np.array([list(range(0, 10))]).T
            f_list2 = np.array([list(range(0, 106))]).T
            for i, st in enumerate(stracks):
                if st.state == TrackState.Lost:
                    # 追跡が途絶えている場合は途絶える前に予測しておいたOF値を用いる
                    # 最大値はtrack_buffer+14らしい
                    st.flow_pred = st.flow_pred_list[frame_id - st.end_frame + 14]
                else:
                    # 回帰分析
                    # 直近5フレームを除いた10フレーム(6フレーム前から15フレーム前まで)で回帰分析を行い，後のOF値を予測する
                    # 回帰分析の結果は(基本的には)lostしてから利用する
                    x_temp = kalman_of.kalman_filter_for_of(
                        np.array(st.flow_log[:10, 0])
                    )  # 10フレームのOFをカルマンフィルタに通してノイズの影響を小さくする（しなくてもいいかも）
                    model.fit(f_list, x_temp)  # 10フレームのOFから学習（？）
                    x_pred = model.predict(f_list2)  # 90フレーム先までのOFを予測
                    # 同様の操作をy方向（フレーム高さ方向）でも行う。
                    y_temp = kalman_of.kalman_filter_for_of(
                        np.array(st.flow_log[:10, 1])
                    )
                    model.fit(f_list, y_temp)
                    y_pred = model.predict(f_list2)
                    x_pred = np.array(x_pred)
                    y_pred = np.array(y_pred)
                    st.flow_pred_list = np.array(
                        [x_pred[:, 0], y_pred[:, 0]]
                    ).T  # 90フレーム先までのOFを格納
                    st.flow_pred = st.flow_pred_list[14]  # tフレーム目のOFはflow_predに格納する

    def activate(self, kalman_filter, frame_id, image):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # self.score_log = self.score
        self.stop = False
        self.flow_pred_list = np.array([self.flow_new])
        self.area_log = self.area
        self.frame_out = False
        self.dup = -1
        x1, x2, y1, y2 = resize_img(self.tlwh)
        self.crop_image = image[y1:y2, x1:x2, :]
        self.keep_mean = self.mean

    def re_activate(self, new_track, frame_id, image, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        # self.score_log = self.score
        self.area = new_track.area
        self.area_log = self.area
        self.stop = False
        self.frame_out = False
        self.flow_pred_list = np.array([self.flow_new])
        self.dup = -1
        x1, x2, y1, y2 = resize_img(new_track.tlwh)
        self.crop_image = image[y1:y2, x1:x2, :]
        self.keep_mean = self.mean

    def update(self, new_track, frame_id, image):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1  # 追跡しているフレーム数をインクリメント

        if self.dup != -1 and new_track.area / self.area < 0.95:
            # 重複を受けていて、面積変化率が0.95未満であればオクルージョン検出と判定
            self.stop = True
        elif self.stop == False or new_track.area / self.area > 1.0:
            # 面積変化率が1.0より大きければオクルージョンが生じていないと判定
            self.stop = False
            self.keep_mean = self.mean  # オクルージョンを検出したときに用いるBBの更新
            x1, x2, y1, y2 = resize_img(new_track.tlwh)
            self.crop_image = image[y1:y2, x1:x2, :]  # 画像類似度に用いる画像の更新

        if self.frame_out == True:
            # フレームアウトしていると判定した場合はオクルージョンの影響を考慮しない
            self.stop = False
            self.keep_mean = self.mean

        self.area_log = self.area  # t-1フレームの面積
        self.area = new_track.area  # tフレームの面積

        new_tlwh = new_track.tlwh
        self.frame_out = frame_out(new_tlwh, new_track.flow_pred)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.flow_new = new_track.flow_new
        if self.frame_id - self.start_frame == 1:
            # 追跡を開始した直後は回帰分析にも値いることができるOFがないため、OFのログを最初のOFで埋めておく
            self.flow_pred_list = np.array([self.flow_new])
        # OFの記録の更新
        self.flow_pred_list = queue(self.flow_pred_list, self.flow_new, 2)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlwh_for_stop(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.keep_mean is None:
            return self._tlwh.copy()
        ret = self.keep_mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    # ByteTrackの操作は基本的にここで行われる

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.refind_cand_stracks = []

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1  # 対応付けできなかったBBを新規車両として追跡するかを判定するしきい値
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = (
            self.buffer_size
        )  # lostした後再追跡を試みるフレーム数。今は直接90としており、ここの値は無視している
        self.kalman_filter = KalmanFilter()

    def update(
        self,
        output_results,
        img_info,
        img_size,
        mask_filter,
        class_list,
        image,
        flo,
        predictor,
    ):
        mask = np.zeros((1080, 1920, 3), dtype="uint8")  # 長方形などを描画するときに用いる。
        self.frame_id += 1
        activated_stracks = []  # 追跡中と判定した車両を格納
        refind_stracks = []  # 再検出と判定した車両を格納
        lost_stracks = []  # 追跡が途絶えていると判定した車両を格納
        removed_stracks = []  # 追跡を終了する車両を格納
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        # 各BBのOFを格納
        target_flow = np.zeros((bboxes.shape[0], 2))

        if flo is not None:
            x1, y1, x2, y2 = res_flow(bboxes[:, :4])
            flo = np.array(flo)
            for i in range(len(x1)):
                bb_flow = flo[y1[i] : y2[i], x1[i] : x2[i], :]
                # BB内OFの代表値
                x_med = np.median(bb_flow[:, :, 0])
                y_med = np.median(bb_flow[:, :, 1])
                target_flow[i, 0] = x_med
                target_flow[i, 1] = y_med
        flows = target_flow

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # 指定クラスのみ出力
        remain_cls = []
        for index, cls in enumerate(class_list):
            if cls == 2 or cls == 3 or cls == 5 or cls == 7:
                if area_pre(
                    [
                        bboxes[index, 1],
                        bboxes[index, 0],
                        bboxes[index, 3],
                        bboxes[index, 2],
                    ],
                    mask_filter,
                ):
                    remain_cls.append(True)
                    continue
            remain_cls.append(False)

        # 新たに検出した車両をスコアから3つのカテゴリに分類する
        remain_inds = np.logical_and(scores > self.args.track_thresh, remain_cls)
        inds_low = np.logical_and(scores > 0.1, remain_cls)
        inds_high = np.logical_and(scores < self.args.track_thresh, remain_cls)

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        flows_keep = flows[remain_inds]
        flows_second = flows[inds_second]
        """
        self.tracked_stracks:追跡しようとしている車両(追跡中はis_activated=True、開始前はF)
        self.lost_stracks:見失い中(90frameは追跡を試みる)
        self.removed_stracks:これまでに追跡を終了した車両
        matches:追跡中車両とdetectionsのid対応付けを2次元配列で持つ
        u_track:追跡中車両の内、対応付けされなかった追跡中車両の要素番号
        u_detection:検出した車両の内、対応付けされなかった検出車両の要素番号
        unconfirmed:前フレームで検出し、まだ追跡開始していない車両。既存の車両の対応付けをしたあと対応付けが行われる
        """

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, f)
                for (tlbr, s, f) in zip(dets, scores_keep, flows_keep)
            ]
        else:
            detections = []
        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        """ Step 2: First association, with high score detection boxes"""
        # 追跡中の車両と見失い中の車両1つにまとめる(これをプールと呼ぶことにする)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # プールに属する車両の次フレームでのOF値を予測する
        STrack.flow_predict(strack_pool, self.frame_id)
        # プールに属する車両の次フレームでのBBの位置をKalmanFilterを用いて予測する
        by_bb_diff = [track for track in strack_pool if track.stop == False]
        STrack.multi_predict(by_bb_diff)
        # lostしている車両はOFをもとにBBの位置を予測する
        by_flow_diff = [track for track in strack_pool if track.stop == True]
        STrack.multi_predict_by_flow(by_flow_diff)

        # IoUとOF値から対応付け用の2次元配列を生成
        dists = matching.similarity(
            strack_pool, detections, self.frame_id, self.args.mot20
        )
        # 対応付けを行う
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )
        # 対応付けできた車両は追跡中とする
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # 追跡継続
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, image)
                activated_stracks.append(track)
            # 追跡再開
            else:
                track.re_activate(det, self.frame_id, image, new_id=False)
                refind_stracks.append(track)
        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, f)
                for (tlbr, s, f) in zip(dets_second, scores_second, flows_second)
            ]
        else:
            detections_second = []
        # Step2で対応付けできなかった車両の内，前フレームで追跡中だったものを残す
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        # この段階では常に検出スコアを参考にしていなかったため，引数にTrueを入れる（そんな気にしなくていいかも）
        dists = matching.similarity(
            r_tracked_stracks, detections_second, self.frame_id, True
        )
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, image)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, image, new_id=False)
                refind_stracks.append(track)
        # この時点で対応付けできていない追跡中車両はlostとみなす
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                # lostした場合、保持していたkeep_meanをmeanとする
                track.mean = track.keep_mean
                track.mark_lost()
                if track.stop == False:
                    track.stop = True
                lost_stracks.append(track)
        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.similarity(
            unconfirmed, detections, self.frame_id, self.args.mot20
        )
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, image)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        # 画像類似度
        refind_cand_stracks = (
            []
        )  # 「検出したがオクルージョンを受けている車両」と、「その車両にオクルージョンを生じさせている車両が今オクルージョンを生じさせている車両」を保持する
        for track_cand, occ_list in self.refind_cand_stracks:
            if track_cand.dup == -1:
                # オクルージョンを受けていない状態になったとき
                x1, x2, y1, y2 = resize_img(track_cand.tlwh)
                track_cand.crop_image = image[y1:y2, x1:x2, :]  # 車両の画像を保持

                # ret = yolox_sim(track_cand, occ_list, predictor) #YOLOXの特徴量を用いて類似度を計算する
                ret = disandcol_sim(track_cand, occ_list)  # カラーヒストグラムを用いて類似度を計算する
                # retには、occ_listのうちtrack_candと画像類似度の大きかった車両が格納される
                if ret is not None:
                    # 画像類似度が大きかったとしても、IoUやOF類似度が小さければ対応づけない
                    if (
                        1
                        - matching.flow_distance([ret], [track_cand], self.frame_id)[
                            0, 0
                        ]
                        - matching.iou_distance([ret], [track_cand], self.frame_id)[
                            0, 0
                        ]
                        > 0.1
                    ):
                        # しきい値より大きければ対応づけ
                        # 現在のIDをremovedとして、対応づけたIDをreactivateする
                        new_act_list = []
                        activated_stracks = [
                            act
                            for act in activated_stracks
                            if act.track_id != track_cand.track_id
                        ]
                        track_cand.mark_removed()
                        removed_stracks.append(track_cand)
                        ret.re_activate(track_cand, self.frame_id, image, new_id=False)
                        refind_stracks.append(ret)

            else:
                refind_cand_stracks.append([track_cand, occ_list])
        self.refind_cand_stracks = refind_cand_stracks

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            # 検出スコアがしきい値より小さければ追跡を開始しない
            if track.score < self.det_thresh:
                continue

            # 新たに検出した車両がどの車両の影から出てきた車両かを調べる
            dup_max = 0
            dup_max_track = -1
            for track_now in activated_stracks:
                dup_num, cover_flag = area_dup(track.tlwh, track_now.tlwh)
                if dup_max < dup_num and cover_flag:
                    dup_max = dup_num
                    dup_max_track = track_now
            track.dup = dup_max_track
            # 現在lostしている車両に、同じ車両によってlostしている車両がないかをチェック
            if track.dup != -1:
                occ_list = []
                for occ in self.lost_stracks:
                    if occ.state < 2 or occ.dup == -1 or occ.dup.is_activated == False:
                        continue
                    if occ.dup.track_id == track.dup.track_id:
                        occ_list.append(occ)
                self.refind_cand_stracks.append([track, occ_list])
            track.activate(self.kalman_filter, self.frame_id, image)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            # lostしている車両について、予測したBBを出力する
            mask = cv2.rectangle(
                mask,
                (int(track.tlwh[0]), int(track.tlwh[1])),
                (
                    int(track.tlwh[0] + track.tlwh[2]),
                    int(track.tlwh[1] + track.tlwh[3]),
                ),
                (255, 255, 255),
                thickness=2,
            )
            # 一定時間lostにいる場合は追跡をやめる（現在は90フレームに設定）
            # if self.frame_id - track.end_frame > self.max_time_lost
            if self.frame_id - track.end_frame > 90:
                track.mark_removed()
                removed_stracks.append(track)
            # 予測されるBBの座標が完全に画面外になったら追跡を終了する
            elif 0 > track.tlwh[0] + track.tlwh[2] or 1920 < track.tlwh[0]:
                track.mark_removed()
                removed_stracks.append(track)
            elif 0 > track.tlwh[1] + track.tlwh[3] or 1080 < track.tlwh[1]:
                track.mark_removed()
                removed_stracks.append(track)
            # 追跡の対象でない場所に30フレーム以上いた場合は追跡を終了する
            elif (
                self.frame_id - track.end_frame > 30
                and area_pre(
                    [
                        track.tlwh[1],
                        track.tlwh[0],
                        track.tlwh[1] + track.tlwh[3],
                        track.tlwh[0] + track.tlwh[2],
                    ],
                    mask_filter,
                )
                == False
            ):
                track.mark_removed()
                removed_stracks.append(track)

        # 整理
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks, self.frame_id
        )

        # BBの重複が大きい車両を保持
        for atrack in self.tracked_stracks:
            if atrack.stop == True:
                mask = cv2.rectangle(
                    mask,
                    (int(atrack.tlwh[0]), int(atrack.tlwh[1])),
                    (
                        int(atrack.tlwh[0] + atrack.tlwh[2]),
                        int(atrack.tlwh[1] + atrack.tlwh[3]),
                    ),
                    (255, 255, 0),
                    thickness=2,
                )
            dup_max = 0
            dup_max_track = -1
            for btrack in self.tracked_stracks:
                if atrack.track_id == btrack.track_id:
                    continue
                dup_num, cover_flag = area_dup(atrack.tlwh, btrack.tlwh)
                if dup_max < dup_num and cover_flag:
                    dup_max = dup_num
                    dup_max_track = btrack
            atrack.dup = dup_max_track

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        mask = np.zeros((1080, 1920, 3), dtype="uint8")  # 余計なものを出力させないときに使う
        return output_stracks, mask


def joint_stracks(tlista, tlistb):
    # 配列の結合(重複を許さない)
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    # tilstaからtilstbの要素を削除
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, movie_frame_id):
    # 出番少なそう
    pdist = matching.similarity(stracksa, stracksb, movie_frame_id, True)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
