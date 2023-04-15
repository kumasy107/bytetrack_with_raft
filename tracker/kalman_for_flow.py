import numpy as np


def start(initial_xy):
    dt = 1  # 計測間隔
    x = np.array(
        [[initial_xy[0]], [initial_xy[1]], [0.0], [0.0]]
    )  # 初期位置と初期速度を代入した「4次元状態」
    u = np.array([[0.0], [0.0], [0.0], [0.0]])  # 外部要素
    P = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
        ]
    )  # 共分散行列
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # 状態遷移行列
    H = np.array([[1.0, 0.0, 0, 0], [0.0, 1.0, 0.0, 0.0]])  # 観測行列
    R = np.array([[0.1, 0], [0, 0.1]])  # ノイズ
    I = np.identity((len(x)))  # 4次元単位行列

    return x, u, P, F, H, R, I


class FlowPrediction:
    def __init__(self, flow):
        self.x, self.u, self.P, self.F, self.H, self.R, self.I = start(flow)
        self.measurements = np.array([[flow[0], flow[1]]])
        self.measurements_pred = np.array([[flow[0], flow[1]]])

    def kalman_of_pred(self):
        # 予測
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)
        self.x = self.x.tolist()
        self.P = self.P.tolist()

        # 保存
        self.measurements_pred = np.vstack(
            (self.measurements_pred, np.array([[self.x[0][0], self.x[1][0]]]))
        )
        return self.x[0][0], self.x[1][0]

    def kalman_of_update(self, flow):
        self.flow = flow
        # 格納
        self.measurements = np.vstack((self.measurements, self.flow))

        # 更新
        self.Z = np.array([self.flow])
        self.y = self.Z.T - np.dot(self.H, self.x)
        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.y)
        self.P = np.dot((self.I - np.dot(self.K, self.H)), self.P)


def kalman_of(optical_flow_list):
    N = len(optical_flow_list)
    track = FlowPrediction(optical_flow_list[0])
    pred_x, pred_y = track.kalman_of_pred()  # i=1の予測
    for i in range(1, N):
        track.kalman_of_update(optical_flow_list[i])
        pred_x, pred_y = track.kalman_of_pred()  # i+1での予測
    return pred_x, pred_y  # Nでの予測
