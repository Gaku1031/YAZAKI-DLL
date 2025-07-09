# 動画をアップロードして血圧を推定するプログラム（OpenCV DNN版）

import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import signal
import logging
from collections import deque
from scipy.stats import zscore
import csv
from datetime import datetime
from scipy.signal import butter, find_peaks, lfilter
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pywt
import os

logging.basicConfig(filename='output.log', level=logging.INFO)
logger = logging.getLogger()

class BandpassFilter:
    def __init__(self, fs, low_freq, high_freq, order=5):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        # ハイパスフィルタとローパスフィルタの設計
        self.high_b, self.high_a = signal.butter(
            self.order, self.low_freq / (self.fs / 2), btype='high')
        self.low_b, self.low_a = signal.butter(
            self.order, self.high_freq / (self.fs / 2), btype='low')

    def apply(self, signal_in):
        # フィルタの適用
        filtered_signal = signal.filtfilt(self.high_b, self.high_a, signal_in)
        filtered_signal = signal.filtfilt(
            self.low_b, self.low_a, filtered_signal)
        return filtered_signal


class RealTimePPG(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # 解析する動画ファイルのパス
        self.video_path = None
        self.cap = None
        self.video_fps = 30  # デフォルト値（後で動画ファイルから取得して上書き）

        # rPPG関連
        self.rppg_data = []       # プロット用のrPPG信号値
        self.time_data = []       # プロット用の時刻

        # ピーク情報（プロット用）
        self.all_peaks_time = []
        self.all_peaks_value = []

        # RRIと推定血圧を逐次的に記録するリスト
        # → RRIの数とSBP, DBPの数が常に一致
        #   [(time, RRI, SBP, DBP), ...]
        self.rri_bp_records = []

        # 10秒区間ごとの「平均血圧」を管理するリスト
        # [(start_t, end_t, avg_SBP, avg_DBP), ...]
        self.segment_estimates = []

        # 実測血圧(フォーム入力)
        self.measured_SBP = None
        self.measured_DBP = None

        # キャリブレーション用パラメータ(初期値)
        self.alpha_s = 1.0
        self.beta_s = 100.0
        self.alpha_d = 1.0
        self.beta_d = 60.0

        # 10秒ごとに平均血圧を更新
        self.segment_duration = 5.0
        # self.segment_duration = 10.0
        # self.segment_duration = 60.0
        self.last_segment_time = 0.0

        # ウィンドウサイズ（POSアルゴリズムで使用）
        self.window_size = 30

        # ROIピクセル保存（夜間用）
        self.roi_pixels_list = deque(maxlen=300)

        # 検出開始時刻（動画再生開始時点）
        self.start_timestamp_str = None  # CSV保存用の開始時刻文字列

        # 10秒区切りでグラフ描画用のリスト
        self.segment_times = []  # 10秒区間の終了時刻
        self.segment_sbps = []   # 上の終了時刻に対応する平均SBP
        self.segment_dbps = []   # 上の終了時刻に対応する平均DBP

        self.initUI()

        # OpenCV DNN Face Detection の初期化
        self.face_net = None
        self.init_face_detector()

        # タイマー設定（動画のフレームを一定周期で読む）
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # POSアルゴリズムで使用するDeque
        self.skin_means = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)

        # Matplotlibの設定
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.rppg_plot, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('rPPG Signal')
        self.ax.set_title('Real-time rPPG Signal')
        self.ax.grid(True)

        # BP表示用のレイアウト
        self.bp_display_layout = QtWidgets.QVBoxLayout()

        # メインレイアウト設定
        main_layout = QtWidgets.QVBoxLayout()

        # --- (A) 上半分: 動画 + グラフ (左) と 10秒BPスクロール (右) ---
        top_layout = QtWidgets.QHBoxLayout()

        # 左側（動画＋グラフ）
        left_top_layout = QtWidgets.QVBoxLayout()
        left_top_layout.addWidget(self.image_label)  # 動画表示用
        left_top_layout.addWidget(self.canvas)       # rPPGグラフ
        top_layout.addLayout(left_top_layout)

        # 右側（10秒平均BPグラフ + スクロールテキスト）
        right_top_layout = QtWidgets.QVBoxLayout()

        # 10秒平均BP用のグラフ
        self.fig_bp, self.ax_bp = plt.subplots()
        self.canvas_bp = FigureCanvas(self.fig_bp)
        right_top_layout.addWidget(self.canvas_bp)

        # BPグラフのライン2本（SBP, DBP）
        self.bp_line_s, = self.ax_bp.plot([], [], 'r-', label='SBP')
        self.bp_line_d, = self.ax_bp.plot([], [], 'b-', label='DBP')
        self.ax_bp.set_xlabel('Time (s)')
        self.ax_bp.set_ylabel('Blood Pressure (mmHg)')
        self.ax_bp.set_title('5-sec Average BP')
        self.ax_bp.legend()
        self.ax_bp.grid(True)

        # スクロールエリア
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setStyleSheet("background-color: white;")
        self.bp_display_layout = QtWidgets.QVBoxLayout(scroll_widget)
        self.scroll_area.setWidget(scroll_widget)
        right_top_layout.addWidget(self.scroll_area)

        top_layout.addLayout(right_top_layout)
        main_layout.addLayout(top_layout)

        # --- (B) 下半分: パラメータ入力, 実測値入力, ボタン類 ---
        bottom_layout = QtWidgets.QGridLayout()

        bottom_layout.addWidget(QtWidgets.QLabel("α_s:"), 0, 0)
        bottom_layout.addWidget(self.alpha_s_edit,        0, 1)
        bottom_layout.addWidget(QtWidgets.QLabel("β_s:"), 0, 2)
        bottom_layout.addWidget(self.beta_s_edit,         0, 3)

        bottom_layout.addWidget(QtWidgets.QLabel("α_d:"), 1, 0)
        bottom_layout.addWidget(self.alpha_d_edit,        1, 1)
        bottom_layout.addWidget(QtWidgets.QLabel("β_d:"), 1, 2)
        bottom_layout.addWidget(self.beta_d_edit,         1, 3)

        # 実測値入力
        bottom_layout.addWidget(QtWidgets.QLabel("Measured SBP:"), 2, 0)
        bottom_layout.addWidget(self.measured_sbp_edit,           2, 1)
        bottom_layout.addWidget(QtWidgets.QLabel("Measured DBP:"), 2, 2)
        bottom_layout.addWidget(self.measured_dbp_edit,           2, 3)

        # ボタン類
        bottom_layout.addWidget(self.select_button, 3, 0)  # 動画ファイル選択
        bottom_layout.addWidget(self.run_button,    3, 1)  # 解析開始
        bottom_layout.addWidget(self.stop_button,   3, 2)  # 停止
        bottom_layout.addWidget(self.save_button,   3, 3)  # CSV保存

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
        self.show()

    def init_face_detector(self):
        """OpenCV DNN Face Detector の初期化"""
        try:
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.face_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                print("OpenCV DNN Face Detector loaded successfully")
            else:
                print(f"Model files not found: {model_file}, {config_file}")
                self.face_net = None
        except Exception as e:
            print(f"Error loading face detector: {e}")
            self.face_net = None

    def initUI(self):
        self.setWindowTitle('PPG + BP Estimation from Video File (OpenCV DNN)')
        self.setGeometry(100, 100, 800, 1200)

        # ボタン
        self.select_button = QtWidgets.QPushButton('Select Video', self)
        self.select_button.clicked.connect(self.select_video_file)

        self.run_button = QtWidgets.QPushButton('Run', self)
        self.run_button.clicked.connect(self.run_analysis)

        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_analysis)

        self.save_button = QtWidgets.QPushButton('Save to CSV', self)
        self.save_button.clicked.connect(self.save_data_to_csv)

        # ラベル（動画表示）
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(800, 480)

        # キャリブレーションパラメータ入力
        self.alpha_s_edit = QtWidgets.QLineEdit(str(self.alpha_s))
        self.beta_s_edit = QtWidgets.QLineEdit(str(self.beta_s))
        self.alpha_d_edit = QtWidgets.QLineEdit(str(self.alpha_d))
        self.beta_d_edit = QtWidgets.QLineEdit(str(self.beta_d))

        # 実測血圧入力
        self.measured_sbp_edit = QtWidgets.QLineEdit("")
        self.measured_dbp_edit = QtWidgets.QLineEdit("")

    def select_video_file(self):
        """ファイル選択ダイアログを開いて動画ファイルを指定"""
        file_dialog = QtWidgets.QFileDialog(self, "Select video file")
        file_dialog.setNameFilters(["Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"])
        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            if os.path.isfile(selected_file):
                self.video_path = selected_file
                print(f"Selected video file: {self.video_path}")

    def run_analysis(self):
        """選択した動画を開いて解析開始"""
        if not self.video_path:
            QtWidgets.QMessageBox.warning(
                self, "No Video Selected", "先に動画ファイルを選択してください。")
            return

        if self.face_net is None:
            QtWidgets.QMessageBox.critical(
                self, "Error", "顔検出モデルが読み込めませんでした。")
            return

        # 入力されたキャリブレーションパラメータを更新
        try:
            self.alpha_s = float(self.alpha_s_edit.text())
            self.beta_s = float(self.beta_s_edit.text())
            self.alpha_d = float(self.alpha_d_edit.text())
            self.beta_d = float(self.beta_d_edit.text())
        except ValueError:
            pass

        # 動画キャプチャを開く
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "動画を開けませんでした。")
            return

        # FPSを取得してタイマー間隔を調整
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps > 0 else 30
        interval_ms = int(1000 / self.video_fps)

        # 解析スタート時点での初期化
        self.reset_analysis()
        self.start_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # タイマー開始
        self.timer.start(interval_ms)

    def stop_analysis(self):
        """解析停止"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def reset_analysis(self):
        """解析結果などを初期化"""
        self.rppg_data.clear()
        self.time_data.clear()
        self.all_peaks_time.clear()
        self.all_peaks_value.clear()
        self.rri_bp_records.clear()
        self.segment_estimates.clear()
        self.segment_times.clear()
        self.segment_sbps.clear()
        self.segment_dbps.clear()
        self.roi_pixels_list.clear()
        self.skin_means.clear()
        self.timestamps.clear()

        self.last_segment_time = 0.0

        # グラフ初期化
        self.ax.clear()
        self.ax.set_title('Real-time rPPG Signal')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('rPPG Signal')
        self.ax.grid(True)
        self.rppg_plot, = self.ax.plot([], [], 'r-')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-3, 3)
        self.canvas.draw()

        # BPグラフ初期化
        self.ax_bp.clear()
        self.ax_bp.set_title('5-sec Average BP')
        self.ax_bp.set_xlabel('Time (s)')
        self.ax_bp.set_ylabel('Blood Pressure (mmHg)')
        self.ax_bp.grid(True)
        self.bp_line_s, = self.ax_bp.plot([], [], 'r-', label='SBP')
        self.bp_line_d, = self.ax_bp.plot([], [], 'b-', label='DBP')
        self.ax_bp.legend()
        self.canvas_bp.draw()

        # スクロールエリアもクリア
        while self.bp_display_layout.count():
            item = self.bp_display_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def update_frame(self):
        """動画からフレームを読み、解析。最後まで読んだら停止。"""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            # 動画終了
            self.stop_analysis()
            return

        # 動画フレームごとの "解析時間(秒)" を計算する
        # → time_dataに使う連続時間として扱う。フレームインデックスから計算してもOK。
        current_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time_sec = current_frame_index / self.video_fps

        # カラー or グレースケール判定（昼夜で分ける）
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 3):
            self.process_nighttime(frame, current_time_sec)
        else:
            self.process_daytime(frame, current_time_sec)

        # 表示用変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        converted_frame = QtGui.QImage(
            frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(converted_frame))

        # 区間ごとの血圧計算チェック
        while current_time_sec - self.last_segment_time >= self.segment_duration:
            self.calculate_segment_bp_estimate()
            self.last_segment_time += self.segment_duration

    def detect_faces_opencv_dnn(self, frame):
        """OpenCV DNNを使用した顔検出"""
        if self.face_net is None:
            return []

        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # 信頼度閾値
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append((startX, startY, endX, endY))
            return faces
        except Exception as e:
            logger.error(f"OpenCV DNN face detection error: {e}")
            return []

    def create_face_roi_from_detection(self, face_box, frame_shape):
        """顔検出結果からROIマスクを作成"""
        startX, startY, endX, endY = face_box
        h, w = frame_shape[:2]
        
        # 顔の中心部分を使用（頬の部分を重点的に）
        face_center_x = (startX + endX) // 2
        face_center_y = (startY + endY) // 2
        face_width = endX - startX
        face_height = endY - startY
        
        # 頬の領域を推定（顔の左右と中央部分）
        roi_width = int(face_width * 0.6)
        roi_height = int(face_height * 0.4)
        
        # ROI座標を計算
        roi_x1 = max(0, face_center_x - roi_width // 2)
        roi_y1 = max(0, face_center_y - roi_height // 2)
        roi_x2 = min(w, face_center_x + roi_width // 2)
        roi_y2 = min(h, face_center_y + roi_height // 2)
        
        # マスクを作成
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (roi_x1, roi_y1), (roi_x2, roi_y2), 255, -1)
        
        return mask

    def process_daytime(self, frame, current_time):
        """昼間カラー映像用の処理（OpenCV DNN版）"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces_opencv_dnn(frame_rgb)

        if faces:
            # 最も大きい顔を選択
            largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            
            # 顔検出結果からROIマスクを作成
            mask = self.create_face_roi_from_detection(largest_face, frame.shape)
            
            # ROIから肌色領域を抽出
            roi = cv2.bitwise_and(frame, frame, mask=mask)
            roi_ycbcr = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
            skin_mask = cv2.inRange(roi_ycbcr, (0, 100, 130), (255, 140, 175))
            skin_pixels = cv2.bitwise_and(roi, roi, mask=skin_mask)
            skin_mean = cv2.mean(skin_pixels, skin_mask)[:3]

            self.skin_means.append(skin_mean)
            self.timestamps.append(current_time)

            if len(self.skin_means) > 30:
                rppg_signal = self.calculate_rppg_signal()
                if len(rppg_signal) > 0:
                    self.rppg_data.append(rppg_signal[-1])
                    self.time_data.append(current_time)
                    self.update_rppg_plot()

    def process_nighttime(self, frame, current_time):
        """夜間(赤外 or 白黒カメラ)映像用の処理（OpenCV DNN版）"""
        # frame が既にグレースケールなら shape=(H, W)
        # そうでない場合は BGR2GRAY
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.detect_faces_opencv_dnn(frame)
        if faces:
            # 最も大きい顔を選択
            largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            
            # 顔検出結果からROIマスクを作成
            mask = self.create_face_roi_from_detection(largest_face, gray.shape)
            
            roi_pixels = gray[mask == 255]
            self.roi_pixels_list.append(roi_pixels)
            self.timestamps.append(current_time)

            if len(self.roi_pixels_list) >= self.window_size:
                rppg_signal = self.calculate_rppg_signal_night()
                if len(rppg_signal) > 0:
                    self.rppg_data.append(rppg_signal[-1])
                    self.time_data.append(current_time)
                    self.update_rppg_plot()

    def calculate_rppg_signal(self):
        """POSベースのrPPG信号抽出（カラー映像向け）"""
        fs = 30
        lowcut = 0.7
        # highcut = 4.0
        highcut = 3.0
        order = 7
        try:
            if len(self.skin_means) <= 10:
                return np.array([])

            skin_means = np.array(list(self.skin_means)[10:])
            if not np.all(np.isfinite(skin_means)):
                return np.array([])

            C = skin_means.T
            frames = C.shape[1]
            L = self.window_size
            if frames < L:
                return np.array([])

            H = np.zeros(frames)
            for f in range(frames - L + 1):
                block = C[:, f:f+L].T
                mu_C = np.mean(block, axis=0)
                mu_C[mu_C == 0] = 1e-8
                C_normed = block / mu_C
                M = np.array([[0, 1, -1], [-2, 1, 1]])
                S = np.dot(M, C_normed.T)
                alpha = np.std(S[0, :]) / (np.std(S[1, :]) + 1e-8)
                P = S[0, :] + alpha * S[1, :]
                P_std = np.std(P) if np.std(P) != 0 else 1e-8
                P_normalized = (P - np.mean(P)) / P_std
                H[f:f + L] += P_normalized

            pulse = -1 * H
            pulse_z = zscore(pulse)
            pulse_baseline = pulse_z - \
                signal.convolve(pulse_z, np.ones(6)/6, mode='same')

            filtered = self.bandpass_filter(
                pulse_baseline, lowcut, highcut, fs, order)
            norm_sig = (filtered - np.mean(filtered)) / \
                (np.std(filtered) + 1e-8)
            return norm_sig
        except Exception as e:
            logger.error("calculate_rppg_signalエラー: %s", e)
            return np.array([])

    def calculate_rppg_signal_night(self):
        """夜間(グレースケール)映像向けrPPG計算"""
        try:
            mean_intensities = [np.mean(roi_pixels)
                                for roi_pixels in self.roi_pixels_list]
            arr = np.array(mean_intensities)
            mean_val = np.mean(arr)
            normalized = arr / mean_val

            # PCA
            pca = PCA(n_components=1)
            pca_signal = pca.fit_transform(normalized.reshape(-1, 1))

            # ウェーブレット
            coeffs = pywt.wavedec(pca_signal[:, 0], 'db4', level=3)
            coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
            reconstructed = pywt.waverec(coeffs, 'db4')

            # バンドパス
            fs = 30
            bpf = BandpassFilter(fs=fs, low_freq=0.7, high_freq=4, order=5)
            filtered = bpf.apply(reconstructed)
            norm_sig = (filtered - np.mean(filtered)) / \
                (np.std(filtered) + 1e-8)

            # 平滑化
            wl = min(25, len(norm_sig) - 1 if len(norm_sig) %
                     2 == 0 else len(norm_sig))
            smoothed = savgol_filter(norm_sig, window_length=wl, polyorder=4)
            return smoothed
        except Exception as e:
            logger.error("calculate_rppg_signal_nightエラー: %s", e)
            return np.array([])

    def bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')

        # lfilterを使用してフィルタリングを行う
        y = lfilter(b, a, data)

        return y

    def update_rppg_plot(self):
        """rPPG波形グラフの更新 + ピーク検出 & 描画"""
        try:
            if len(self.time_data) < 2:
                return

            current_time = self.time_data[-1]
            x_min = max(0, current_time - 10)
            x_max = current_time
            self.ax.set_xlim(x_min, x_max)

            time_array = np.array(self.time_data)
            rppg_array = np.array(self.rppg_data)

            # 表示用マスク
            mask = (time_array >= x_min) & (time_array <= x_max)
            visible_times = time_array[mask]
            visible_data = rppg_array[mask]

            if len(visible_data) > 0:
                y_min, y_max = np.min(visible_data), np.max(visible_data)
                y_margin = (y_max - y_min) * 0.1
                if y_min == y_max:
                    y_margin = abs(y_min) * 0.1 if y_min != 0 else 0.1
                self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

            self.rppg_plot.set_data(visible_times, visible_data)

            # 全区間でのピーク検出
            all_peaks, _ = find_peaks(rppg_array, distance=10)
            peak_times_all = time_array[all_peaks]
            peak_values_all = rppg_array[all_peaks]

            # グローバルに保存
            self.all_peaks_time = peak_times_all.tolist()
            self.all_peaks_value = peak_values_all.tolist()

            # 可視範囲のピークだけ再描画
            in_view_mask = (peak_times_all >= x_min) & (
                peak_times_all <= x_max)
            view_peak_times = peak_times_all[in_view_mask]
            view_peak_values = peak_values_all[in_view_mask]

            # 既存ピークマーカーを削除し、新規に描画
            for line in self.ax.lines[1:]:
                line.remove()
            self.ax.plot(view_peak_times, view_peak_values, 'ro')

            self.canvas.draw()
        except Exception as e:
            logger.error("update_rppg_plotエラー: %s", e)

    def calculate_segment_bp_estimate(self):
        """
        直近 segment_duration 秒間に検出された"すべての"ピークから
        RRI→推定SBP/DBPの平均を求める。
        """
        segment_start = self.last_segment_time
        segment_end = segment_start + self.segment_duration

        # 今回の区間内ピークを取得
        peak_times_in_segment = [
            t for t in self.all_peaks_time if segment_start <= t < segment_end
        ]
        peak_times_in_segment.sort()

        if len(peak_times_in_segment) < 2:
            return

        segment_sbps = []
        segment_dbps = []

        for i in range(len(peak_times_in_segment) - 1):
            t1 = peak_times_in_segment[i]
            t2 = peak_times_in_segment[i+1]
            rri = t2 - t1

            # 血圧推定
            sbp = self.alpha_s * rri + self.beta_s
            dbp = self.alpha_d * rri + self.beta_d

            measure_time = t2
            self.rri_bp_records.append((measure_time, rri, sbp, dbp))

            segment_sbps.append(sbp)
            segment_dbps.append(dbp)

        avg_sbp = np.mean(segment_sbps)
        avg_dbp = np.mean(segment_dbps)
        self.segment_estimates.append(
            (segment_start, segment_end, avg_sbp, avg_dbp))

        self.segment_times.append(segment_end)
        self.segment_sbps.append(avg_sbp)
        self.segment_dbps.append(avg_dbp)

        self.update_bp_plot()

        # テキスト表示
        seg_end_sec = int(segment_end)
        text = f"{seg_end_sec}秒: {avg_sbp:.2f} / {avg_dbp:.2f}"
        new_label = QtWidgets.QLabel(text)
        self.bp_display_layout.addWidget(new_label)

    def update_bp_plot(self):
        """区切り平均BPグラフ更新"""
        self.bp_line_s.set_data(self.segment_times, self.segment_sbps)
        self.bp_line_d.set_data(self.segment_times, self.segment_dbps)

        self.ax_bp.relim()
        self.ax_bp.autoscale_view()
        self.canvas_bp.draw()

    def save_data_to_csv(self):
        """
        RRIごとの推定値と、属する区間の平均BPなどをCSVに書き出す
        """
        try:
            # 実測血圧があれば取得
            try:
                self.measured_SBP = float(self.measured_sbp_edit.text())
                self.measured_DBP = float(self.measured_dbp_edit.text())
            except ValueError:
                self.measured_SBP = None
                self.measured_DBP = None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"BP_estimation_opencv_{timestamp}.csv"

            # 10秒区切りリストを end_time 昇順にソート
            seg_list = sorted(self.segment_estimates, key=lambda x: x[1])
            # seg_listの各要素: (seg_start, seg_end, avg_sbp, avg_dbp)

            # RRIレコードを time 昇順にソート
            rri_list = sorted(self.rri_bp_records, key=lambda x: x[0])
            # rri_listの各要素: (time, rri, sbp, dbp)

            def get_segment_avg(time_value):
                for (start_t, end_t, s_sbp, s_dbp) in seg_list:
                    if start_t <= time_value < end_t:
                        return s_sbp, s_dbp
                return "", ""

            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Time(s)",
                    "RRI(s)",
                    "EstSBP",
                    "EstDBP",
                    "MeasuredSBP",
                    "MeasuredDBP",
                    "ErrorSBP",
                    "ErrorDBP",
                    f"{int(self.segment_duration)}sec_AvgSBP",
                    f"{int(self.segment_duration)}sec_AvgDBP"
                ])

                for (t, rri, sbp, dbp) in rri_list:
                    row_avg_sbp, row_avg_dbp = get_segment_avg(t)

                    if (self.measured_SBP is not None) and (self.measured_DBP is not None):
                        err_sbp = sbp - self.measured_SBP
                        err_dbp = dbp - self.measured_DBP
                    else:
                        err_sbp = ""
                        err_dbp = ""

                    writer.writerow([
                        f"{t:.3f}",
                        f"{rri:.3f}",
                        f"{sbp:.2f}",
                        f"{dbp:.2f}",
                        f"{self.measured_SBP if self.measured_SBP else ''}",
                        f"{self.measured_DBP if self.measured_DBP else ''}",
                        f"{err_sbp}",
                        f"{err_dbp}",
                        f"{row_avg_sbp:.2f}" if isinstance(
                            row_avg_sbp, float) else row_avg_sbp,
                        f"{row_avg_dbp:.2f}" if isinstance(
                            row_avg_dbp, float) else row_avg_dbp
                    ])

            print(f"Data saved to {filename}")

        except Exception as e:
            logger.error("save_data_to_csvエラー: %s", e)
            print("Error saving data:", e)

    def closeEvent(self, event):
        self.stop_analysis()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimePPG()
    window.show()
    sys.exit(app.exec_())