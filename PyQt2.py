import sys
import os
import cv2
import datetime
import numpy as np
import mediapipe as mp
import csv
import time
from firebase_configuration import FirebaseWorker
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from picamera2 import Picamera2
import psutil


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

save_folder = "outputs"
os.makedirs(save_folder, exist_ok=True)

class ErgoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PERANGKAT ASSESMEN ERGONOMIKA')
        self.resize(1280, 720)

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        #Calibration Update
        self.calibrated = False
        self.calibration_counter = 0
        self.calibration_frames = 15
        self.calibration_data = []
        self.last_time = time.time()
        self.camera_running = False


        self.init_ui()
        
        #hitung latency
        self.latencies = []
        
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        #picamera setup
        self.picam2 = None
        self.picam_timer = QTimer()
        self.picam_timer.timeout.connect(self.update_frame)
        self.frame = None
        
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None
        
        self.latency_csv_file = None
        self.latency_csv_writer = None
        
        #Performcance Logging
        self.sysmon_timer = QTimer()
        self.sysmon_timer.timeout.connect(self.log_system_usage)
        self.sysmon_csv_file = None
        self.sysmon_csv_writer = None


        
        #Konfigurasi Firebase
        self.firebase = FirebaseWorker()
        self.firebase.run()



    def init_ui(self):
        """Setup layout & tombol utama"""
        self.video_btn = QPushButton('Video Input')
        self.image_btn = QPushButton('Image Input')
        self.camera_btn = QPushButton('Play Camera')
        self.snapshot_btn = QPushButton('Save Snapshot')
        self.export_btn = QPushButton('Export Data')
        self.calibrate_btn = QPushButton("Recalibrate")
        self.status_indicator = QLabel("Status")
        self.status_indicator.setFixedHeight(50)
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("background-color: grey; color: white; font-weight: bold; font-size: 16px;")

        # Progress bar kalibrasi
        self.calibration_bar = QProgressBar()
        self.calibration_bar.setRange(0, self.calibration_frames)
        self.calibration_bar.setValue(0)
        self.calibration_bar.setFormat("Calibration: %p%")
        
        self.info_label = QLabel("Info Performance: -")
        self.info_label.setStyleSheet("color: green; font-size: 14px;")

        # Event handler tombol
        self.video_btn.clicked.connect(self.load_video)
        self.image_btn.clicked.connect(self.load_image)
        self.camera_btn.clicked.connect(self.start_camera)
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.calibrate_btn.clicked.connect(self.force_recalibrate)
        self.export_btn.clicked.connect(self.export_data)
        
        # Layout kiri (kontrol)
        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.video_btn)
        vbox_left.addWidget(self.image_btn)
        vbox_left.addWidget(self.camera_btn)
        vbox_left.addWidget(self.snapshot_btn)
        vbox_left.addWidget(self.calibrate_btn)
        vbox_left.addWidget(self.calibration_bar)
    
        self.calibrate_btn.setEnabled(False)
        vbox_left.addWidget(self.export_btn)
        
        # Label video preview
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        
        # Info REBA detail
        self.info_label = QLabel(
                            f"REBA Score:\n\n"
                            f"Neck Flexion:\n"
                            f"Trunk Flexion:\n"
                            f"Right Leg Flexion:\n"
                            f"Left Leg Flexion:\n"
                            f"Right Shoulder Flexion:\n"
                            f"Left Shoulder Flexion:\n"
                            f"Right Elbow Flexion:\n"
                            f"Left Elbow Flexion:\n"
                            f"Right Wrist Flexion:\n"
                            f"Left Wrist Flexion:\n\n"
                            f"Neck Score:\n"
                            f"Trunk Score:\n"
                            f"Leg Score:\n"
                            f"Upper Arm Score:\n"
                            f"Lower Arm Score:\n"
                            f"Wrist Score:\n"
                        )
        
        # Panel log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Layout utama
        hbox_top = QHBoxLayout()
        hbox_top.addLayout(vbox_left)
        hbox_top.addWidget(self.video_label)

        # Create a VBox to hold indicator and info
        vbox_info = QVBoxLayout()
        vbox_info.addWidget(self.status_indicator)
        vbox_info.addWidget(self.info_label)
        hbox_top.addLayout(vbox_info)
        vbox_main = QVBoxLayout()
        vbox_main.addLayout(hbox_top)
        vbox_main.addWidget(self.log_text)
        self.setLayout(vbox_main)
        
        self.set_placeholder()

    def update_log(self, message):
        self.log_text.append(message)

    def log_system_usage(self):
        """Log CPU & RAM usage ke CSV"""
        if self.sysmon_csv_writer:
            cpu_usage = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            self.sysmon_csv_writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cpu_usage,
                ram.percent,
                round(ram.used / (1024 * 1024), 2),
                round(ram.total / (1024 * 1024), 2)
            ])
        self.sysmon_csv_file.flush()

    
    def load_image(self):
        """Proses gambar tunggal"""
        self.stop_capture()
        
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            self.prepare_output_files(for_image=True)
            
            self.processed_frame_count = 0
            self.video_duration = 0
            # simpan frame image ke variabel
            self.frame = cv2.imread(file_path)
            if self.frame is None:
                self.update_log("Failed to load image.")
                return
            # paksa jalankan update_frame sekali
            self.update_frame()
            self.process_image(file_path)
            

    def load_video(self):
        """Buka file video"""
        self.set_placeholder("Memuat Video...")
        QApplication.processEvents() 
        self.stop_capture()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Videos (*.mp4 *.avi)')
        if file_path:
            self.capture = cv2.VideoCapture(file_path)
            self.video_duration = self.capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.capture.get(cv2.CAP_PROP_FPS)
            self.processed_frame_count = 0
            self.latencies = []
            self.prepare_output_files()
            self.timer.start(30)

    def start_camera(self):
        """Start / Stop kamera realtime (Picamera2)"""
        self.camera_running = True
        self.set_placeholder("Loading Kamera...")
        QApplication.processEvents()  
          
        if self.camera_btn.text() == "Play Camera":
            self.camera_btn.setText("Stop Camera")
            self.picam2 = Picamera2()
            #pengaturan resolusi
            config = self.picam2.create_preview_configuration(main={"size": (1280, 720)})
            self.picam2.configure(config)
            self.picam2.start()
            
            
            #if not self.capture.isOpened():
                #self.update_log("Failed to Access Camera")
            
            self.video_duration = None  # Kamera real-time → tidak ada durasi
            self.processed_frame_count = 0
            self.latencies = []    
            
            self.prepare_output_files()
            self.picam_timer.start(30)
            
            self.latencies = []
            self.last_time = time.time()  # Untuk hitung FPS per frame
            self.start_time = time.time()  # Jika nanti mau hitung rata-rata FPS keseluruhan
            self.frame_count = 0           # Untuk total frame count kalau dibutuhkan 
            # Disable/enable tombol
            self.image_btn.setEnabled(False)
            self.video_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.calibrate_btn.setEnabled(True)
            self.snapshot_btn.setEnabled(False)
               
            self.update_log("Camera Started")
        else:
            self.stop_capture()            
            self.image_btn.setEnabled(True)
            self.video_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.snapshot_btn.setEnabled(True)
            self.calibrate_btn.setEnabled(False)
            self.camera_btn.setText("Play Camera")

    def save_snapshot(self):
        """Simpan 1 frame ke file"""
        if self.frame is not None:
            filename = os.path.join(save_folder, f"snapshot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
            cv2.imwrite(filename, self.frame)
            self.update_log(f"Snapshot saved: {filename}")

    def export_data(self):
         if self.csv_file:
             self.csv_file.close()
             self.update_log("CSV data exported and file closed.")

    def prepare_output_files(self, for_image=False):
        """Buat file output (video, CSV, latency log, system log)"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_folder = "outputs"
        os.makedirs(save_folder, exist_ok=True)
        
        # Video writer
        if not for_image:
            if self.capture is not None:
                frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_writer = cv2.VideoWriter(
                    os.path.join(save_folder, f"output_video_{timestamp}.mp4"),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    15,
                    (frame_width, frame_height))
            elif self.picam2 is not None:
                frame_width = 1280
                frame_height = 720
                self.video_writer = cv2.VideoWriter(
                    os.path.join(save_folder, f"output_video_{timestamp}.mp4"),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    15,
                    (frame_width, frame_height))
                    
        # CSV utama (REBA)
        self.csv_file = open(os.path.join(save_folder, f"output_data_{timestamp}.csv"), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp',
            'REBA_SCORE',
            'neck_flexion', 'trunk_flexion',
            'right_leg_flexion', 'left_leg_flexion',
            'right_shoulder_flexion', 'left_shoulder_flexion',
            'right_elbow_flexion', 'left_elbow_flexion',
            'right_wrist_flexion', 'left_wrist_flexion',
            'neck_score', 'trunk_score', 'leg_score',
            'upperarm_score', 'lowerarm_score', 'wrist_score'
        ])
        self.csv_file.flush()
        
        # CSV latency
        self.latency_csv_file = open(os.path.join(save_folder, f"latency_log_{timestamp}.csv"), mode='w', newline='')
        self.latency_csv_writer = csv.writer(self.latency_csv_file)
        self.latency_csv_writer.writerow([
            'timestamp',
            'latency_ms'
        ])
        self.latency_csv_file.flush()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # CSV system usage
        self.sysmon_csv_file = open(os.path.join("outputs", f"system_usage_{timestamp}.csv"), mode='w', newline='')
        self.sysmon_csv_writer = csv.writer(self.sysmon_csv_file)
        self.sysmon_csv_writer.writerow(["timestamp", "cpu_usage_percent", "ram_usage_percent", "ram_used_mb", "ram_total_mb"])
        self.sysmon_timer.start(1000)  # every 1 second

        self.update_log("Output files prepared.")

    def process_image(self, file_path):
        """Proses image tunggal → pose detection + REBA + simpan CSV & image"""
        frame = cv2.imread(file_path)
        if frame is None:
            self.update_log("Failed to load image.")
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results and results.pose_world_landmarks and results.pose_world_landmarks.landmark:
            landmarks = results.pose_world_landmarks.landmark
            result = self.classify_reba(landmarks)
            self.update_info(result)

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

            save_image_path = os.path.join("outputs", f"output_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            self.update_log(f"Image Path {save_image_path}")
            cv2.imwrite(save_image_path, frame)
            self.update_log(f"Processed image saved: {save_image_path}")

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.csv_writer.writerow([
                timestamp,
                result['reba_score'],
                result['neck_flexion'], result['trunk_flexion'],
                result['right_leg_flexion'], result['left_leg_flexion'],
                result['right_shoulder_flexion'], result['left_shoulder_flexion'],
                result['right_elbow_flexion'], result['left_elbow_flexion'],
                result['right_wrist_flexion'], result['left_wrist_flexion'],
                result['neck_score'], result['trunk_score'], result['leg_score'],
                result['upperarm_score'], result['lowerarm_score'], result['wrist_score']
            ])
            self.update_log("REBA score saved to CSV.")

        else:
            self.update_log("No pose detected in image.")

        self.display_image(frame)

    def update_frame(self):
        #ret, frame = self.capture.read()
        #if not ret:
         #   self.stop_capture()
          #  return
        """Loop utama frame update (kamera, video, atau image)"""
        result = None
        start_time = time.time()  # ⬅️ Start timestamp
        
        frame = None
        results = None
        
        # Kamera (picam2)
        if self.picam2 and self.camera_running:
            frame = self.picam2.capture_array()        
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
        
        # Video file    
        elif self.capture:  
            ret, frame = self.capture.read()
            if not ret:
                self.stop_capture()
                return
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
        # Image tunggal (self.frame sudah ada dari load_image)
        elif self.frame is not None:
            frame = self.frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            # setelah sekali diproses, kosongkan supaya tidak berulang
            self.frame = None
            
            
        
        # Jika tidak ada sumber, keluar
        if frame is None:
            return
            
        # ------------------- Pose Processing -------------------
        if results.pose_landmarks:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # konversi ke BGR
            mp_drawing.draw_landmarks(bgr_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if not self.calibrated:
                self.calibrate_posture(results.pose_world_landmarks.landmark)
            # else:
            #     result = self.classify_reba(results.pose_world_landmarks.landmark)
            #     timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            #     self.csv_writer.writerow([
            #         timestamp,
            #         result['reba_score'],
            #         result['neck_flexion'], result['trunk_flexion'],
            #         result['right_leg_flexion'], result['left_leg_flexion'],
            #         result['right_shoulder_flexion'], result['left_shoulder_flexion'],
            #         result['right_elbow_flexion'], result['left_elbow_flexion'],
            #         result['right_wrist_flexion'], result['left_wrist_flexion'],
            #         result['neck_score'], result['trunk_score'], result['leg_score'],
            #         result['upperarm_score'], result['lowerarm_score'], result['wrist_score']])
            #     self.update_info(result)
            result = self.classify_reba(results.pose_world_landmarks.landmark)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.csv_writer.writerow([
                timestamp,
                result['reba_score'],
                result['neck_flexion'], result['trunk_flexion'],
                result['right_leg_flexion'], result['left_leg_flexion'],
                result['right_shoulder_flexion'], result['left_shoulder_flexion'],
                result['right_elbow_flexion'], result['left_elbow_flexion'],
                result['right_wrist_flexion'], result['left_wrist_flexion'],
                result['neck_score'], result['trunk_score'], result['leg_score'],
                result['upperarm_score'], result['lowerarm_score'], result['wrist_score']])
            self.update_info(result)
            
            # Upload high risk ke Firebase
            if result and result['reba_score'] is not None and result['reba_score'] >= 8:
                json_timestamp = datetime.datetime.now().strftime("%H:%M:%S - %d_%m_%Y")
                firebase_data = {"reba_score": result['reba_score'], "timestamp": json_timestamp}
                self.firebase.upload_file(firebase_data)
                
        else:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ------------------- Save Video -------------------    
        if self.video_writer:
            self.video_writer.write(bgr_frame)

        self.display_image(bgr_frame)
        self.frame = bgr_frame
        
         # Logging latency
        self.processed_frame_count += 1
        end_time = time.time()  #  End timestamp
        latency = (end_time - start_time) * 1000  # ms
        self.latencies.append(latency)
        self.latency_csv_writer.writerow([
            datetime.datetime.now().strftime("%H:%M:%S"),
            f"{latency:.2f}"
        ])
        self.latency_csv_file.flush()


    def display_image(self, frame):
        """Tampilkan frame (OpenCV BGR) ke QLabel atau GUI (video_label)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height()))

    def stop_capture(self):
        """Hentikan semua sumber video & simpan summary"""
        if self.timer.isActive():
            self.timer.stop()
        self.camera_running = False
        if self.picam_timer.isActive():
            self.picam_timer.stop()
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.calibrated = False
            
         # Summary latency & throughput
        if self.latency_csv_file and self.latencies:
            avg_latency = np.mean(self.latencies) 
            min_latency = np.min(self.latencies) 
            max_latency = np.max(self.latencies)

            throughput = 0
            if self.video_duration and self.video_duration > 0:
                throughput = self.processed_frame_count / self.video_duration
            elif self.video_duration is None:
                elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
                throughput = self.processed_frame_count / elapsed if elapsed > 0 else 0

            self.latency_csv_writer.writerow([])
            self.latency_csv_writer.writerow(["Summary"])
            self.latency_csv_writer.writerow(["Average Latency (ms)", f"{avg_latency:.2f}"])
            self.latency_csv_writer.writerow(["Min Latency (ms)", f"{min_latency:.2f}"])
            self.latency_csv_writer.writerow(["Max Latency (ms)", f"{max_latency:.2f}"])
            self.latency_csv_writer.writerow(["Throughput (fps)", f"{throughput:.2f}"])
            self.latency_csv_file.flush()

            self.update_log(
                f"Summary → Avg: {avg_latency:.2f} ms | "
                f"Min: {min_latency:.2f} ms | "
                f"Max: {max_latency:.2f} ms | "
                f"Throughput: {throughput:.2f} fps"
            )

        if self.latency_csv_file:
            self.latency_csv_file.flush()
            self.latency_csv_file.close()
            self.latency_csv_file = None
        if self.sysmon_timer.isActive():
            self.sysmon_timer.stop()
        if self.sysmon_csv_file:
            self.sysmon_csv_file.close()
            self.sysmon_csv_file = None
            self.sysmon_csv_writer = None
        
        self.calibration_counter = 0
        self.calibration_data = []
        self.calibration_bar.setValue(0)
        self.calibration_bar.show()

        self.update_log("Capture stopped.")
        self.set_placeholder("No Video")
        
    def draw_text_with_background(image, text, position, font_scale=0.5, font_color=(0, 0, 0), thickness=2):
        """Helper: render teks dengan background putih"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        padding = 5
        x, y = position
        top_left = (x - padding, y + padding)
        bottom_right = (x + text_w + padding, y - text_h - padding)
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
        cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    def calibrate_posture(self, landmarks):
        # Ambil titik utama (hidung & pinggul)
        def get_point(landmark): return (landmark.x, landmark.y, landmark.z)

        nose = get_point(landmarks[mp_pose.PoseLandmark.NOSE.value])
        r_hip = get_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        l_hip = get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        mid_hip = tuple((r + l) / 2 for r, l in zip(r_hip, l_hip))
        
        # Hitung baseline awal (leher, batang tubuh, kaki kanan, kaki kiri)
        neck_baseline = self.calculate_angle(mid_hip[:3], nose[:3], nose[:3], plane='yz')
        trunk_baseline = 0
        right_leg = self.calculate_angle2(
            get_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
            get_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
            get_point(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        )
        left_leg = self.calculate_angle2(
            get_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
            get_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
            get_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        )
        # Simpan hasil kalibrasi frame per frame
        self.calibration_data.append((neck_baseline, trunk_baseline, right_leg, left_leg))
        self.calibration_counter += 1
        self.calibration_bar.setValue(self.calibration_counter)
        self.update_log(f"Calibrating... ({self.calibration_counter}/{self.calibration_frames})")
        # Jika sudah cukup frame → ambil rata-rata dan selesai kalibrasi
        if self.calibration_counter >= self.calibration_frames:
            self.neck_baseline = np.mean([d[0] for d in self.calibration_data])
            self.trunk_baseline = np.mean([d[1] for d in self.calibration_data])
            self.right_leg_baseline = np.mean([d[2] for d in self.calibration_data])
            self.left_leg_baseline = np.mean([d[3] for d in self.calibration_data])
            self.calibrated = True
            self.calibration_bar.hide()
            self.update_log("✅ Calibration complete.")

    def force_recalibrate(self):
        self.calibrated = False
        self.calibration_counter = 0
        self.calibration_data = []
        self.calibration_bar.setValue(0)
        self.calibration_bar.show()
        self.update_log("Manual recalibration triggered. Please return to neutral posture.")


    def calculate_angle2(self, a, b, c):
        """Menghitung sudut antara tiga titik dengan referensi di titik b."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle2 = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle2)

    def calculate_angle(self, a, b, c, plane):
        a, b, c = np.array(a), np.array(b), np.array(c)

        if plane == "xy":
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        elif plane == "yz":
            radians = np.arctan2(c[2] - b[2], c[1] - b[1]) - np.arctan2(a[2] - b[2], a[1] - b[1])
        elif plane == "xz":
            radians = np.arctan2(c[2] - b[2], c[0] - b[0]) - np.arctan2(a[2] - b[2], a[0] - b[0])

        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle

        return angle
    
    def calculate_distance(point1, point2):
        """
        Menghitung panjang vektor antara dua titik dalam 3D.
        point1 dan point2 adalah tuple (x, y, z).
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_landmark_info(self, landmark):
        return (landmark.x, landmark.y, landmark.z, landmark.visibility)
    
    def classify_reba(self, landmarks):
        # Get coordinates
        nose = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.NOSE.value])
        
        right_shoulder= self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        left_shoulder= self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        # Menghitung koordinat leher (neck) sebagai titik tengah antara right_shoulder dan left_shoulder
        neck = ((right_shoulder[0] + left_shoulder[0]) / 2,  # X coordinate
            (right_shoulder[1] + left_shoulder[1]) / 2,  # Y coordinate
            (right_shoulder[2] + left_shoulder[2]) / 2)   # Z coordinate
        right_hip = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        left_hip = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        mid_hip = ((right_hip[0] + left_hip[0]) / 2,  # X coordinate
            (right_hip[1] + left_hip[1]) / 2,  # Y coordinate
            (right_hip[2] + left_hip[2]) / 2)   # Z coordinate
        vertical_line = ((mid_hip[0]), (mid_hip[1]-1), (mid_hip[2]))
        right_knee = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        left_knee = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

        mid_knee = ((right_knee[0] + left_knee[0]) / 2,  # X coordinate
            (right_knee[1] + left_knee[1]) / 2,  # Y coordinate
            (right_knee[2] + left_knee[2]) / 2)   # Z coordinate)
        
        right_ankle = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        left_ankle = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
        right_elbow = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        left_elbow = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        
        right_wrist = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        left_wrist = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        
        right_index = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])
        left_index = self.get_landmark_info(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value])
    
        #Calculate new angle
        #STEP 1 : Neck
        neck_raw = self.calculate_angle(mid_hip[:3], neck[:3], nose[:3], plane="yz")
        neck_flexion = neck_raw - self.neck_baseline if self.calibrated else neck_raw
        neck_abduction = 180 - self.calculate_angle(mid_hip[:3], neck[:3], nose[:3], plane="xy")
        neck_twistion = abs(nose[0] - neck[0])
                      
        #STEP 2: Trunk
        trunk_raw = self.calculate_angle(neck[:3], mid_hip[:3], vertical_line[:3], plane="yz")
        trunk_flexion = trunk_raw - self.trunk_baseline if self.calibrated else trunk_raw
        trunk_abduction = self.calculate_angle(neck[:3], mid_hip[:3], vertical_line[:3], plane="xy")
        RS = right_shoulder[0]
        LS = left_shoulder[0]
        #STEP 3: Legs
        right_leg_raw = self.calculate_angle2(right_hip[:3], right_knee[:3], right_ankle[:3])
        right_leg_flexion = right_leg_raw - self.right_leg_baseline if self.calibrated else right_leg_raw

        left_leg_raw = self.calculate_angle2(left_hip[:3], left_knee[:3], left_ankle[:3])
        left_leg_flexion = left_leg_raw - self.left_leg_baseline if self.calibrated else left_leg_raw   
        
        
        # STEP 7 : Shoulder
        right_shoulder_flexion = self.calculate_angle(right_elbow[:3], right_shoulder[:3], right_hip[:3], plane="yz")
        right_shoulder_abduction = self.calculate_angle(right_elbow[:3], right_shoulder[:3], right_hip[:3], plane="xy")
        left_shoulder_flexion = self.calculate_angle(left_elbow[:3], left_shoulder[:3], left_hip[:3], plane="yz")
        left_shoulder_abduction = self.calculate_angle(left_elbow[:3], left_shoulder[:3], left_hip[:3], plane="xy")
        # STEP 8 : Elbow
        right_elbow_flexion = 180 - self.calculate_angle2(right_shoulder[:3], right_elbow[:3], right_wrist[:3])
        left_elbow_flexion = 180 - self.calculate_angle2(left_shoulder[:3], left_elbow[:3], left_wrist[:3])
        #STEP 9 : Sudut Wrist
        right_wrist_flexion = 180 - self.calculate_angle2(right_elbow[:3], right_wrist[:3], right_index[:3])
        left_wrist_flexion = 180 - self.calculate_angle2(left_elbow[:3], left_wrist[:3], left_index[:3])

        #SCORING
        #STEP 1 NECK SCORE
        neck_score = 0
        if -5 < neck_flexion < 20:
            neck_score = 1
        else:
            neck_score = 2
        neck_twist = ""
        neck_bend = ""
        if neck_abduction > 30:
            neck_score += 1
            neck_bend ="Bend"
        elif neck_twistion > 0.1:
            neck_score += 1
            neck_twist = "Twisted"
        #STEP 2 TRUNK SCORE
        trunk_score = 1
        if trunk_flexion < -5:
            trunk_score = 2
        elif 5 < trunk_flexion < 20:
            trunk_score = 2
        elif 20 < trunk_flexion < 60:
            trunk_score = 3
        elif trunk_flexion > 60:
            trunk_score = 4
        else:
            trunk_score = 1
        trunk_twist = ""
        trunk_bend = ""
        
        if trunk_abduction > 10:
            trunk_score += 1
            trunk_bend ="Bend"
        elif RS > -0.03 or LS < 0.03:
            trunk_score += 1
            trunk_twist = "Twisted"
        #STEP 3 LEG SCORE
        leg_score = 1
        if abs(right_leg_flexion - left_leg_flexion) > 15:  # Threshold selisih 15° untuk asimetri
            leg_score = 2  # Bertumpu pada satu kaki
        min_leg_flexion = min(right_leg_flexion, left_leg_flexion)
        if 30 <= min_leg_flexion <= 60:
            leg_score += 1  # Tambahan jika lutut menekuk antara 30°-60°
        elif min_leg_flexion > 60:
            leg_score += 2  # Tambahan jika lutut menekuk lebih dari 60°
        #STEP 4
        table_a = np.empty((5,3,4), dtype=int)
        table_a = np.array([
        
                [
                    [1,2,3,4],
                    [1,2,3,4],
                    [3,3,5,6]
                ],
                [
                    [2,3,4,5],
                    [3,4,5,6],
                    [4,5,6,7]
                ],
                [
                    [2,4,5,6],
                    [4,5,6,7],
                    [5,6,7,8]
                ],
                [
                    [3,5,6,7],
                    [5,6,7,8],
                    [6,7,8,9]
                ],
                [
                    [4,6,7,8],
                    [6,7,8,9],
                    [7,8,9,9]
                ]
        ]
        )
        
        
        table_a_score = (table_a[trunk_score - 1][neck_score - 1][leg_score - 1])
        
        
        #STEP 7 : Upper Arm
        #RIGHT
        right_upperarm_score_flexion = 0
        right_upperarm_score_abduction = 0
        
        if right_shoulder_flexion < 20:
            right_upperarm_score_flexion = 1
        elif 20 <= right_shoulder_flexion <= 45:
            right_upperarm_score_flexion = 2
        elif 45 < right_shoulder_flexion <= 90:
            right_upperarm_score_flexion = 3
        else:
            right_upperarm_score_flexion = 4
        
        # Ambil skor tertinggi dari flexion atau abduction
        right_upperarm_score = right_upperarm_score_flexion
        
        right_upperarm_status = ""
        
        if right_shoulder_abduction > 30:
            right_upperarm_score += 1
            right_upperarm_status +="Abduction"
            
        #LEFT
        left_upperarm_score_flexion = 0
        left_upperarm_score_abduction = 0
        
        if left_shoulder_flexion < 20:
            left_upperarm_score_flexion = 1
        elif 20 <= left_shoulder_flexion <= 45:
            left_upperarm_score_flexion = 2
        elif 45 < left_shoulder_flexion <= 90:
            left_upperarm_score_flexion = 3
        else:
            left_upperarm_score_flexion = 4
        
        # Ambil skor tertinggi dari flexion atau abduction
        left_upperarm_score = left_upperarm_score_flexion
        
        left_upperarm_status = ""
        
        if left_shoulder_abduction > 30:
            left_upperarm_score += 1
            left_upperarm_status += "Abduction"
        upperarm_score = max(right_upperarm_score, left_upperarm_score)
        #STEP 8 RIGHT LOWER ARM SCORE
        right_lowerarm_score = 2
        if 60 <= right_elbow_flexion <= 100:
            right_lowerarm_score = 1
    
        #STEP 8 LEFT LOWER ARM SCORE
        left_lowerarm_score = 2
        if 60 <= left_elbow_flexion <= 100:
            left_lowerarm_score = 1
    
        #STEP 8 LOWER ARM SCORE
        lowerarm_score = max(left_lowerarm_score, right_lowerarm_score)
        #STEP 9 RIGHT WRIST SCORE
        right_wrist_score = 2
        if 0 <= right_wrist_flexion <= 15:
            right_wrist_score = 1
    
        if right_wrist_flexion > 15:
            right_wrist_score += 1
    
        #STEP 9 LEFT WRIST SCORE
        left_wrist_score = 2
        if 0 <= left_wrist_flexion <= 15:
            left_wrist_score = 1
    
        if left_wrist_flexion > 15:
            left_wrist_score += 1
    
        #STEP 9 WRIST SCORE
        wrist_score = max(left_wrist_score, right_wrist_score)
        #STEP 10 TABLE B
        table_b = np.empty((6,2,3), dtype=int)
        table_b = np.array([
        
                [
                    [1,2,2],
                    [1,2,3]
                ],
                [
                    [1,2,3],
                    [2,3,4]
                ],
                [
                    [3,4,5],
                    [4,5,5]
                ],
                [
                    [4,5,5],
                    [5,6,7]
                ],
                [
                    [6,7,8],
                    [7,8,8]
                ],
                [
                    [7,8,8],
                    [8,9,9]
                ]
        ]
        )
    
        table_b_score = (table_b[upperarm_score - 1][lowerarm_score - 1][wrist_score - 1])
        #STEP AKHIR
        table_c = np.empty((12,12), dtype=int)
        table_c = np.array([
            [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
            [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
            [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
            [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
            [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
            [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
            [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
            [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
            [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
        ])
        
        reba_score = (table_c[table_a_score - 1][table_b_score - 1])
        return {
                "reba_score": reba_score,
                "neck_flexion": neck_flexion,
                "trunk_flexion": trunk_flexion,
                "right_leg_flexion": right_leg_flexion,
                "left_leg_flexion": left_leg_flexion,
                "right_shoulder_flexion": right_shoulder_flexion,
                "left_shoulder_flexion": left_shoulder_flexion,
                "right_elbow_flexion": right_elbow_flexion,
                "left_elbow_flexion": left_elbow_flexion,
                "right_wrist_flexion": right_wrist_flexion,
                "left_wrist_flexion": left_wrist_flexion,
                "neck_score": neck_score,
                "trunk_score": trunk_score,
                "leg_score": leg_score,
                "upperarm_score": upperarm_score,
                "lowerarm_score": lowerarm_score,
                "wrist_score": wrist_score,
            }

    def update_info(self, result):
        text = (
            f"REBA Score: {result['reba_score']}\n\n"
            f"Neck Flexion: {result['neck_flexion']:.1f}°\n"
            f"Trunk Flexion: {result['trunk_flexion']:.1f}°\n"
            f"Right Leg Flexion: {result['right_leg_flexion']:.1f}°\n"
            f"Left Leg Flexion: {result['left_leg_flexion']:.1f}°\n"
            f"Right Shoulder Flexion: {result['right_shoulder_flexion']:.1f}°\n"
            f"Left Shoulder Flexion: {result['left_shoulder_flexion']:.1f}°\n"
            f"Right Elbow Flexion: {result['right_elbow_flexion']:.1f}°\n"
            f"Left Elbow Flexion: {result['left_elbow_flexion']:.1f}°\n"
            f"Right Wrist Flexion: {result['right_wrist_flexion']:.1f}°\n"
            f"Left Wrist Flexion: {result['left_wrist_flexion']:.1f}°\n\n"
            f"Neck Score: {result['neck_score']}\n"
            f"Trunk Score: {result['trunk_score']}\n"
            f"Leg Score: {result['leg_score']}\n"
            f"Upper Arm Score: {result['upperarm_score']}\n"
            f"Lower Arm Score: {result['lowerarm_score']}\n"
            f"Wrist Score: {result['wrist_score']}\n"
        )
        self.info_label.setText(text)
        score = result['reba_score']
        if score < 4:
            color = "green"
            status_text = "Aman"
        elif score < 8:
            color = "yellow"
            status_text = "Hati-hati"
        else:
            color = "red"
            status_text = "Bahaya"

        self.status_indicator.setText(status_text)
        self.status_indicator.setStyleSheet(
            f"background-color: {color}; color: black; font-weight: bold; font-size: 16px;"
        )

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

    def set_placeholder(self, text="No Video"):
        # Buat gambar kosong abu-abu
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (220, 220, 220)  # Warna abu-abu terang

         # Tambahkan teks custom di tengah
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        textX = (placeholder.shape[1] - textsize[0]) // 2
        textY = (placeholder.shape[0] + textsize[1]) // 2
        cv2.putText(
            placeholder, text,
            (textX, textY),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (70, 70, 70), 3, cv2.LINE_AA
        )

        self.display_image(placeholder)

# === RUN ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ErgoApp()
    window.show()
    sys.exit(app.exec_())
