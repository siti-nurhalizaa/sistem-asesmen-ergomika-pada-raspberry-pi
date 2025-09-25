import os
import json
import socket
import uuid
import firebase_admin
from firebase_admin import firestore, credentials
import threading
import time

class FirebaseWorker:
    def __init__(self, buffer_file="firebase_buffer.json", flush_interval=30):
        self.buffer = []
        self.buffer_file = buffer_file
        self.flush_interval = flush_interval  # detik
        self.cred = None
        self.db = None
        self.app = None
        self.running = False
        self.flush_thread = None
        self.load_buffer()

    def run(self):
         """Inisialisasi Firebase"""
        try:
            self.cred = credentials.Certificate("./ServiceAccount.json")
            if not firebase_admin._apps:
                self.app = firebase_admin.initialize_app(credential=self.cred)
            self.db = firestore.client()
            print("✅ Firebase initialized.")
            self.running = True
            self.start_flush_loop()
        except Exception as e:
            print("❌ Firebase init failed:", e)
            self.db = None

    def upload_file(self, data):
        """Tambahkan data ke buffer untuk dikirim kemudian (tanpa flush langsung)"""
        if not data.get("reba_score") or not data.get("timestamp"):
            print("⚠️ Data tidak lengkap, tidak dimasukkan ke buffer.")
            return
        try:
            real_data = {
                "reba_score": int(data["reba_score"]),
                "timestamp": data["timestamp"]
            }
            self.buffer.append(real_data)
            self.save_buffer()
            print("🟡 Data buffered:", real_data)
        except Exception as e:
            print("❌ Gagal menambahkan ke buffer:", e)

    def flush_buffer(self):
        """Kirim semua data di buffer ke Firestore"""
        if not self.db or not self.is_online():
            print("⚠️ Offline atau Firebase belum siap. Lewati flush.")
            return
        if not self.buffer:
            print("ℹ️ Buffer kosong, tidak ada yang di-upload.")
            return

        print(f"📤 [flush_buffer] Mulai upload jam {time.strftime('%H:%M:%S')}...")
        success = []
        for data in self.buffer:
            try:
                uid = str(uuid.uuid4())
                self.db.collection("ergonomika").document(uid).set(data)
                success.append(data)
                print("✅ Flushed:", data)
            except Exception as e:
                print("❌ Upload gagal, akan dicoba lagi nanti:", e)
                break  # stop flush jika 1 gagal, retry nanti

        self.buffer = [d for d in self.buffer if d not in success]
        self.save_buffer()

    def start_flush_loop(self):
          """Jalankan loop periodic flush di thread terpisah"""
        def loop():
            while self.running:
                self.flush_buffer()
                time.sleep(self.flush_interval)
        self.flush_thread = threading.Thread(target=loop, daemon=True)
        self.flush_thread.start()
        print(f"🔁 Mulai flush loop setiap {self.flush_interval} detik.")

    def stop(self):
        """Hentikan worker dan flush terakhir"""
        self.running = False
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=1)
        self.flush_buffer()  # flush terakhir saat berhenti
        print("🛑 FirebaseWorker dihentikan.")

    def save_buffer(self):
        """Simpan buffer ke file lokal"""
        try:
            with open(self.buffer_file, "w") as f:
                json.dump(self.buffer, f, indent=2)
        except Exception as e:
            print("❌ Gagal menyimpan buffer:", e)

    def load_buffer(self):
        """Muat buffer dari file lokal jika ada"""
        if os.path.exists(self.buffer_file):
            try:
                with open(self.buffer_file, "r") as f:
                    self.buffer = json.load(f)
                print(f"📁 Buffer dimuat. {len(self.buffer)} item di buffer.")
            except json.JSONDecodeError:
                print("⚠️ Buffer rusak, memulai kosong.")
                self.buffer = []
        else:
            print("📁 Tidak ada buffer lama. Mulai baru.")

    def is_online(self, host="8.8.8.8", port=53, timeout=2):
        """Cek koneksi internet dengan ping socket"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False
