"""
ระบบตรวจจับความง่วงนอนด้วย PERCLOS + MAR
==============================================
PERCLOS = Percentage of Eye Closure
วัดเปอร์เซ็นต์ที่ตาปิดใน window เวลาที่กำหนด
ตรวจจับความง่วงได้ "ก่อน" ที่จะหลับจริง

ระดับการแจ้งเตือน:
  NORMAL   : PERCLOS < 20%  → ปกติ (เขียว)
  DROWSY   : PERCLOS 20-40% → เริ่มง่วง (เหลือง) ⚠
  CRITICAL : PERCLOS > 40%  → ง่วงมาก (แดง) 🚨

Tech Stack:
  - Python 3.11
  - MediaPipe Face Mesh
  - OpenCV
  - NumPy
  - collections.deque (sliding window)
  - threading (non-blocking alert)
==============================================
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import sys
from collections import deque


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
# EAR
EAR_THRESHOLD       = 0.20

# PERCLOS window = กี่วินาทีที่ใช้คำนวณ
PERCLOS_WINDOW_SEC  = 15     # มาตรฐานงานวิจัยใช้ 30–60 วินาที
FPS_ESTIMATE        = 15     # FPS โดยประมาณ (ใช้ขนาด window)
PERCLOS_WINDOW_SIZE = PERCLOS_WINDOW_SEC * FPS_ESTIMATE  # จำนวน frame ใน window

# PERCLOS threshold
PERCLOS_DROWSY    = 0.20    # 20% → เริ่มง่วง
PERCLOS_CRITICAL  = 0.40    # 40% → ง่วงมาก

# MAR
MAR_THRESHOLD       = 0.75
MAR_CONSEC_FRAMES   = 12

# Blink rate (ครั้ง/นาที) — คนง่วงกะพริบถี่กว่าปกติ
BLINK_RATE_HIGH     = 25    # > 25 ครั้ง/นาที = สัญญาณง่วง

# ─────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX     = [61, 37, 0, 267, 291, 314, 17, 84]


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [get_point(landmarks, i, w, h) for i in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def mouth_aspect_ratio(landmarks, w, h):
    pts = [get_point(landmarks, i, w, h) for i in MOUTH_IDX]
    A = np.linalg.norm(pts[1] - pts[7])
    B = np.linalg.norm(pts[2] - pts[6])
    C = np.linalg.norm(pts[3] - pts[5])
    D = np.linalg.norm(pts[0] - pts[4])
    return (A + B + C) / (2.0 * D) if D != 0 else 0.0


def compute_perclos(eye_closed_buffer):
    """
    คำนวณ PERCLOS จาก sliding window
    = จำนวน frame ที่ตาปิด / ขนาด window ทั้งหมด
    """
    if len(eye_closed_buffer) == 0:
        return 0.0
    return sum(eye_closed_buffer) / len(eye_closed_buffer)


def draw_landmarks(frame, landmarks, indices, w, h, color):
    pts = []
    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 2, color, -1)
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)],
                  isClosed=True, color=color, thickness=1)


def draw_perclos_bar(frame, perclos, img_w, img_h):
    """วาดแถบ PERCLOS แนวนอนพร้อม threshold markers"""
    bx, by = 10, img_h - 80
    bw, bh = img_w - 20, 20

    # พื้นหลัง
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)

    # fill สี
    fill_w = int(bw * min(perclos, 1.0))
    if perclos < PERCLOS_DROWSY:
        color = (0, 200, 0)       # เขียว
    elif perclos < PERCLOS_CRITICAL:
        color = (0, 200, 255)     # เหลือง
    else:
        color = (0, 0, 255)       # แดง
    cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bh), color, -1)

    # เส้น threshold 20%
    t1x = bx + int(bw * PERCLOS_DROWSY)
    cv2.line(frame, (t1x, by - 4), (t1x, by + bh + 4), (0, 220, 255), 1)
    cv2.putText(frame, "20%", (t1x - 14, by - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

    # เส้น threshold 40%
    t2x = bx + int(bw * PERCLOS_CRITICAL)
    cv2.line(frame, (t2x, by - 4), (t2x, by + bh + 4), (0, 80, 255), 1)
    cv2.putText(frame, "40%", (t2x - 14, by - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 255), 1)

    # label
    cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%  ({len(eye_closed_buffer)}/{PERCLOS_WINDOW_SIZE} frames)",
                (bx, by - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # border
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)


def play_alert(level="drowsy"):
    """เสียงแจ้งเตือนแยกตามระดับ"""
    try:
        if sys.platform == "win32":
            import winsound
            if level == "critical":
                for _ in range(10):
                    winsound.Beep(1400, 400)
                    time.sleep(0.05)
            elif level == "drowsy":
                for _ in range(3):
                    winsound.Beep(900, 300)
                    time.sleep(0.1)
            else:  # yawn
                for _ in range(2):
                    winsound.Beep(700, 400)
                    time.sleep(0.1)
        else:
            count = 6 if level == "critical" else 3
            for _ in range(count):
                print("\a", end="", flush=True)
                time.sleep(0.2)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

# Sliding window สำหรับ PERCLOS (deque จำกัดขนาดอัตโนมัติ)
eye_closed_buffer = deque(maxlen=PERCLOS_WINDOW_SIZE)


def main():
    print("=" * 55)
    print("  PERCLOS Drowsiness Detection System")
    print(f"  Window: {PERCLOS_WINDOW_SEC} วินาที | FPS ~{FPS_ESTIMATE}")
    print(f"  เตือนระดับ 1 (เหลือง): PERCLOS >= {PERCLOS_DROWSY*100:.0f}%")
    print(f"  เตือนระดับ 2 (แดง)  : PERCLOS >= {PERCLOS_CRITICAL*100:.0f}%")
    print("  กด ESC หรือ Q เพื่อออก")
    print("=" * 55)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] ไม่พบกล้อง")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──
    mar_counter       = 0
    total_yawns       = 0
    total_blinks      = 0
    prev_eye_open     = True
    prev_mouth_closed = True
    alert_thread      = None
    last_alert_level  = None

    # blink rate tracking (sliding window 60 วินาที)
    blink_times       = deque()
    session_start     = time.time()

    print("[INFO] ระบบพร้อม กำลังสะสม frame สำหรับ PERCLOS...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        img_h, img_w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        ear_val  = 0.0
        mar_val  = 0.0
        perclos  = compute_perclos(eye_closed_buffer)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark

            # ── คำนวณ EAR / MAR ──
            left_ear = eye_aspect_ratio(lms, LEFT_EYE_IDX,  img_w, img_h)
            right_ear= eye_aspect_ratio(lms, RIGHT_EYE_IDX, img_w, img_h)
            ear_val  = (left_ear + right_ear) / 2.0
            mar_val  = mouth_aspect_ratio(lms, img_w, img_h)

            eye_closed = ear_val < EAR_THRESHOLD
            mouth_open = mar_val > MAR_THRESHOLD

            # ── อัปเดต PERCLOS buffer ──
            eye_closed_buffer.append(1 if eye_closed else 0)
            perclos = compute_perclos(eye_closed_buffer)

            # ── นับ blink ──
            if prev_eye_open and eye_closed:
                pass
            elif not prev_eye_open and not eye_closed:
                total_blinks += 1
                blink_times.append(now)
            prev_eye_open = not eye_closed

            # ── นับ yawn ──
            if not prev_mouth_closed and not mouth_open:
                total_yawns += 1
            prev_mouth_closed = not mouth_open
            if mouth_open:
                mar_counter += 1
            else:
                mar_counter = 0

            # ── ลบ blink เก่าออกจาก window 60 วินาที ──
            while blink_times and now - blink_times[0] > 60:
                blink_times.popleft()
            blink_rate = len(blink_times)  # ครั้ง/นาที (approximate)

            # ── สี landmark ──
            eye_color   = (0, 0, 255) if eye_closed else (0, 255, 0)
            mouth_color = (0, 165, 255) if mouth_open else (0, 220, 0)
            draw_landmarks(frame, lms, LEFT_EYE_IDX,  img_w, img_h, eye_color)
            draw_landmarks(frame, lms, RIGHT_EYE_IDX, img_w, img_h, eye_color)
            draw_landmarks(frame, lms, MOUTH_IDX,     img_w, img_h, mouth_color)

            # ── ระดับแจ้งเตือน ──
            alert_level = None
            if perclos >= PERCLOS_CRITICAL:
                alert_level = "critical"
            elif perclos >= PERCLOS_DROWSY or blink_rate >= BLINK_RATE_HIGH:
                alert_level = "drowsy"
            if mar_counter >= MAR_CONSEC_FRAMES:
                alert_level = alert_level or "yawn"

            # ── overlay สี ──
            if alert_level == "critical":
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (img_w, img_h), (0, 0, 200), -1)
                cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
            elif alert_level == "drowsy":
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (img_w, img_h), (0, 160, 255), -1)
                cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

            # ── เล่นเสียงเมื่อระดับเปลี่ยน ──
            if alert_level and alert_level != last_alert_level:
                if alert_thread is None or not alert_thread.is_alive():
                    alert_thread = threading.Thread(
                        target=play_alert, args=(alert_level,), daemon=True)
                    alert_thread.start()
            last_alert_level = alert_level

        else:
            # ไม่เจอหน้า — ไม่บันทึกลง buffer (ไม่นับว่าตาปิด)
            pass

        # ── HUD กล่องซ้ายบน ──
        cv2.rectangle(frame, (0, 0), (300, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (300, 130), (60, 60, 60), 1)

        elapsed = int(now - session_start)
        cv2.putText(frame, f"EAR      : {ear_val:.3f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"MAR      : {mar_val:.3f}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"PERCLOS  : {perclos*100:.1f}%  ({len(eye_closed_buffer)}/{PERCLOS_WINDOW_SIZE})",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Blinks/m : {len(blink_times)}  (total {total_blinks})",
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Yawns    : {total_yawns}  | Time: {elapsed}s",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Status banner ──
        if perclos >= PERCLOS_CRITICAL:
            status, s_color = "CRITICAL — PLEASE STOP!", (0, 0, 255)
        elif perclos >= PERCLOS_DROWSY:
            status, s_color = "DROWSY WARNING", (0, 200, 255)
        elif results.multi_face_landmarks:
            status, s_color = "NORMAL", (0, 220, 0)
        else:
            status, s_color = "NO FACE DETECTED", (160, 160, 160)

        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        tx = (img_w - tw) // 2
        cv2.putText(frame, status, (tx, img_h - 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, status, (tx, img_h - 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, s_color, 2)

        # ── PERCLOS bar ──
        draw_perclos_bar(frame, perclos, img_w, img_h)

        cv2.imshow("PERCLOS Drowsiness Detection | ESC to quit", frame)

        if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
            break

    elapsed_total = int(time.time() - session_start)
    print(f"\n[INFO] สรุปผล")
    print(f"  เวลารวม  : {elapsed_total} วินาที")
    print(f"  Blinks   : {total_blinks} ครั้ง")
    print(f"  Yawns    : {total_yawns} ครั้ง")
    print(f"  PERCLOS สุดท้าย: {perclos*100:.1f}%")

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()