import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=0,
)

CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# Key landmarks only: wrist, fingertips, knuckles (9 points)
KEY_LANDMARKS = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]

def _build_lut(n=256):
    lut = np.zeros((n, 3), dtype=np.uint8)
    stops_bgr = [
        (255, 255,   0),   # cyan
        (255,   0, 150),   # violet
        (  0, 165, 255),   # orange
        (  0,   0, 220),   # red
    ]
    stops_t = [0.0, 0.33, 0.66, 1.0]
    for i in range(n):
        t = i / (n - 1)
        for s in range(len(stops_t) - 1):
            if stops_t[s] <= t <= stops_t[s + 1]:
                lo = np.array(stops_bgr[s], dtype=float)
                hi = np.array(stops_bgr[s + 1], dtype=float)
                f  = (t - stops_t[s]) / (stops_t[s + 1] - stops_t[s])
                lut[i] = (lo + (hi - lo) * f).astype(np.uint8)
                break
    return lut

GRADIENT_LUT = _build_lut()


def draw_gradient_line(img, p1, p2, steps=22, thickness=1):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    for i in range(steps):
        t0 = i / steps
        t1 = (i + 1) / steps
        sx = int(x1 + (x2 - x1) * t0);  sy = int(y1 + (y2 - y1) * t0)
        ex = int(x1 + (x2 - x1) * t1);  ey = int(y1 + (y2 - y1) * t1)
        idx   = int(((t0 + t1) / 2) * 255)
        color = (int(GRADIENT_LUT[idx, 0]),
                 int(GRADIENT_LUT[idx, 1]),
                 int(GRADIENT_LUT[idx, 2]))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def draw_skeleton(img, landmarks, node_color, thickness=1):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c1, c2 in CONNECTIONS:
        cv2.line(img, pts[c1], pts[c2], node_color, thickness, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(img, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pt, 3, node_color,       1, cv2.LINE_AA)
    return pts


def draw_inter_hand_mesh(img, left_pts, right_pts):
    # Only connect the 10 key landmarks across hands (10 lines)
    for idx in KEY_LANDMARKS:
        draw_gradient_line(img, left_pts[idx], right_pts[idx], steps=28, thickness=1)


class FPS:
    def __init__(self, n=30):
        self._t = []
        self._n = n
    def tick(self):
        self._t.append(time.perf_counter())
        if len(self._t) > self._n: self._t.pop(0)
    @property
    def value(self):
        if len(self._t) < 2: return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span else 0.0


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    fps = FPS()
    print("Hand Mesh  |  Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        canvas = frame.copy()

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left_pts = right_pts = None

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lms, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                label = handedness.classification[0].label
                if label == "Left":
                    left_pts  = draw_skeleton(canvas, hand_lms.landmark,
                                              (255, 255, 0), thickness=1)
                else:
                    right_pts = draw_skeleton(canvas, hand_lms.landmark,
                                              (0, 0, 220),   thickness=1)

            if left_pts and right_pts:
                draw_inter_hand_mesh(canvas, left_pts, right_pts)

        fps.tick()
        cv2.putText(canvas, f"FPS {fps.value:5.1f}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 1, cv2.LINE_AA)
        status = "MESH ACTIVE" if (left_pts and right_pts) else "show both hands"
        cv2.putText(canvas, status, (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("Hand Mesh", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()