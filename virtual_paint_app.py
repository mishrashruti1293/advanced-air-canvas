import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ---------------- INITIAL SETUP ---------------- #

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Camera opened:",cap.isOpened())

mask = None
curr_tool = "draw"
thickness = 5
color = (0, 0, 0)

prev_x, prev_y = 0, 0
start_x, start_y = 0, 0
drawing = False

alpha = 0.7  # smoothing factor
ptime = 0

# ---------------- HELPER FUNCTIONS ---------------- #

def fingers_up(hand_landmarks, h, w):
    tips = [8, 12, 16, 20]
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count


def get_distance(p1, p2):
    return int(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


# ---------------- MAIN LOOP ---------------- #

while True:
    ret, frame = cap.read()
    print("Frame Read:",ret)
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if mask is None:
        mask = np.ones((h, w), dtype="uint8") * 255

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        x = int(hand.landmark[8].x * w)
        y = int(hand.landmark[8].y * h)

        thumb_x = int(hand.landmark[4].x * w)
        thumb_y = int(hand.landmark[4].y * h)

        # -------- Brush Thickness Control (Pinch) -------- #
        pinch_dist = get_distance((x, y), (thumb_x, thumb_y))
        thickness = max(2, min(40, pinch_dist // 5))

        # -------- Smoothing -------- #
        x = int(alpha * prev_x + (1 - alpha) * x)
        y = int(alpha * prev_y + (1 - alpha) * y)

        finger_count = fingers_up(hand, h, w)

        # -------- Tool Selection by Finger Count -------- #
        if finger_count == 1:
            curr_tool = "draw"
        elif finger_count == 2:
            curr_tool = "line"
        elif finger_count == 3:
            curr_tool = "rectangle"
        elif finger_count == 4:
            curr_tool = "circle"
        elif finger_count == 0:
            curr_tool = "erase"

        # -------- Drawing Logic -------- #
        if curr_tool == "draw":
            cv2.line(mask, (prev_x, prev_y), (x, y), 0, thickness)

        elif curr_tool == "erase":
            cv2.circle(mask, (x, y), thickness + 10, 255, -1)

        elif curr_tool == "line":
            if not drawing:
                start_x, start_y = x, y
                drawing = True
            cv2.line(frame, (start_x, start_y), (x, y), (255, 0, 0), thickness)

        elif curr_tool == "rectangle":
            if not drawing:
                start_x, start_y = x, y
                drawing = True
            cv2.rectangle(frame, (start_x, start_y), (x, y), (0, 255, 0), thickness)

        elif curr_tool == "circle":
            if not drawing:
                start_x, start_y = x, y
                drawing = True
            radius = get_distance((start_x, start_y), (x, y))
            cv2.circle(frame, (start_x, start_y), radius, (0, 255, 255), thickness)

        prev_x, prev_y = x, y

    else:
        drawing = False
        prev_x, prev_y = 0, 0

    # -------- Apply Mask -------- #
    result = cv2.bitwise_and(frame, frame, mask=mask)
    frame[:, :, 1] = result[:, :, 1]
    frame[:, :, 2] = result[:, :, 2]

    # -------- FPS -------- #
    ctime = time.time()
    fps = 1 / (ctime - ptime) if ptime != 0 else 0
    ptime = ctime

    cv2.putText(frame, f"Tool: {curr_tool}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Thickness: {thickness}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Advanced Air Canvas", frame)

    key = cv2.waitKey(1)

    # Save image
    if key == ord('s'):
        cv2.imwrite("air_canvas_output.png", frame)
        print("Saved as air_canvas_output.png")

    # Clear canvas
    if key == ord('c'):
        mask = np.ones((h, w), dtype="uint8") * 255

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

 