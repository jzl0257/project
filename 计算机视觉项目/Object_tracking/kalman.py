import cv2
import numpy as np

def draw_path(image, points, color=(0, 0, 255), thickness=2):
    for i in range(1, len(points)):
        cv2.line(image, points[i - 1], points[i], color, thickness)


def avg_center(points):
    x = sum(pt[0] for pt in points) / 4
    y = sum(pt[1] for pt in points) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


def initialize_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    return kf


def main():
    video_file = '/Users/CV/project/b.mp4'
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()

    track_window = cv2.selectROI('img', frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((30, 40, 50)), np.array((80, 255, 255)))
    hist = cv2.calcHist([hsv], [0], mask, [181], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    klm = initialize_kalman_filter()

    x, y, w, h = track_window
    cent = avg_center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    klm.statePre = np.array([cent[0], cent[1], 0, 0], dtype=np.float32)
    predicted_positions = [(int(cent[0]), int(cent[1]))]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window

        cent = avg_center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])

        klm.correct(cent)
        pred_coords = klm.predict()

        cv2.circle(frame, (int(pred_coords[0]), int(pred_coords[1])), 30, (255, 255, 0), -1)
        drawn_rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        (img_x, img_y) = drawn_rect.shape[:2]
        predicted_positions.append((int(pred_coords[0]), int(pred_coords[1])))

        cv2.imshow('img', drawn_rect)
        if cv2.waitKey(1) == ord('q'):
            break

    draw_path(drawn_rect, predicted_positions)
    cv2.imshow('Trajectory', drawn_rect)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


main()