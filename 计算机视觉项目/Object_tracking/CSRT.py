import cv2

def draw_path(image, points, color=(0, 0, 255), thickness=2):
    for i in range(1, len(points)):
        cv2.line(image, points[i - 1], points[i], color, thickness)

def create_csrt_tracker():
    major, minor, _ = cv2.__version__.split(".")
    if int(major) < 4 or (int(major) == 4 and int(minor) < 5):
        return cv2.Tracker_create("CSRT")
    else:
        return cv2.TrackerCSRT_create()

tracker = create_csrt_tracker()

def main():
    video_file = '/Users/CV/project/1.mp4'
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()

    track_window = cv2.selectROI('img', frame)
    x, y, w, h = track_window
    initial_bbox = (x, y, w, h)

    # Replace with cv2.TrackerKCF_create() or cv2.TrackerMOSSE_create() if needed
    tracker.init(frame, initial_bbox)

    predicted_positions = [(x + w // 2, y + h // 2)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ret, bbox = tracker.update(frame)
        if ret:
            x, y, w, h = tuple(map(int, bbox))
            center = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            predicted_positions.append(center)
            draw_path(frame, predicted_positions)
        else:
            cv2.putText(frame, "Tracking failure detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('img', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()