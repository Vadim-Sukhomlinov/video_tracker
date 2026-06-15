import cv2
import time
from tracker import VideoPointTracker, VideoRenderer


class FPSMeter:

    def __init__(self):
        self.fps = 0.0
        self._frame_count = 0
        self._start_time = time.time()

    def update(self) -> float:
        self._frame_count += 1
        elapsed = time.time() - self._start_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()
        return self.fps


def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        tracker, current_gray = param
        tracker.set_tracking_point(x, y, current_gray)
        print(f"[INFO] Tracking point set at: ({x}, {y})")


def main():
    tracker = VideoPointTracker()
    renderer = VideoRenderer()
    fps_meter = FPSMeter()

    print("=" * 50)
    print("Video Point Tracker v2.0 (Improved LK + Smoothing)")
    print("=" * 50)

    video_path = input("Enter video file path: ").strip()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Failed to open video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {frame_width}x{frame_height}, {total_frames} frames")
    print("[INFO] Click to set tracking point. Press 'q' or 'Esc' to quit.")

    window_name = 'Video Point Tracker'
    cv2.namedWindow(window_name)

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read video")
        return

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.update_old_frame(current_gray)
    cv2.setMouseCallback(window_name, mouse_callback, (tracker, current_gray))

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        display_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.setMouseCallback(window_name, mouse_callback,
                             (tracker, frame_gray))

        if tracker.tracking_point is not None:
            success = tracker.track_point(frame_gray)
            if success:
                renderer.draw_tracking_marker(
                    display_frame, tracker.current_coordinates)
            else:
                print("[WARNING] Tracking point lost.")

        current_fps = fps_meter.update()
        renderer.draw_info_overlay(
            display_frame, tracker.current_coordinates, current_fps)

        proc_time = time.time() - start_time
        cv2.putText(display_frame, f"Process: {proc_time*1000:.1f}ms",
                    (frame_width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        tracker.update_old_frame(frame_gray)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
