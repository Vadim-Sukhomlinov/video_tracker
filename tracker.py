import cv2
import numpy as np
from typing import Tuple, Optional
from config import LK_PARAMS, SMOOTHING_FACTOR


class VideoPointTracker:

    def __init__(self):
        self.tracking_point: Optional[Tuple[int, int]] = None
        self.old_points: Optional[np.ndarray] = None
        self.old_frame: Optional[np.ndarray] = None
        self.current_coordinates: Optional[Tuple[float, float]] = None
        self.smoothed_coordinates: Optional[Tuple[float, float]] = None
        self.smoothing_factor = SMOOTHING_FACTOR

    def set_tracking_point(self, x: int, y: int, frame_gray: np.ndarray) -> None:

        half_win = 20
        y_min = max(0, y - half_win)
        y_max = min(frame_gray.shape[0], y + half_win)
        x_min = max(0, x - half_win)
        x_max = min(frame_gray.shape[1], x + half_win)

        search_region = frame_gray[y_min:y_max, x_min:x_max]
        corners = cv2.goodFeaturesToTrack(
            search_region,
            maxCorners=1,
            qualityLevel=0.01,
            minDistance=10
        )

        if corners is not None:
            corner_x, corner_y = corners[0][0]
            x = int(x_min + corner_x)
            y = int(y_min + corner_y)
            print(f"[INFO] Auto-adjusted to corner at: ({x}, {y})")

        self.tracking_point = (x, y)
        self.old_points = np.array([[x, y]], dtype=np.float32)
        self.current_coordinates = (float(x), float(y))
        self.smoothed_coordinates = (float(x), float(y))

    def track_point(self, current_frame_gray: np.ndarray) -> bool:

        if self.old_points is None or self.old_frame is None:
            return False

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.old_frame,
            current_frame_gray,
            self.old_points,
            None,
            **LK_PARAMS
        )

        if status[0] == 1:
            x, y = new_points[0].ravel()

            if self.smoothed_coordinates is not None:
                smooth_x = (self.smoothing_factor * x +
                            (1 - self.smoothing_factor) * self.smoothed_coordinates[0])
                smooth_y = (self.smoothing_factor * y +
                            (1 - self.smoothing_factor) * self.smoothed_coordinates[1])
            else:
                smooth_x, smooth_y = float(x), float(y)

            self.smoothed_coordinates = (smooth_x, smooth_y)
            self.current_coordinates = self.smoothed_coordinates
            self.old_points = new_points
            return True
        else:
            self.reset_tracking()
            return False

    def reset_tracking(self) -> None:
        self.tracking_point = None
        self.old_points = None
        self.current_coordinates = None
        self.smoothed_coordinates = None

    def update_old_frame(self, frame_gray: np.ndarray) -> None:
        self.old_frame = frame_gray.copy()


class VideoRenderer:

    @staticmethod
    def draw_tracking_marker(
        frame: np.ndarray,
        coordinates: Tuple[float, float],
        orbit_radius: int = 7,
        point_radius: int = 3
    ) -> None:

        x, y = int(coordinates[0]), int(coordinates[1])
        cv2.circle(frame, (x, y), orbit_radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), point_radius, (0, 0, 255), -1)

    @staticmethod
    def draw_info_overlay(
        frame: np.ndarray,
        coordinates: Optional[Tuple[float, float]],
        fps: float = 0.0
    ) -> None:

        if coordinates is None:
            return

        x, y = int(coordinates[0]), int(coordinates[1])
        coord_text = f"X: {x}, Y: {y}"
        fps_text = f"FPS: {fps:.1f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 0.7, 2

        (coord_w, coord_h), _ = cv2.getTextSize(
            coord_text, font, font_scale, thickness)
        (fps_w, fps_h), _ = cv2.getTextSize(
            fps_text, font, font_scale, thickness)

        overlay = frame.copy()
        max_w = max(coord_w, fps_w)
        total_h = coord_h + fps_h + 10

        cv2.rectangle(overlay, (5, 5), (15 + max_w,
                      15 + total_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, coord_text, (10, 10 + coord_h),
                    font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (10, 20 + coord_h + fps_h),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
