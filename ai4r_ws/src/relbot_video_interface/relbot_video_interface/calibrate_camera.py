#!/usr/bin/env python3
"""
Camera Calibration Script using OpenCV Chessboard Detection

Performs intrinsic camera calibration using a printed 9x6 chessboard pattern
(8x5 internal corners, 3 cm square size). The script captures images from a
live webcam feed and computes the camera matrix and distortion coefficients.

The focal length (fx) extracted from the camera matrix is used by the
video_interface_node for pinhole-model distance estimation.

Controls:
    SPACE  — Capture current frame (only if chessboard corners are detected)
    Q      — Finish capture and run calibration (minimum 10 frames required)
    ESC    — Abort without saving

Usage:
    python3 calibrate_camera.py
"""

import sys
import os
import json
import signal
import datetime
import numpy as np
import cv2


# Chessboard configuration — must match the printed pattern
CHESSBOARD_INNER_CORNERS = (8, 5)   # (cols-1, rows-1) for a 9x6 board
SQUARE_SIZE_CM = 3.0                # physical size of each square in cm
MIN_CAPTURES = 10                   # minimum frames needed for calibration


def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n  Interrupted — cleaning up...")
    cv2.destroyAllWindows()
    sys.exit(1)


def calibrate_camera(camera_index=0, output_path='calibration.json'):
    """Run interactive camera calibration and save results to JSON."""
    signal.signal(signal.SIGINT, _signal_handler)

    # Termination criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points in real-world coordinates (z=0 plane)
    # Each point represents an internal corner position in cm
    objp = np.zeros((CHESSBOARD_INNER_CORNERS[0] * CHESSBOARD_INNER_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_INNER_CORNERS[0],
        0:CHESSBOARD_INNER_CORNERS[1]
    ].T.reshape(-1, 2) * SQUARE_SIZE_CM

    # Storage for calibration data
    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in image plane
    capture_count = 0

    print(f"Opening camera at /dev/video{camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {camera_index}")
        return False

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=" * 60)
    print("  Camera Calibration — Chessboard Detection")
    print("=" * 60)
    print(f"  Chessboard: 9x6 squares ({CHESSBOARD_INNER_CORNERS[0]}x{CHESSBOARD_INNER_CORNERS[1]} internal corners)")
    print(f"  Square size: {SQUARE_SIZE_CM} cm")
    print(f"  Minimum captures: {MIN_CAPTURES}")
    print()
    print("  Controls:")
    print("    SPACE  — Capture frame (when corners are detected)")
    print("    Q      — Finish and calibrate")
    print("    ESC    — Abort")
    print("=" * 60)

    cv2.namedWindow('Camera Calibration', cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Try to find chessboard corners
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_INNER_CORNERS, None)

        if found:
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Draw detected corners on display frame
            cv2.drawChessboardCorners(display, CHESSBOARD_INNER_CORNERS, corners_refined, found)
            status_text = f"Corners DETECTED | Captures: {capture_count}/{MIN_CAPTURES} | Press SPACE to capture"
            status_color = (0, 255, 0)  # green
        else:
            status_text = f"No corners found | Captures: {capture_count}/{MIN_CAPTURES}"
            status_color = (0, 0, 255)  # red

        # Draw status bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(display, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow('Camera Calibration', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and found:
            # Capture this frame
            obj_points.append(objp)
            img_points.append(corners_refined)
            capture_count += 1
            print(f"  Captured frame {capture_count}")

        elif key == ord('q') or key == ord('Q'):
            if capture_count < MIN_CAPTURES:
                print(f"\n  Need at least {MIN_CAPTURES} captures (have {capture_count}). Keep capturing or press ESC to abort.")
            else:
                print(f"\n  Finishing with {capture_count} captures...")
                break

        elif key == 27:  # ESC
            print("\n  Calibration aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    if capture_count < MIN_CAPTURES:
        print(f"ERROR: Not enough captures ({capture_count}/{MIN_CAPTURES})")
        return False

    # Run calibration
    print("\n  Running camera calibration...")
    image_size = (gray.shape[1], gray.shape[0])  # (width, height)
    ret_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Extract focal length
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("\n" + "=" * 60)
    print("  Calibration Results")
    print("=" * 60)
    print(f"  Reprojection error: {ret_error:.4f} px")
    print(f"  Focal length (fx):  {fx:.2f} px")
    print(f"  Focal length (fy):  {fy:.2f} px")
    print(f"  Principal point:    ({cx:.2f}, {cy:.2f}) px")
    print(f"  Image size:         {image_size[0]}x{image_size[1]}")
    print(f"  Distortion coeffs:  {dist_coeffs.flatten().tolist()}")

    if ret_error > 1.0:
        print("\n  WARNING: Reprojection error > 1.0 px. Consider recalibrating")
        print("  with better images (varied angles, covering frame edges).")
    else:
        print("\n  Calibration quality: GOOD")

    # Save calibration data to JSON
    calibration_data = {
        'focal_length_px': float(fx),
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.flatten().tolist(),
        'reprojection_error': float(ret_error),
        'image_size': list(image_size),
        'square_size_cm': SQUARE_SIZE_CM,
        'chessboard_inner_corners': list(CHESSBOARD_INNER_CORNERS),
        'num_captures': capture_count,
        'calibration_date': datetime.datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\n  Calibration saved to: {output_path}")
    print("=" * 60)
    return True


if __name__ == '__main__':
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'calibration.json'
    calibrate_camera(camera_index, output_path)
