"""
NATS Camera Server Script

This script captures video from a locally connected camera (e.g., a USB webcam via OpenCV),
encodes frames as JPEG, and publishes them to a NATS (Neural Autonomic Transport System)
server subject. It aims to maintain a specified frame rate (FPS) for publishing.
The script handles graceful shutdown via SIGINT (Ctrl+C) and SIGTERM signals.

Purpose:
  - To provide a simple video stream over NATS from a local camera.
  - Can be used as a data source for the `NatsCamera` class in LeRobot, allowing
    LeRobot to consume video frames from a NATS subject as if it were a direct
    camera feed.

Command-Line Arguments:
  --nats_ip (str, default: "0.0.0.0"):
    IP address of the NATS server.
  --nats_port (int, default: 4222):
    Port of the NATS server.
  --camera_index (int, default: 0):
    Index of the camera to use (e.g., 0 for /dev/video0, 1 for /dev/video1).
  --width (int, default: 640):
    Desired width of the captured frames. The script will attempt to set this
    on the camera and will resize frames if the camera output doesn't match.
  --height (int, default: 480):
    Desired height of the captured frames. Similar behavior to --width.
  --fps (int, default: 30):
    Target frames per second for capturing and publishing. The loop will try
    to maintain this rate.
  --subject (str, default: "camera.image.raw"):
    NATS subject to publish the JPEG-encoded frames to.
  --jpeg_quality (int, default: 90, range: 0-100):
    Quality of the JPEG encoding for the published frames. Higher values mean
    better quality and larger frame sizes.

Basic Usage Examples:
  1. Publish from camera index 0 to subject "my.cam.stream" at 20 FPS:
     python examples/nats_camera_server.py --camera_index 0 --subject my.cam.stream --fps 20

  2. Publish from camera index 1, with specific resolution and NATS server IP:
     python examples/nats_camera_server.py --camera_index 1 --width 1280 --height 720 \
            --nats_ip 192.168.1.100 --subject office.cam.main

  3. To see all options:
     python examples/nats_camera_server.py -h

Required Dependencies:
  - opencv-python (cv2): For camera capture and JPEG encoding.
  - nats-py: For NATS communication.
    (These should be installed as part of LeRobot's dependencies if using NatsCamera)
"""
import asyncio
import argparse
import logging
import signal
import time

import cv2
import nats
import nats.errors

# Global event to signal shutdown
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """
    Handles termination signals (SIGINT, SIGTERM) to initiate a graceful shutdown.
    Sets the global shutdown_event.
    """
    logging.info(f"Shutdown signal {signal.Signals(signum).name} received...")
    shutdown_event.set()


async def async_main(args):
    """
    Main asynchronous function to run the NATS camera server.
    Initializes the camera, connects to NATS, and enters a loop to
    capture, encode, and publish frames until a shutdown signal is received.
    """
    cap = None
    nc = None

    try:
        # Camera Initialization
        logging.info(f"Initializing camera with index: {args.camera_index}")
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video capture device at index {args.camera_index}")
            return

        # Attempt to set camera properties
        logging.info(f"Attempting to set camera properties: {args.width}x{args.height} @ {args.fps}FPS")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        # Log actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Actual camera properties: {actual_width}x{actual_height} @ {actual_fps:.2f}FPS")
        if actual_width != args.width or actual_height != args.height:
            logging.warning(
                f"Requested resolution {args.width}x{args.height} not fully supported. "
                f"Using {actual_width}x{actual_height}."
            )
        # Note: FPS setting might not be effective for all cameras/backends.
        # The loop timing will try to match args.fps regardless.

        # NATS Connection
        nats_server_url = f"nats://{args.nats_ip}:{args.nats_port}"
        logging.info(f"Attempting to connect to NATS server at {nats_server_url}")
        try:
            nc = await nats.connect(nats_server_url, timeout=5)
            logging.info(f"Connected to NATS server at {nats_server_url}")
        except nats.errors.NoServersError:
            logging.error(f"Could not connect to NATS: No servers available at {nats_server_url}.")
            return
        except nats.errors.TimeoutError:
            logging.error(f"Could not connect to NATS: Connection timeout to {nats_server_url}.")
            return
        except Exception as e:
            logging.error(f"Error connecting to NATS: {e}")
            return

        # Publishing Loop
        frame_duration = 1.0 / args.fps
        logging.info(f"Starting frame publishing loop to subject '{args.subject}' at {args.fps} FPS target.")

        while not shutdown_event.is_set():
            loop_start_time = time.monotonic()

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Failed to retrieve frame from camera. Retrying...")
                await asyncio.sleep(0.1)  # Wait a bit before retrying
                continue

            # Resize frame to target dimensions (OpenCV might not always set it perfectly)
            # This ensures the output matches args.width and args.height.
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                 resized_frame = cv2.resize(frame, (args.width, args.height))
            else:
                 resized_frame = frame

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            is_success, buffer = cv2.imencode(".jpg", resized_frame, encode_param)

            if is_success:
                try:
                    await nc.publish(args.subject, buffer.tobytes())
                    logging.debug(f"Published frame ({len(buffer)} bytes) to {args.subject}")
                except nats.errors.ConnectionClosedError:
                    logging.error("NATS connection closed while publishing. Attempting to reconnect...")
                    # Basic reconnect logic, could be more robust
                    try:
                        if not nc.is_connected: # Check if already reconnected by client
                            await nc.connect(servers=[nats_server_url], timeout=5) # Reconnect
                            logging.info("Reconnected to NATS server.")
                    except Exception as e_reconnect:
                        logging.error(f"Failed to reconnect to NATS: {e_reconnect}. Exiting loop.")
                        break # Exit while loop if reconnect fails
                except Exception as e:
                    logging.error(f"Error publishing to NATS: {e}")
                    # Depending on the error, might want to break or implement other handling
            else:
                logging.warning("JPEG encoding failed for a frame.")

            elapsed_time = time.monotonic() - loop_start_time
            sleep_duration = frame_duration - elapsed_time

            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                # Only log if significantly behind, not for minor fluctuations
                if sleep_duration < -0.005: # e.g., more than 5ms behind schedule
                     logging.warning(
                        f"Desired FPS ({args.fps}) too high for processing speed. "
                        f"Loop took {elapsed_time:.4f}s, target {frame_duration:.4f}s."
                    )
    except Exception as e:
        logging.error(f"An unexpected error occurred in async_main: {e}", exc_info=True)
    finally:
        logging.info("Shutting down NATS camera server...")
        if nc and nc.is_connected:
            try:
                await nc.drain() # Drain before closing for graceful publish completion
                logging.info("NATS connection drained.")
            except Exception as e:
                logging.error(f"Error during NATS drain: {e}")
            try:
                await nc.close()
                logging.info("NATS connection closed.")
            except Exception as e:
                logging.error(f"Error closing NATS connection: {e}")
        elif nc: # If nc exists but not connected (e.g. initial connection failed but nc object was created)
            try:
                await nc.close() # Attempt close anyway
            except Exception: # nosemgrep: geprüfter-assert
                pass # Ignore errors if already closed or uninitialized fully

        if cap and cap.isOpened():
            cap.release()
            logging.info("Camera released.")
        logging.info("async_main finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="NATS Camera Server")
    parser.add_argument(
        "--nats_ip", type=str, default="0.0.0.0", help="IP address of the NATS server."
    )
    parser.add_argument(
        "--nats_port", type=int, default=4222, help="Port of the NATS server."
    )
    parser.add_argument(
        "--camera_index", type=int, default=0, help="Index of the camera to use (e.g., 0, 1)."
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Width of the captured frames."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height of the captured frames."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target frames per second for capture and publishing."
    )
    parser.add_argument(
        "--subject", type=str, default="camera.image.raw", help="NATS subject to publish frames to."
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        choices=range(0, 101),
        metavar="[0-100]",
        help="JPEG encoding quality (0-100).",
    )

    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("Starting NATS Camera Server...")
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logging.info("Main process interrupted by KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}", exc_info=True)
    finally:
        # Ensure shutdown_event is set if loop exited due to unhandled exception in async_main
        # or if KeyboardInterrupt happened directly in asyncio.run() before async_main completes.
        if not shutdown_event.is_set():
             shutdown_event.set() 
        logging.info("NATS Camera Server has shut down.")

```
