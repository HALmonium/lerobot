import logging

import torch

from lerobot.common.robot_devices.robots.configs import DummyRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)


class DummyRobot(ManipulatorRobot):
    """
    Dummy robot for testing with real cameras but dummy (no-op) robot actions.
    Inherits from ManipulatorRobot but primarily focuses on camera operations
    and provides no-op implementations for robot actions.
    """

    def __init__(self, config: DummyRobotConfig):
        super().__init__(config)
        self.robot_type = "dummy_robot"
        # Ensure config is stored for later access if needed, super() should do this
        self.config: DummyRobotConfig = config

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError("DummyRobot is already connected.")

        logging.info(f"Attempting to connect {len(self.cameras)} cameras for DummyRobot...")
        for name, camera_instance in self.cameras.items():
            camera_type = camera_instance.config.type if hasattr(camera_instance, 'config') and hasattr(camera_instance.config, 'type') else 'N/A'
            logging.info(f"Connecting camera '{name}' of type: {camera_type}")
            try:
                camera_instance.connect()
            except Exception as e:
                logging.error(f"Failed to connect camera '{name}': {e}")
                # Depending on desired behavior, either raise e or try to disconnect already connected cameras
                # For simplicity, we'll log and continue, then set is_connected based on outcome.
                # However, a more robust approach might be to raise or ensure all cameras connect.
                # For this dummy robot, we will proceed and is_connected will be set at the end.
                # If any camera fails, connect might be considered partially successful or failed.
                # Let's assume for now that if any camera fails, the robot connect fails.
                # Re-raising the error after attempting to disconnect others might be an option.
                raise RobotDeviceNotConnectedError(f"Failed to connect camera '{name}' for DummyRobot.") from e


        self.is_connected = True
        logging.info("DummyRobot connected (cameras only).")

    def disconnect(self):
        # Allow disconnecting even if not fully connected, to clean up partial connections.
        # if not self.is_connected:
        #     # If strict adherence to connect-before-disconnect is required:
        #     # raise RobotDeviceNotConnectedError("DummyRobot is not connected.")
        #     # For robustness, allow disconnect to be called to clean up:
        #     logging.info("DummyRobot disconnect called but was not marked as fully connected.")
        #     # Still proceed to attempt camera disconnections.

        logging.info(f"Attempting to disconnect {len(self.cameras)} cameras for DummyRobot...")
        for name, camera_instance in self.cameras.items():
            camera_type = camera_instance.config.type if hasattr(camera_instance, 'config') and hasattr(camera_instance.config, 'type') else 'N/A'
            logging.info(f"Disconnecting camera '{name}' of type: {camera_type}")
            try:
                # Check if camera has a disconnect method and is connected (if camera tracks its own state)
                if hasattr(camera_instance, 'disconnect'):
                     # Ideal: if hasattr(camera_instance, 'is_connected') and camera_instance.is_connected:
                    camera_instance.disconnect()
            except Exception as e:
                logging.error(f"Error disconnecting camera '{name}': {e}")
                # Continue to disconnect other cameras

        self.is_connected = False
        logging.info("DummyRobot disconnected (cameras only).")

    def capture_observation(self) -> dict[str, torch.Tensor]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "DummyRobot is not connected. Call connect() before capturing observations."
            )

        obs_dict = {}
        for key, camera_instance in self.cameras.items():
            try:
                # Assuming camera.read() returns a numpy array HWC, channel last
                image = camera_instance.read()
                if image is None:
                    raise ValueError(f"Camera {key} read() returned None.")
                # Convert to torch tensor. Assuming HWC format from camera.
                # Policies might expect CHW, but dataset typically stores HWC.
                # Let's stick to HWC for now as per common camera output.
                obs_dict[f"observation.images.{key}"] = torch.from_numpy(image)
            except Exception as e:
                logging.error(f"Failed to read from camera '{key}': {e}")
                # Provide a zero image or raise error, depending on desired robustness.
                # For testing, raising an error might be better.
                # However, to match potential real-world partial failures, one might insert a placeholder.
                # For now, let's re-raise to make test failures explicit.
                raise RuntimeError(f"Failed to read from camera '{key}' during capture_observation.") from e


        # For DummyRobot, motor states are always empty, aligned with `features` override.
        obs_dict["observation.state"] = torch.empty(0, dtype=torch.float32)
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            # While send_action might not strictly need connection if it's a no-op,
            # it's good practice for robot methods.
            raise RobotDeviceNotConnectedError(
                 "DummyRobot is not connected. Call connect() before sending actions."
            )
        logging.debug(f"DummyRobot received action: {action}, but is not acting on it.")
        # Returns an empty tensor as per `features` override for action space.
        return torch.empty(0, dtype=torch.float32)

    @property
    def features(self):
        # Start with camera features from the parent class (ManipulatorRobot or its base)
        # ManipulatorRobot.camera_features is a @property itself.
        feat = {**super().camera_features}
        
        # Define observation.state for an empty state vector
        feat["observation.state"] = {"dtype": "float32", "shape": (0,), "names": []}
        
        # Define action for an empty action vector
        feat["action"] = {"dtype": "float32", "shape": (0,), "names": []}
        
        return feat

    def __del__(self):
        # Ensure resources are released if the object is garbage collected
        # This is a fallback; explicit disconnect() is preferred.
        if hasattr(self, 'is_connected') and self.is_connected:
            logging.info("DummyRobot.__del__ called while connected. Attempting disconnect.")
            try:
                self.disconnect()
            except Exception as e:
                # Suppress errors during __del__ as interpreter state might be unpredictable
                logging.error(f"Error during DummyRobot.__del__: {e}", exc_info=True)
```
