import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.configs import NatsCameraConfig # Keep NatsCameraConfig for example camera
from lerobot.common.robot_devices.robots.configs import DummyRobotConfig
from lerobot.common.robot_devices.robots.dummy_robot import DummyRobot


class TestDummyRobot(unittest.TestCase):
    def setUp(self):
        # Using NatsCameraConfig here as an example of *a* camera config.
        # The DummyRobot is generic and can take any CameraConfig.
        self.example_camera_config = NatsCameraConfig(
            nats_server_ip="127.0.0.1",
            nats_server_port=4222,
            subject="test.cam.subject",
            width=128,
            height=96,
            channels=3,
            color_mode="rgb",
            mock=True, 
        )
        # The mock=True on RobotConfig will propagate to camera configs
        self.robot_config = DummyRobotConfig(
            cameras={"test_cam": self.example_camera_config}, 
            mock=True 
        )
        self.dummy_image_np = np.random.randint(
            0, 256, (self.example_camera_config.height, self.example_camera_config.width, 3), dtype=np.uint8
        )

    def test_robot_instantiation(self):
        robot = DummyRobot(self.robot_config)
        self.assertIsNotNone(robot)
        self.assertEqual(robot.robot_type, "dummy_robot")
        self.assertIn("test_cam", robot.cameras)
        self.assertEqual(len(robot.cameras), 1)
        # Check if mock status propagated to camera instance config
        self.assertTrue(robot.cameras["test_cam"].config.mock)


    # Patching the specific camera type used in the config (NatsCamera here)
    # If testing with other camera types, these patches would need to change or be more generic.
    @patch("lerobot.common.robot_devices.cameras.nats.NatsCamera.disconnect")
    @patch("lerobot.common.robot_devices.cameras.nats.NatsCamera.connect")
    def test_connect_disconnect(self, mock_camera_connect, mock_camera_disconnect):
        # Since DummyRobot uses make_cameras_from_configs, which instantiates
        # the actual camera (NatsCamera in this test setup), we patch NatsCamera's methods.
        
        robot = DummyRobot(self.robot_config)
        
        # The actual camera instance is robot.cameras["test_cam"].
        # Its connect/disconnect methods should be the mocks.
        # Patching at the class level means all NatsCamera instances will use these mocks.
        
        robot.connect()
        self.assertTrue(robot.is_connected)
        mock_camera_connect.assert_called_once()

        robot.disconnect()
        self.assertFalse(robot.is_connected)
        mock_camera_disconnect.assert_called_once()


    @patch("lerobot.common.robot_devices.cameras.utils.make_cameras_from_configs")
    def test_capture_observation(self, mock_make_cameras):
        mock_camera_instance = MagicMock()
        mock_camera_instance.read.return_value = self.dummy_image_np
        # Configure the mock camera instance to also have a 'config' attribute with a 'type'
        mock_camera_instance.config = MagicMock()
        # Set a type for the mocked camera's config, e.g. "nats" or "opencv"
        mock_camera_instance.config.type = "test_camera_type" 

        # make_cameras_from_configs should return a dict of camera_name: mock_camera_instance
        mock_make_cameras.return_value = {"test_cam": mock_camera_instance}

        robot = DummyRobot(self.robot_config)
        
        # robot.connect() will call connect on the cameras returned by make_cameras_from_configs
        robot.connect() 
        mock_camera_instance.connect.assert_called_once() # Ensure robot.connect called camera's connect

        obs = robot.capture_observation()

        mock_camera_instance.read.assert_called_once()
        self.assertIn("observation.images.test_cam", obs)
        self.assertTrue(
            torch.equal(obs["observation.images.test_cam"], torch.from_numpy(self.dummy_image_np))
        )
        self.assertIn("observation.state", obs)
        self.assertEqual(obs["observation.state"].shape, (0,))
        self.assertEqual(obs["observation.state"].dtype, torch.float32)

        robot.disconnect() # Should call disconnect on the mock_camera_instance
        mock_camera_instance.disconnect.assert_called_once()


    def test_send_action(self):
        # Mock connect to avoid actual camera connection attempts during this test.
        with patch.object(DummyRobot, 'connect', MagicMock()): 
            robot = DummyRobot(self.robot_config)
            robot.is_connected = True # Manually set as we mocked connect

            test_action = torch.tensor([1.0, 2.0]) 
            returned_action = robot.send_action(test_action)

            self.assertEqual(returned_action.shape, (0,))
            self.assertEqual(returned_action.dtype, torch.float32)

    def test_features_property(self):
        robot = DummyRobot(self.robot_config)
        features = robot.features

        self.assertIn("observation.images.test_cam", features)
        self.assertEqual(features["observation.images.test_cam"]["shape"], (self.example_camera_config.height, self.example_camera_config.width, 3))
        self.assertEqual(features["observation.images.test_cam"]["dtype"], "uint8")


        self.assertIn("observation.state", features)
        self.assertEqual(features["observation.state"]["shape"], (0,))
        self.assertEqual(features["observation.state"]["dtype"], "float32")
        self.assertEqual(features["observation.state"]["names"], [])


        self.assertIn("action", features)
        self.assertEqual(features["action"]["shape"], (0,))
        self.assertEqual(features["action"]["dtype"], "float32")
        self.assertEqual(features["action"]["names"], [])


if __name__ == "__main__":
    unittest.main()
```
