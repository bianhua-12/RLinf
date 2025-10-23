import unittest
import numpy as np
import threading
from unittest.mock import MagicMock, patch
import sys


mock_envs = MagicMock()
sys.modules["envs"] = mock_envs
mock_gen_instr = MagicMock()
sys.modules["generate_episode_instructions"] = mock_gen_instr
import os
os.environ["PYTHONPATH"] = os.getcwd()

try:
    from RoboTwin_wrapper import RobotwinEnvWrapper, get_robotwin2_task
except NameError:
    raise ImportError("RobotwinEnvWrapper not found")


class TestRobotwinEnvWrapper(unittest.TestCase):
    def setUp(self):
        
        self.task_name = "CloseDrawer"
        self.trial_id = 0
        self.trial_seed = 42
        self.config = {"twin2_task_config": "test_config"}

        
        self.mock_env_instance = MagicMock()
        self.mock_env_instance.step_lim = 100  
        self.mock_env_instance.eval_success = False 
        self.mock_env_instance.get_obs.return_value = np.zeros((10,)) 
        self.mock_env_instance.get_info.return_value = {"mock_info": "data"}
        
        self.mock_args = {"instruction_type": "default"}
        
        self.wrapper = RobotwinEnvWrapper(
             self.task_name, self.trial_id, self.trial_seed, self.config, version="2.0"
        )

    @patch("generate_episode_instructions.generate_episode_descriptions")
    @patch("RoboTwin_wrapper.get_robotwin2_task") 
    def test_initialize(self, mock_get_task, mock_gen_desc):
        print("\n--- Testing Initialize ---")
        mock_get_task.return_value = (self.mock_env_instance, self.mock_args)
        mock_gen_desc.return_value = [{"default": ["Mock Instruction"]}] 
        self.wrapper.initialize()

        self.mock_env_instance.setup_demo.assert_called_once()
        self.mock_env_instance.set_instruction.assert_called_with(instruction="Mock Instruction")
        print("Initialization success.")

    def _init_wrapper(self):
        
        with patch("RoboTwin_wrapper.get_robotwin2_task", return_value=(self.mock_env_instance, self.mock_args)):
            with patch("generate_episode_instructions.generate_episode_descriptions", return_value=[{"default": ["Instr"]}]):
                self.wrapper.initialize()

    def test_step_normal(self):
        """测试正常的 Step (未完成，未超时)"""
        print("\n--- Testing Normal Step ---")
        self._init_wrapper()
        
        # 模拟输入动作
        action = np.ones((5, 7)) # 假设动作是 (轨迹长度, 动作维度)
        
        # 执行一步
        obs, reward, terminated, truncated, info = self.wrapper.step(action)

        # 验证
        self.assertFalse(terminated, "Normal step should not be terminated")
        self.assertFalse(truncated, "Normal step should not be truncated")
        self.assertEqual(reward, 0.0, "Normal step reward should be 0.0 (sparse)")
        self.assertTrue(self.wrapper.active, "Wrapper should remain active")
        self.assertEqual(self.wrapper.finish_step, 5, "Finish step count should update correctly")

    def test_step_success(self):
        """测试任务成功的情况"""
        print("\n--- Testing Success Step ---")
        self._init_wrapper()
        action = np.ones((1, 7))

        # 模拟环境返回成功
        self.mock_env_instance.eval_success = True
        
        obs, reward, terminated, truncated, info = self.wrapper.step(action)

        self.assertTrue(terminated, "Success should result in terminated=True")
        self.assertFalse(truncated, "Success should NOT be truncated") 
        self.assertEqual(reward, 1.0, "Success reward should be 1.0")
        self.assertTrue(info['success'])
        self.assertFalse(self.wrapper.active, "Wrapper should be inactive after success")

    def test_step_timeout(self):
        """测试超时的情况"""
        print("\n--- Testing Timeout Step ---")
        self._init_wrapper()
        
        # 设置当前步数临近超时
        self.wrapper.finish_step = 99 
        self.mock_env_instance.step_lim = 100
        
        action = np.ones((2, 7)) # 执行2步，总步数将达到 101，超过 100
        
        obs, reward, terminated, truncated, info = self.wrapper.step(action)

        self.assertFalse(terminated, "Timeout (without success) should not be terminated (depends on definition, but usually it is just truncated)")
        self.assertTrue(truncated, "Timeout must result in truncated=True")
        self.assertEqual(reward, 0.0, "Timeout should have 0 reward")
        self.assertTrue(info['timeout'])
        self.assertFalse(self.wrapper.active)

    def test_step_exception(self):
        """测试底层环境发生异常时，Wrapper 是否能安全处理"""
        print("\n--- Testing Exception Handling in Step ---")
        self._init_wrapper()
        action = np.zeros((1, 7))

        # 让 take_action 抛出异常
        self.mock_env_instance.take_action.side_effect = RuntimeError("Simulator crashed!")

        # 执行 step，不应导致整个程序崩溃
        try:
            obs, reward, terminated, truncated, info = self.wrapper.step(action)
            print("Successfully caught exception in step.")
        except Exception as e:
            self.fail(f"Wrapper.step failed to catch exception: {e}")

        # 验证返回了安全的默认值 (根据您目前的实现，异常可能会导致它继续往下走，
        # 因为 try-except 只是打印了错误并将 done=False。
        
        self.assertIsNotNone(obs) # 因为您后面又 try 了一次 get_obs
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)

    def test_close(self):
        """测试关闭环境"""
        print("\n--- Testing Close ---")
        self._init_wrapper()
        self.wrapper.close()
        self.mock_env_instance.close_env.assert_called_once()


if __name__ == '__main__':
    unittest.main()