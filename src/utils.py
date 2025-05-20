import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.items():
            if key == "image":
                # Create CNN layers before flattening
                cnn_layers = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1)),
                    nn.ReLU(),
                )
                
                # Calculate the shape after CNN layers using a dummy input
                sample_input = torch.zeros(1, *subspace.shape)
                with torch.no_grad():
                    output_shape = cnn_layers(sample_input).shape
                    print(output_shape)
                # Calculate flattened size
                flattened_size = int(np.prod(output_shape[1:]))
                
                # Now create the full network with correct sizes
                extractors[key] = nn.Sequential(
                    cnn_layers,
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(),
                )
                total_concat_size += 512
            elif key == "hvert":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], subspace.shape[0]))
                total_concat_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():

            encoded_tensor_list.append(extractor(observations[key]))
        
        return torch.cat(encoded_tensor_list, dim=1)
    
class DummyEnv(gym.Env):
    def __init__(self, obs):
        self.obs = obs
        if self.obs == "rgb_pos":
            self.observation_space = gym.spaces.Dict({
                    "image": gym.spaces.Box(low=0, high=255, shape=(3, 214, 214), dtype=np.uint8),
                    "hvert": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype='float32')
                })
        elif self.obs == "pos":
            self.observation_space =  gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype='float32')
            
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    
    def reset(self, **kwargs):
        if self.obs == "rgb_pos":
            return {
                "image": np.zeros((3, 214, 214), dtype=np.uint8),
                "hvert": np.zeros(9, dtype=np.float32)
            }, {}
        elif self.obs == "pos":
            return {np.zeros(12, dtype=np.float32)
        }, {}
    
    def step(self, action):
        if self.obs == "rgb_pos":
            obs = {
                "image": np.zeros((3, 214, 214), dtype=np.uint8),
                "hvert": np.zeros(9, dtype=np.float32)
            }
        elif self.obs == "pos":
            obs = np.zeros(12, dtype=np.float32)
        return obs, 0.0, False, False, {}

# Create a new model with the dummy environment
# dummy_env = DummyEnv()
