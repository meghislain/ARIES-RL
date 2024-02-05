import gym
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

class Custom_CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(Custom_CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # self.cnn : extractor des 3 features concatenees, 

        for key, subspace in observation_space.spaces.items():
            print(key)
            if key == "doseMaps" or key == "incert":#is_image_space(subspace):
                extractors[key] = Custom_CNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
        
        # extractors["DoseMaps"] = NatureCNN(subspace, features_dim=cnn_output_dim)
        # total_concat_size += cnn_output_dim
        # extractors["beam_pos"] = nn.Flatten()
        # total_concat_size += get_flattened_obs_dim(subspace)
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        # obs (bp, dmp, dmt, p )
        # features = concat bp, dmp, dmt -> Bx3xHxW
        # fm = self.cnn(concat)
        # flatten fm
        #concat (flatten fm, p) Bx???????
        # print("yeee")

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
    
class Custom_CNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512,kernel_size: int = 3):
        super(Custom_CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # assert is_image_space(observation_space, check_channels=False), (
        #     "You should use NatureCNN "
        #     f"only with images not with {observation_space}"" )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
