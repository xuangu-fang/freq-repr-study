"""
Amplitude-phase autoencoder representation: learn nonlinear dimensionality reduction
using amplitude and phase as input channels.

振幅-相位自编码器表示：使用振幅和相位作为输入通道学习非线性降维。

设计特点：
1. 相位解缠绕：沿频率维度解缠绕相位，改善连续性
2. 双通道输入：振幅（非负）和相位（实数）作为两个通道
3. 与现有AutoencoderRepresentation共享架构，但输入处理不同
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import existing ConvAutoencoder
from .autoencoder_repr import ConvAutoencoder


class AmplitudePhaseAutoencoderRepresentation:
    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.
            Possible keys:
            - latent_dim: int (default 32) dimension of latent representation
            - learning_rate: float (default 1e-3)
            - num_epochs: int (default 50)
            - batch_size: int (default 32)
            - device: str (default 'cuda' if available else 'cpu')
            - phase_unwrap: bool (default True) whether to unwrap phase along frequency
        """
        self.config = config or {}
        self.latent_dim = int(self.config.get("latent_dim", 32))
        self.learning_rate = float(self.config.get("learning_rate", 1e-3))
        self.num_epochs = int(self.config.get("num_epochs", 50))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.phase_unwrap = bool(self.config.get("phase_unwrap", True))

        self.field_shape = None
        self.complex_input = False
        self.input_channels = 2  # Always 2 channels: amplitude and phase
        self.autoencoder = None

    def unwrap_phase_along_frequency(self, phase_trajectory):
        """
        沿频率维度解缠绕相位。

        Parameters
        ----------
        phase_trajectory : ndarray, shape (num_freq, H, W)
            相位轨迹

        Returns
        -------
        unwrapped_phase : ndarray, shape (num_freq, H, W)
            解缠绕后的相位
        """
        unwrapped = np.zeros_like(phase_trajectory)
        for i in range(phase_trajectory.shape[1]):  # H
            for j in range(phase_trajectory.shape[2]):  # W
                unwrapped[:, i, j] = np.unwrap(phase_trajectory[:, i, j])
        return unwrapped

    def fit(self, train_data, config=None):
        """
        Train autoencoder on training data.

        Parameters
        ----------
        train_data : list of ndarray
            List of 2D fields from training trajectories.
        config : dict, optional
            Additional configuration.
        """
        if config:
            self.config.update(config)
            self.latent_dim = int(self.config.get("latent_dim", 32))
            self.learning_rate = float(self.config.get("learning_rate", 1e-3))
            self.num_epochs = int(self.config.get("num_epochs", 50))
            self.batch_size = int(self.config.get("batch_size", 32))
            self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.phase_unwrap = bool(self.config.get("phase_unwrap", True))

        if not train_data:
            return

        self.field_shape = train_data[0].shape
        self.complex_input = np.iscomplexobj(train_data[0])

        if not self.complex_input:
            raise ValueError("AmplitudePhaseAutoencoderRepresentation requires complex input fields")

        # Prepare training data
        # Convert list to numpy array
        train_array = np.array(train_data)  # shape: (N, H, W)

        # Extract amplitude and phase
        amplitude = np.abs(train_array)  # shape: (N, H, W)
        phase = np.angle(train_array)    # shape: (N, H, W)

        # Note: In fit(), we have individual samples from different trajectories
        # and frequencies, so phase unwrapping along frequency dimension doesn't make sense here.
        # Phase unwrapping will be applied in transform() where we have complete trajectories.

        # Stack amplitude and phase along channel dimension
        train_tensor = np.stack([amplitude, phase], axis=1)  # shape: (N, 2, H, W)

        # Convert to torch tensor
        train_tensor = torch.FloatTensor(train_tensor).to(self.device)

        # Create autoencoder
        H, W = self.field_shape
        self.autoencoder = ConvAutoencoder(
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            input_height=H,
            input_width=W
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)

        # Create data loader
        dataset = TensorDataset(train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        self.autoencoder.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                inputs = batch[0]

                optimizer.zero_grad()
                outputs, _ = self.autoencoder(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Amplitude-phase autoencoder epoch {epoch+1}/{self.num_epochs}, loss: {avg_loss:.6f}")

    def transform(self, trajectory):
        """
        Transform a trajectory into autoencoder latent representation.

        Parameters
        ----------
        trajectory : list of ndarray
            List of 2D fields.

        Returns
        -------
        repr_trajectory : list of ndarray
            List of latent vectors.
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not fitted")

        if self.field_shape is None:
            self.field_shape = trajectory[0].shape
            self.complex_input = np.iscomplexobj(trajectory[0])

        if not self.complex_input:
            raise ValueError("AmplitudePhaseAutoencoderRepresentation requires complex input fields")

        # Prepare input data
        if isinstance(trajectory, list):
            traj_array = np.array(trajectory)  # shape: (num_freq, H, W)
        else:
            traj_array = trajectory

        num_freq = traj_array.shape[0]

        # Check shape consistency
        if traj_array.shape[1:] != self.field_shape:
            raise ValueError(
                f"Field shape {traj_array.shape[1:]} does not match stored shape {self.field_shape}"
            )

        # Extract amplitude and phase
        amplitude = np.abs(traj_array)  # shape: (num_freq, H, W)
        phase = np.angle(traj_array)    # shape: (num_freq, H, W)

        # Phase unwrapping along frequency dimension if enabled
        if self.phase_unwrap:
            phase = self.unwrap_phase_along_frequency(phase)

        # Stack amplitude and phase along channel dimension
        input_tensor = np.stack([amplitude, phase], axis=1)  # shape: (num_freq, 2, H, W)

        # Convert to torch tensor
        input_tensor = torch.FloatTensor(input_tensor).to(self.device)

        # Encode
        self.autoencoder.eval()
        with torch.no_grad():
            latent = self.autoencoder.encode(input_tensor)

        # Convert to numpy and return as list
        latent_np = latent.cpu().numpy()
        repr_traj = [latent_np[i] for i in range(num_freq)]

        return repr_traj

    def inverse_transform(self, repr_trajectory):
        """
        Reconstruct fields from autoencoder latent representation.

        Parameters
        ----------
        repr_trajectory : list of ndarray
            List of latent vectors.

        Returns
        -------
        trajectory : list of ndarray
            List of 2D fields.
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not fitted")

        # Prepare latent vectors
        latent_array = np.array(repr_trajectory)  # shape: (num_freq, latent_dim)
        latent_tensor = torch.FloatTensor(latent_array).to(self.device)

        # Decode
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder.decode(latent_tensor)

        # Convert to numpy
        reconstructed_np = reconstructed.cpu().numpy()

        # Post-process: split channels into amplitude and phase, then reconstruct complex field
        trajectory = []
        for i in range(reconstructed_np.shape[0]):
            amplitude = reconstructed_np[i, 0]
            phase = reconstructed_np[i, 1]
            # Reconstruct complex field
            field = amplitude * np.exp(1j * phase)
            trajectory.append(field)

        return trajectory

    def metadata(self):
        """
        Return metadata about the representation.

        Returns
        -------
        metadata : dict
            Dictionary with representation info.
        """
        return {
            "name": "amplitude_phase_autoencoder",
            "field_shape": self.field_shape,
            "complex_input": self.complex_input,
            "latent_dim": self.latent_dim,
            "phase_unwrap": self.phase_unwrap,
            "input_channels": self.input_channels,
            "config": self.config,
        }