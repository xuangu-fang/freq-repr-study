"""
Autoencoder representation: learn a nonlinear dimensionality reduction using neural networks.

自编码器表示：使用神经网络学习非线性降维表示。

物理意义：通过编码器-解码器架构学习数据的最紧凑表示。编码器将高维场
映射到低维潜空间，解码器从潜表示重建场。自编码器能够捕获非线性特征，
可能比PCA等线性方法提供更好的压缩。

优点：非线性表示能力，可以学习复杂的数据流形。
缺点：需要大量训练数据，训练时间长，可能过拟合，可解释性差。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ConvAutoencoder(nn.Module):
    """Simple convolutional autoencoder for 2D fields."""

    def __init__(self, input_channels=1, latent_dim=32, input_height=64, input_width=64):
        super().__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width

        # Encoder: 2 convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened size after convolutions
        # After 2 stride-2 convolutions: H_out = H_in // 4, W_out = W_in // 4
        self.encoder_output_size = 32 * (input_height // 4) * (input_width // 4)

        # Latent representation
        self.fc_encoder = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.encoder_output_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # No activation - output can be any real value
        )

    def encode(self, x):
        # Pass through encoder convolutions
        conv_features = self.encoder(x)

        # Map to latent space
        latent = self.fc_encoder(conv_features)
        return latent

    def decode(self, latent):
        # Map from latent space to decoder input
        decoder_input = self.fc_decoder(latent)

        # Reshape to expected decoder input shape
        batch_size = latent.shape[0]
        decoder_input = decoder_input.view(
            batch_size, 32,
            self.input_height // 4, self.input_width // 4
        )

        # Pass through decoder
        reconstructed = self.decoder(decoder_input)
        return reconstructed

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class AutoencoderRepresentation:
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
        """
        self.config = config or {}
        self.latent_dim = int(self.config.get("latent_dim", 32))
        self.learning_rate = float(self.config.get("learning_rate", 1e-3))
        self.num_epochs = int(self.config.get("num_epochs", 50))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.field_shape = None
        self.complex_input = False
        self.input_channels = 1  # Will be updated based on input
        self.autoencoder = None

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

        if not train_data:
            return

        self.field_shape = train_data[0].shape
        self.complex_input = np.iscomplexobj(train_data[0])

        # Prepare training data
        # Convert list to numpy array
        train_array = np.array(train_data)  # shape: (N, H, W)

        # Handle complex input
        if self.complex_input:
            # For complex fields, use real and imaginary parts as separate channels
            self.input_channels = 2
            # Stack real and imaginary parts along channel dimension
            train_tensor = np.stack([train_array.real, train_array.imag], axis=1)  # shape: (N, 2, H, W)
        else:
            # For real fields, add channel dimension
            self.input_channels = 1
            train_tensor = train_array[:, np.newaxis, :, :]  # shape: (N, 1, H, W)

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
                print(f"Autoencoder epoch {epoch+1}/{self.num_epochs}, loss: {avg_loss:.6f}")

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

        # Handle complex input
        if self.complex_input:
            # Stack real and imaginary parts
            input_tensor = np.stack([traj_array.real, traj_array.imag], axis=1)  # shape: (num_freq, 2, H, W)
        else:
            # Add channel dimension
            input_tensor = traj_array[:, np.newaxis, :, :]  # shape: (num_freq, 1, H, W)

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

        # Post-process based on input type
        trajectory = []
        for i in range(reconstructed_np.shape[0]):
            if self.complex_input:
                # Split channels into real and imaginary parts
                real_part = reconstructed_np[i, 0]
                imag_part = reconstructed_np[i, 1]
                field = real_part + 1j * imag_part
            else:
                # Remove channel dimension
                field = reconstructed_np[i, 0]

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
            "name": "autoencoder",
            "field_shape": self.field_shape,
            "complex_input": self.complex_input,
            "latent_dim": self.latent_dim,
            "input_channels": self.input_channels,
            "config": self.config,
        }