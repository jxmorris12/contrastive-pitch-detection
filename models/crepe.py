import functools

import torch
import torch.nn.functional as F

###########################################################################
# Model definition
###########################################################################

CREPE_MODEL_CAPACITIES = {
    'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
}

# TODO(jxm): remove this device statement once dynamic inference problem is fixed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CREPE(torch.nn.Module):
    """CREPE model definition"""

    def __init__(self, model='full', num_output_nodes=88, out_activation='sigmoid'):
        super().__init__()

        # Model-specific layer parameters
        assert model in CREPE_MODEL_CAPACITIES, f"unknown CREPE model size {model}"
        capacity = CREPE_MODEL_CAPACITIES[model]

        in_channels = [1] + [n * capacity for n in [32, 4, 4, 4, 8]]
        out_channels = [n * capacity for n in [32, 4, 4, 4, 8, 16]]
        self.in_features = 64*capacity

        # TODO: can remove the following lines if it runs. It's from the old CREPE impl,
        # just checking my math against theres
        if model == 'full':
            assert in_channels == [1, 1024, 128, 128, 128, 256]
            assert out_channels == [1024, 128, 128, 128, 256, 512]
            assert self.in_features == 2048
        elif model == 'tiny':
            assert in_channels == [1, 128, 16, 16, 16, 32]
            assert out_channels == [128, 16, 16, 16, 32, 64]
            assert self.in_features == 256

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
                                          eps=0.0010000000474974513,
                                          momentum=0.0)

        # Layer definitions
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = torch.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = torch.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = torch.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = torch.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])

        self.num_output_nodes = num_output_nodes
        self.classifier = torch.nn.Linear(
                    in_features=self.in_features,
                    out_features=self.num_output_nodes).to(device)
        
        assert out_activation in ['sigmoid', 'softmax', None]
        self.out_activation = out_activation

    def forward(self, x):
        # Forward pass through first five layers
        batch_size, frame_length = x.shape
        # Reshape into frames of length 1024 for CREPE
        padding = (0, (1024 - (frame_length%1024)) % 1024) # Pad last dimension along the end
        # breakpoint()
        x = F.pad(x, padding, "constant", 0.0)
        batch_size, frame_length = x.shape
        assert frame_length % 1024 == 0
        num_frames = int(frame_length / 1024)
        x = x.reshape(batch_size * num_frames, 1024)
        x = self.embed(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, num_frames, -1)
        x = x.mean(1)

        # Compute logits
        x = self.classifier(x)
        if self.out_activation == 'sigmoid':
            return torch.sigmoid(x)
        if self.out_activation == 'softmax':
            return F.softmax(x, dim=-1)
        else:
            return x

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)
        x = self.layer(x, self.conv6, self.conv6_BN)

        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))