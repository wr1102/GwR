import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class Attention1dPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None, return_attn=True):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        if return_attn:
            return out, attn
        return out

class Attention1dPoolingProjection(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.25) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final(x)
        return x

class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(self,
                 hidden_dim,
                 num_labels,
                 linear_dropout):
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(hidden_dim)
        self.attention1d_projection = Attention1dPoolingProjection(
            hidden_dim, num_labels, linear_dropout)

    def forward(self, x, input_mask=None):
        x, token_atten = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x