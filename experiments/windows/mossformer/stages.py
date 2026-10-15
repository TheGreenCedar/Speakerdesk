"""Export seams around the unchanged pinned ClearerVoice neural modules."""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.enc
        self.norm = model.mask_net.norm
        self.projection = model.mask_net.conv1d_encoder
        self.position = model.mask_net.pos_enc

    def forward(self, audio):
        encoded = self.encoder(audio)
        projected = self.projection(self.norm(encoded))
        residual = projected + self.position(projected.transpose(1, -1)).transpose(0, -1)
        return encoded, residual, residual.transpose(1, 2)


class Flash(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # Legacy ONNX export passes keyword-only defaults positionally at the root.
        return self.block(x, mask=None)


class Finish(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.intra_norm = model.mask_net.mdl.intra_mdl.norm
        self.global_norm = model.mask_net.mdl.intra_norm
        self.prelu = model.mask_net.prelu
        self.projection = model.mask_net.conv1d_out
        self.output = model.mask_net.output
        self.output_gate = model.mask_net.output_gate
        self.mask_projection = model.mask_net.conv1_decoder
        self.activation = model.mask_net.activation
        self.decoder = model.dec

    def forward(self, audio, encoded, residual, x):
        x = self.global_norm(self.intra_norm(x).transpose(1, 2)) + residual
        x = self.projection(self.prelu(x))
        x = x.reshape(2, 512, x.shape[-1])
        x = self.output(x) * self.output_gate(x)
        x = self.mask_projection(x)
        masks = self.activation(x.reshape(1, 2, 512, x.shape[-1])).transpose(0, 1)
        separated = torch.stack([encoded, encoded]) * masks
        estimates = torch.cat([self.decoder(separated[index]).unsqueeze(-1) for index in range(2)], dim=-1)
        # Kernel16/stride8 decoding leaves exactly the original 0–7 sample zero tail.
        estimates = F.pad(estimates, (0, 0, 0, audio.shape[1] - estimates.shape[1]))
        return estimates[:, :, 0], estimates[:, :, 1]


def padding_to_multiple(n, multiple):
    return (-n) % multiple


def adapt_for_export(model, block_module):
    # Trace both zero/nonzero padding through one tensor expression; never pad the acoustic window.
    block_module.padding_to_multiple_of = padding_to_multiple
    for module in model.modules():
        if hasattr(module, "cache_if_possible"):
            module.cache_if_possible = False


def program(model):
    blocks = model.mask_net.mdl.intra_mdl.mossformerM
    steps = [("encoder", Encoder(model))]
    for index, (flash, fsmn) in enumerate(zip(blocks.layers, blocks.fsmn)):
        steps.extend([(f"flash-{index:02d}", Flash(flash)), (f"fsmn-{index:02d}", fsmn)])
    steps.append(("finish", Finish(model)))
    return steps
