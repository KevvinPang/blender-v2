"""Bound inference memory without splitting the camera's global time context.

Only frame-independent operations and token-wise MLPs are batched. All
inter-frame attention still sees every sampled frame in the original order.
Weights reside on CPU between stages; precision and camera weights are kept.
"""
import torch
from torch import nn


class TokenBatches(nn.Module):
    def __init__(self, module, size=2048):
        super().__init__()
        self.module, self.size = module, size

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        output = None
        for start in range(0, len(flat), self.size):
            value = self.module(flat[start:start + self.size])
            if output is None:
                output = value.new_empty((len(flat), value.shape[-1]))
            output[start:start + len(value)] = value
        return output.reshape(*shape[:-1], -1)


class StagedModule(nn.Module):
    def __init__(self, module, device, *, batch_size=0, label='', progress=False):
        super().__init__()
        self.module, self.device = module, device
        self.batch_size, self.label, self.progress = batch_size, label, progress

    def forward(self, x, *args, **kwargs):
        self.module.to(self.device)
        completed = False
        try:
            if not self.batch_size:
                output = self.module(x, *args, **kwargs)
                completed = True
                return output
            output = None
            for start in range(0, len(x), self.batch_size):
                value = self.module(x[start:start + self.batch_size], *args, **kwargs)
                if isinstance(value, dict):
                    value = value['x_norm_patchtokens']
                if output is None:
                    output = value.new_empty((len(x), *value.shape[1:]))
                output[start:start + len(value)] = value
                if self.progress:
                    print(f'V2_PROGRESS camera {self.label}: {min(start + self.batch_size, len(x))}/{len(x)}', flush=True)
            completed = True
            return output
        finally:
            self.module.cpu()
            if completed and self.label and not self.progress:
                print(f'V2_PROGRESS camera {self.label} complete', flush=True)


def prepare_camera_model(model, device, frame_batch_size=4):
    """Call after strict checkpoint loading, with the model still on CPU."""
    aggregator = model.aggregator
    # The camera head reads only the last layer, unlike dense depth prediction.
    model.dense_head = None
    aggregator.cached_layer_indices = {aggregator.depth - 1}
    aggregator.camera_token.data = aggregator.camera_token.data.to(device)
    aggregator.register_token.data = aggregator.register_token.data.to(device)
    aggregator._resnet_mean = aggregator._resnet_mean.to(device)
    aggregator._resnet_std = aggregator._resnet_std.to(device)
    aggregator.rope_embed.to(device)
    aggregator.patch_embed = StagedModule(aggregator.patch_embed, device,
        batch_size=frame_batch_size, label='image features', progress=True)
    for index in range(aggregator.depth):
        aggregator.frame_blocks[index] = StagedModule(aggregator.frame_blocks[index], device,
            batch_size=frame_batch_size)
        block = aggregator.inter_frame_blocks[index]
        block.mlp = TokenBatches(block.mlp)
        aggregator.inter_frame_blocks[index] = StagedModule(block, device,
            label=f'global layer {index + 1}/{aggregator.depth}')
    model.camera_head = StagedModule(model.camera_head, device, label='camera decode')
    # An autocast weight cache would retain staged GPU weights until the end.
    # The caller supplies an equivalent autocast context with caching disabled.
    model.autocast = False
    return model
