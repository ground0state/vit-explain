"""
Copyright (c) 2021 Masafumi Abeta
Copyright (c) 2020 Jacob Gildenblat
"""
import numpy as np
import torch


def grad_rollout(attentions, gradients, head_fusion, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            if head_fusion == "mean":
                attention_heads_fused = (attention * weights).mean(axis=1)
            elif head_fusion == "max":
                # weights.mean(axis=1)*attention.max(axis=1)[0]
                attention_heads_fused = (attention * weights).max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = (attention * weights).min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(
                attention_heads_fused.size(0), -1)
            _, indices = flat.topk(
                int(flat.size(-1) * discard_ratio), -1, False)
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionGradRollout:
    def __init__(
            self,
            model,
            attention_layer_name='attn_drop',
            head_fusion='mean',
            discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward()

        return output, grad_rollout(
            self.attentions, self.attention_gradients,
            self.head_fusion, self.discard_ratio)
