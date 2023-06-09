from typing import List, Tuple

import torch

import utilities


class Model:
    def __init__(
        self, path: str, device: torch.device, target_image: torch.Tensor,
        layer_weights: List[float] = [1e09, 1e09, 1e09, 1e09, 1e09],
        # layer_weights: List[float] = [0, 0, 0, 0, 1e09],
        # layer_weights: List[float] = [0, 0, 0, 1e09, 0],
        # layer_weights: List[float] = [0, 0, 1e09, 0, 0],
        # layer_weights: List[float] = [0, 1e09, 0, 0, 0],
        important_layers: List[str] = [
            'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
        ],
        topk = 0.2,
        reverse_topk = False
    ):  
        self.net = utilities.load_model(path).to(device).eval()
        self.device = device
        self.target_image = target_image.to(device)
        self.layer_weights = layer_weights
        self.important_layers = important_layers

        self.reverse_topk = reverse_topk
        # extract Gram matrices of the target image
        gram_hook = GramHook(topk=topk, reverse_topk=reverse_topk)
        gram_hook_handles = []
        for name, layer in self.net.named_children():
            if name in self.important_layers:
                handle = layer.register_forward_hook(gram_hook)
                gram_hook_handles.append(handle)
        self.net(self.target_image)

        # register Gram loss hook
        self.gram_loss_hook = GramLossHook(
            gram_hook.gram_matrices, layer_weights, important_layers,
            topk, reverse_topk
        )
        for handle in gram_hook_handles:    # Gram hook is not needed anymore
            handle.remove()

        for name, layer in self.net.named_children():
            if name in self.important_layers:
                handle = layer.register_forward_hook(self.gram_loss_hook)

        # remove unnecessary layers
        i = 0
        for name, layer in self.net.named_children():
            if name == important_layers[-1]:
                break
            i += 1
        self.net = self.net[:(i + 1)]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.gram_loss_hook.clear()

        return self.net(image)

    def get_loss(self) -> torch.Tensor:
        # return sum(self.gram_loss_hook.losses)
        return torch.stack(self.gram_loss_hook.losses, dim=0).sum(dim=0)


class ActivationsHook:
    def __init__(self):
        self.activations = []

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        self.activations.append(layer_out.detach())


class GramHook:
    def __init__(self, topk, reverse_topk):
        self.gram_matrices = []
        self.topk = topk
        self.reverse_topk = reverse_topk

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        # sparsity layer out
        layer_out = sparse_hw(layer_out, self.topk, reverse=self.reverse_topk, device='cuda')
        
        self.gram_matrices.append(layer_out.detach())


class GramLossHook:
    def __init__(
        self, target_gram_matrices: List[torch.Tensor],
        layer_weights: List[float], layer_names: List[str],
        topk: float,
        reverse_topk: bool
    ):
        self.target_gram_matrices = target_gram_matrices
        self.layer_weights = [
            weight * (1.0 / 4.0) for weight in layer_weights
        ]
        self.layer_names = layer_names
        self.losses: List[torch.Tensor] = []
        self.topk = topk
        self.reverse_topk = reverse_topk

    def __call__(
        self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
        layer_out: torch.Tensor
    ):
        i = len(self.losses)
        assert i < len(self.layer_weights)
        assert i < len(self.target_gram_matrices)

        if torch.isnan(layer_out).any():
            print('NaN in layer {}, NaN already in layer input: {}'.format(
                self.layer_names[i], torch.isnan(layer_in[0]).any()
            ))
        
        loss = self.layer_weights[i] * (
            (layer_out - self.target_gram_matrices[i])**2
        ).sum()
        self.losses.append(loss)

    def clear(self):
        self.losses = []


def sparse_hw(x, topk, reverse=False, device='cuda'):
    n, c, h, w = x.shape
    
    x_reshape = x.view(n, c, h * w)
    topk_keep_num = max(1, int(topk * h * w))
    _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
    mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(device)
    if reverse:
        mask = torch.ones_like(mask) - mask
    sparse_x = mask * x_reshape
    
    return sparse_x.view(n, c, h, w)