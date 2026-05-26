import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft


def recursive_getattr(model, module_name):
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def t_product(A, B):
    assert A.dim() == 3 and B.dim() == 3
    assert A.shape[1] == B.shape[0]
    assert A.shape[2] == B.shape[2]

    m, p, k = A.shape
    _, n, _ = B.shape

    A_fft = torch.fft.fft(A, dim=2)   # full FFT
    B_fft = torch.fft.fft(B, dim=2)

    C_fft = torch.empty(m, n, k, dtype=A_fft.dtype, device=A_fft.device)
    for i in range(k):
        C_fft[:, :, i] = A_fft[:, :, i] @ B_fft[:, :, i]

    C = torch.fft.ifft(C_fft, dim=2).real
    return C


class LoraPTGroup(nn.Module):
    """
    t-SVD 版 LoRA-PT group：

    - 将一组同形状的 Linear 权重拼成三阶张量 W ∈ R^{out × in × L}
    - 对 W 做 t-SVD 分解（第三维做 FFT，逐频点做 SVD）
    - 取前 r 个主奇异值/向量，得到 U(out×r×L), S(r×r×L), V(r×in×L)
    - 构造低秩近似 W_low ≈ U * S * V（t-product）
    - 残差 R = W - W_low 冻结不训练
    - 微调时只更新 U, S, V（即 LoRA-PT 的可训练部分）
    """

    def __init__(self, weight_list, lora_dim=0, lora_scaling=1.0,init_from_tsvd=True):
        super().__init__()
        if lora_dim <= 0:
            raise ValueError("lora_dim must be > 0 for LoRA-PT")

        # 假设所有 Linear 的 weight 形状相同
        rows, columns = weight_list[0].shape
        for w in weight_list:
            assert w.shape == (rows, columns), \
                "All weights in a LoraPTGroup must have the same shape"

        self.rows = rows
        self.columns = columns
        self.lora_dim = lora_dim
        self.lora_scaling = lora_scaling
        self.num_layers = len(weight_list)
        self._delta_stack = None  # 缓存 U*S*V 的结果

        # W_stack: (out, in, L)
        W_stack = torch.stack([w.data.clone() for w in weight_list], dim=2)

        # 残差张量 R，register_buffer 保证冻结
        self.register_buffer("residual_stack",
                             torch.zeros_like(W_stack))

        # U: (out, r, L), S: (r, r, L), V: (r, in, L)
        self.lorapt_U_weight = nn.Parameter(
            torch.zeros(rows, lora_dim, self.num_layers))
        self.lorapt_S_weight = nn.Parameter(
            torch.zeros(lora_dim, lora_dim, self.num_layers))
        self.lorapt_V_weight = nn.Parameter(
            torch.zeros(lora_dim, columns, self.num_layers))

        # 用 t-SVD 对 W_stack 初始化 U,S,V 与 residual
        if init_from_tsvd:
            self._init_from_tsvd(W_stack)
        else:
            # 测试阶段：先用 W_stack 填 residual，U/S/V 先保持零，后面 load_state_dict 会覆盖
            self.residual_stack.data.copy_(W_stack)

    def _init_from_tsvd(self, W_stack: torch.Tensor):
        """
        对 W_stack ∈ R^{rows×columns×L} 做 t-SVD，并截断到秩 r：
        - 在第三维做 FFT -> (rows, columns, depth) complex
        - 每个频率切片做矩阵 SVD，取前 r 个奇异值
        - 得到 U_fft(rows×r×depth), S_fft(r×r×depth), V_fft(r×columns×depth)
        - 还原到时域得到 U,S,V，用于初始化可训练参数
        - 计算低秩近似 W_low，并得到 residual = W_stack - W_low
        """
        device = W_stack.device
        dtype = W_stack.dtype
        rows, columns, depth = W_stack.shape
        r = min(self.lora_dim, rows, columns)

        # 1) FFT 沿第三维
        W_fft = torch.fft.fft(W_stack, dim=2)  # complex64/complex128

        # 分配频域的 U,S,V
        U_fft = torch.zeros(rows, r, depth,
                            dtype=W_fft.dtype, device=device)
        S_fft = torch.zeros(r, r, depth,
                            dtype=W_fft.dtype, device=device)
        V_fft = torch.zeros(r, columns, depth,
                            dtype=W_fft.dtype, device=device)

        W_low_fft = torch.zeros_like(W_fft)

        for k in range(depth):
            # 每个频率切片：矩阵 SVD
            Wk = W_fft[:, :, k]  # (rows, columns), complex
            # full_matrices=False 得到 min(rows, columns) 维度
            Uk, Sk, Vhk = torch.linalg.svd(Wk, full_matrices=False)

            rk = min(r, Sk.numel())
            Uk_r = Uk[:, :rk]                       # (rows, rk)
            Sk_r = Sk[:rk].to(Wk.dtype)             # (rk,)
            Vhk_r = Vhk[:rk, :]                     # (rk, columns)

            # 写入 U_fft, S_fft, V_fft
            U_fft[:, :rk, k] = Uk_r
            S_fft[:rk, :rk, k] = torch.diag(Sk_r)
            # V 需要是 (r, columns)，对应 V^H 的共轭
            V_fft[:rk, :, k] = Vhk_r

            # 低秩近似在频域：U_r * diag(S_r) * Vh_r
            W_low_fft[:, :, k] = Uk_r @ torch.diag(Sk_r) @ Vhk_r

        # 2) IFFT 回到时域，取实部
        W_low = torch.fft.ifft(W_low_fft, dim=2).real.to(dtype)  # (rows, columns, depth)
        residual = W_stack - W_low

        U_spatial = torch.fft.ifft(U_fft, dim=2).real.to(dtype)
        S_spatial = torch.fft.ifft(S_fft, dim=2).real.to(dtype)
        V_spatial = torch.fft.ifft(V_fft, dim=2).real.to(dtype)

        # 写入参数和残差
        # 注意：如果 lora_dim > r，会有高维部分保持为 0
        self.lorapt_U_weight.data.zero_()
        self.lorapt_S_weight.data.zero_()
        self.lorapt_V_weight.data.zero_()

        self.lorapt_U_weight.data[:, :r, :] = U_spatial
        self.lorapt_S_weight.data[:r, :r, :] = S_spatial
        self.lorapt_V_weight.data[:r, :, :] = V_spatial

        self.residual_stack.data.copy_(residual)

    def _compute_delta_stack(self):
        """
        计算整组的低秩张量 ΔW_tensor ∈ R^{rows × columns × depth}：
        ΔW = U * S * V（沿最后一维做 t-product）
        """
        US = t_product(self.lorapt_U_weight, self.lorapt_S_weight)
        USV = t_product(US, self.lorapt_V_weight)
        return USV * self.lora_scaling

    def clear_delta_cache(self):
        self._delta_stack = None

    def get_delta_for_layer(self, layer_index: int):
        """
        取出某一层的低秩部分 ΔW_l ∈ R^{out × in}
        """
        if self._delta_stack is None:
            self._delta_stack = self._compute_delta_stack()
        delta_stack = self._delta_stack  # (rows, columns, depth)
        return delta_stack[:, :, layer_index]

    def get_residual_for_layer(self, layer_index: int):
        return self.residual_stack[:, :, layer_index]


class LinearLayer_LoraPT(nn.Module):
    """
    单层 LoRA-PT 线性层（t-SVD 版）：

    - residual_stack 中的残差 R_l 冻结不训练
    - U,S,V 属于共享 LoraPTGroup，更新时只改 U,S,V
    - 前向： y = (R_l + ΔW_l) x + b，其中 ΔW_l = (U*S*V)_l
    """

    def __init__(self,
                 bias,
                 group: LoraPTGroup,
                 layer_index: int,
                 lora_droppout: float = 0.0):
        super().__init__()
        self.bias = bias
        self.group = group
        self.layer_index = layer_index

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def _current_weight(self):
        """
        当前层的等效权重：W_l = R_l + ΔW_l
        """
        R_l = self.group.get_residual_for_layer(self.layer_index)
        delta = self.group.get_delta_for_layer(self.layer_index)
        return R_l + delta

    def fuse_lora_weight(self):
        """
        推理前可选：将当前层的 ΔW 直接加到残差上，得到一个等效的固定 W_l。
        """
        if not self.fuse_lora:
            with torch.no_grad():
                # 直接把 USV 叠加进 residual_stack 中
                delta = self.group.get_delta_for_layer(self.layer_index)
                self.group.residual_stack[:, :, self.layer_index] += delta
                # 清空缓存，避免重复使用旧 ΔW
                self.group.clear_delta_cache()
            self.fuse_lora = True

    def unfuse_lora_weight(self):
        # 如果需要，也可以实现为恢复，但一般推理阶段不会再 unfuse
        pass

    def forward(self, input):
        W_l = self._current_weight()
        if self.bias is None:
            return F.linear(self.lora_dropout(input), W_l, None)
        else:
            return F.linear(self.lora_dropout(input), W_l, self.bias)


def convert_linear_layer_to_lorapt(model,
                                   part_module_name,
                                   lora_dim=0,
                                   lora_scaling=1.0,
                                   lora_droppout=0.0,init_from_tsvd=True):
    """
    将模型中名字包含 part_module_name 的 nn.Linear 替换为 LoRA-PT 版本：
    1) 按权重尺寸 (out, in) 分组，每一组形成一个LoraPTGroup
    2) 组内所有 Linear 的 weight 拼成 3D 张量 W ∈ R^{out × in × L}
    3) 用 t-SVD 初始化 U,S,V 和 residual_stack
    4) 为每一层创建 LinearLayer_Lorapt，引用对应 group 和自己的 layer_index
    """
    shape_to_names = {}
    name_to_module = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            shape = tuple(module.weight.shape)  # (out, in)
            shape_to_names.setdefault(shape, []).append(name)
            name_to_module[name] = module

    if not shape_to_names:
        return model

    if not hasattr(model, "lorapt_groups"):
        model.lorapt_groups = nn.ModuleList()

    for shape, names in shape_to_names.items():
        weight_list = [name_to_module[n].weight for n in names]
        group = LoraPTGroup(weight_list,
                             lora_dim=lora_dim,
                             lora_scaling=lora_scaling,
                            init_from_tsvd=init_from_tsvd).to(
            weight_list[0].device).to(weight_list[0].dtype)

        model.lorapt_groups.append(group)

        # 逐个替换为 LinearLayer_Lorapt
        for idx, name in enumerate(names):
            module = name_to_module[name]
            # bias 保留原始值
            bias = module.bias
            if bias is not None:
                bias = nn.Parameter(bias.data.clone(), requires_grad=True)

            tmp = LinearLayer_LoraPT(
                bias=bias,
                group=group,
                layer_index=idx,
                lora_droppout=lora_droppout,
            ).to(weight_list[0].device).to(weight_list[0].dtype)

            recursive_setattr(model, name, tmp)

    return model


def only_optimize_lora_parameters(model, force_optimize_params=[]):
    """
    Vit网络中只训练 LoRA-PT 的 U,S,V（以及 force_optimize_params 中指定的参数），
    其它参数全部 requires_grad = False。
    """
    for name, param in model.named_parameters():
        if ("lorapt_U_weight" in name or
            "lorapt_S_weight" in name or
            "lorapt_V_weight" in name or
            name in force_optimize_params):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=[
            "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
            "ln_f.weight"
        ],
        lora_name_list=[
            "lorapt_U_weight", "lorapt_S_weight", "lorapt_V_weight"
        ],
):
    """
    为 LoRA-PT 设置不同学习率的参数组：
    - 普通参数：lr = base_lr, weight_decay = weight_decay
    - LoRA-PT 参数（U,S,V）：lr = lora_lr, weight_decay = weight_decay
    - 无权衰减参数（如 bias, norm）：weight_decay = 0
    """
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n for nd in lora_name_list))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n for nd in lora_name_list))
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups
