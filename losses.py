
#第二次改进后的损失函数
# import torch

# def calculate_emd(a: torch.Tensor, b: torch.Tensor, r: float = 2):
#     """
#     标准 EMD_r：
#     EMD_r(a,b) = ( mean_i |CDF_a(i) - CDF_b(i)|^r )^(1/r)
#     a, b: shape (K,)
#     return: scalar tensor
#     """
#     cdf_a = torch.cumsum(a, dim=0)
#     cdf_b = torch.cumsum(b, dim=0)
#     emd_r = torch.mean(torch.abs(cdf_a - cdf_b) ** r)
#     return torch.pow(emd_r, 1.0 / r)


# def earth_mover_distance_loss_batch(p: torch.Tensor, q: torch.Tensor, r: float = 2):
#     """
#     你的接口完全不变：
#     p, q: shape (B, K)
#     return: scalar tensor

#     加入“极端样本加权”(默认 linear)：
#       gt_score = sum_k (k * q_k)
#       mid = (K+1)/2  (K=10 时 mid=5.5)
#       weight = 1 + |gt_score - mid|
#       loss = mean_b [ weight_b * EMD_r(p_b, q_b) ]
#     """
#     loss_collector = []

#     B = p.size(0)
#     K = p.size(1)

#     # 用 1..K 当作 score_levels（K=10 就是 1..10）
#     score_levels = torch.arange(1,K+1, device=q.device, dtype=q.dtype)
#     mid = (K+1)/ 2.0 ##k=10时mid=5.5

#     for row in range(B):
#         emd = calculate_emd(p[row], q[row], r)

#         #权重不参与梯度（只用来放大极端样本的loss）
#         with torch.no_grad():
#             gt_score = (q[row] * score_levels).sum()        # 真实期望分
#             dist = torch.abs(gt_score - mid)               # 距离中间分的距离
#             weight = 1.0 + dist                            # linear 加权（默认）

#             # 如果你想用 quadratic，把上一行替换成：
#             # weight = 1.0 + dist ** 2##效果完全不行，这样导致模型更加只敢打中间分了，极端分的样本loss被放大了很多，模型不敢打极端分了，结果反而更差了。PLCC: 0.1470420732754023, SRCC: 0.13601518932592122
#             #
#             # 如果你想关掉加权，用：
#             # weight = 1.0

#         loss_collector.append(emd * weight)

#     return sum(loss_collector) / B



#标准EMD函数
import torch
import torch.nn as nn


def calculate_emd(a, b, r=2):
    """
    Standard 1D Earth Mover's Distance (Wasserstein) for equal-width bins.
    Input:  a, b shape [n]
    Output: scalar
    """
    # CDF difference (cumulative flow)
    cdf_diff = torch.cumsum(a - b, dim=0)  # [n]
    loss = torch.mean(torch.abs(cdf_diff) ** r)  # scalar
    return loss ** (1.0 / r)


def earth_mover_distance_loss_batch(p, q, r=2):
    """
    Batched version.
    Input:  p, q shape [batch, n]
    Output: scalar (mean over batch)
    """
    cdf_diff = torch.cumsum(p - q, dim=1)  # [batch, n]
    loss_per = torch.mean(torch.abs(cdf_diff) ** r, dim=1) ** (1.0 / r)  # [batch]
    return torch.mean(loss_per)

#第一次改进后的损失函数
# import torch
# import torch.nn as nn

# def calculate_emd(a,b,r=2):

# 	loss = 0.0
# 	for i in range(1, a.size(0)+1):
# 		loss += sum(torch.abs(a[:i] - b[:i])) ** r
# 	return (loss/a.size(0)) ** (1./r)

# def earth_mover_distance_loss_batch(p,q,r=2):

# 	loss_collector = []
# 	for row in range(p.size(0)):
# 		loss_collector.append(calculate_emd(p[row],q[row],r))

# 	return sum(loss_collector)/p.size(0)

