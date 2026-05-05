import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
import sys
from scipy.stats import rankdata

from PIL import Image

def compute_plcc(true_scores, pred_scores):
    true_mean = np.mean(true_scores)
    pred_mean = np.mean(pred_scores)
    numerator = np.sum((true_scores - true_mean) * (pred_scores - pred_mean))
    denominator = np.sqrt(np.sum((true_scores - true_mean) ** 2) * np.sum((pred_scores - pred_mean) ** 2))
    plcc = numerator / denominator
    return plcc

def compute_srcc(true_scores, pred_scores):
    x = np.asarray(true_scores, dtype=float)
    y = np.asarray(pred_scores, dtype=float)

    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt((rx @ rx) * (ry @ ry))
    if denom == 0:
        return np.nan
    return (rx @ ry) / denom

def true_vs_pred_plot(true_scores, pred_scores, out_path):
    plt.figure()
    plt.scatter(true_scores, pred_scores, c='blue', alpha=0.5)
    plt.plot([min(true_scores), max(true_scores)], [min(true_scores), max(true_scores)], 'r--')
    plt.xlabel('True Scores')
    plt.ylabel('Predicted Scores')
    plt.title('True vs Predicted Scores')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    
def pred_minus_true_vs_true_plot(true_scores, pred_scores, out_path):
    y_true = np.asarray(true_scores, dtype=float); y_pred = np.asarray(pred_scores, dtype=float)
    res = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, res, alpha=0.6)
    plt.axhline(0)
    plt.xlabel("True Score")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residual vs True")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()

def plot_score_distribution(true_scores, pred_scores, out_path):
    y_true = np.asarray(true_scores, dtype=float)
    y_pred = np.asarray(pred_scores, dtype=float)

    bins = np.linspace(
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
        21
    )

    plt.figure()
    plt.hist(y_true, bins=bins, alpha=0.5, label="True")
    plt.hist(y_pred, bins=bins, alpha=0.5, label="Pred")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def plot_error_by_bins(true_scores, pred_scores, metric, out_path):
    y_true = np.asarray(true_scores, dtype=float)
    y_pred = np.asarray(pred_scores, dtype=float)

    bins = np.linspace(y_true.min(), y_true.max(), 6)   # 5个区间
    idx = np.digitize(y_true, bins, right=True) - 1
    idx = np.clip(idx, 0, 4)   # 防止最大值落到区间外

    vals = []
    labels = []

    for b in range(5):
        mask = (idx == b)
        labels.append(f"{bins[b]:.2f}-{bins[b+1]:.2f}")

        if np.sum(mask) == 0:
            vals.append(np.nan)
        else:
            e = y_pred[mask] - y_true[mask]

            if metric.lower() == "rmse":
                vals.append(np.sqrt(np.mean(e ** 2)))
            elif metric.lower() == "mae":
                vals.append(np.mean(np.abs(e)))
            else:
                raise ValueError("metric must be 'rmse' or 'mae'")

    plt.figure()
    plt.bar(range(5), vals)
    plt.xticks(range(5), labels, rotation=30)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} by True Score Bin")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()




IMAGE_GLOB = r"D:\\biyesheji\\PyTorch-Neural-Image-Assessment-master\\test\\image\\*.jpg"
TXT_DIR    = r"D:\\biyesheji\\PyTorch-Neural-Image-Assessment-master\\test\\score"
PTH_PATH   = r"D:\\biyesheji\\PyTorch-Neural-Image-Assessment-master\\result\\snapshots\\Epoch4_0.pth"

OUT_DIR    = r"D:\\biyesheji\\PyTorch-Neural-Image-Assessment-master\\test\\_eval_out"



# =========================
# 2) 设备选择
# =========================
judgeNet = model.load_MobileNetV2_judge(pretrained=True).cuda()#model.JudgeMcJudgeFace().cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    normalize])

# =========================
# 7) 载入权重
# =========================
ckpt = torch.load(PTH_PATH)
judgeNet.load_state_dict(ckpt)
judgeNet.eval()

score_list = []
pred_score_list = []

image_list = glob.glob(IMAGE_GLOB)
for image_name in sorted(image_list):
    image_name=image_name.split('\\')[-1][:-4]
    txtFile = TXT_DIR +"\\"+ image_name +'.txt'
    scores = np.loadtxt(txtFile)
    score = round(sum([x*(i+1) for i,x in enumerate(scores)]),3)
    ##接下来要做的就是把图片丢进去模型，得到预测的分数
    img_path = IMAGE_GLOB[:-5] + image_name +'.jpg'
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        print(f"Error opening image: {img_path}")
        sys.exit(1)
    x=test_transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        y=judgeNet(x)
        pred_scores=y.cpu().numpy().squeeze()
        pred_score=round(sum([x*(i+1) for i,x in enumerate(pred_scores)]),3)
        print(f"True Score: {score}, Predicted Score: {pred_score}")
        np.savetxt(OUT_DIR + "\\" + image_name +'.txt', pred_scores)
        score_list.append(score)
        pred_score_list.append(pred_score)

# PLCC (Pearson Linear Correlation Coefficient) 计算
PLCC=compute_plcc(score_list, pred_score_list)
# SRCC (Spearman Rank Correlation Coefficient) 计算
SRCC=compute_srcc(score_list, pred_score_list)
print(f"PLCC: {PLCC}, SRCC: {SRCC}")
true_vs_pred_plot(score_list, pred_score_list, OUT_DIR + "\\compare\\pred_vs_true.png")##真实分数和预测分数的散点对比图
pred_minus_true_vs_true_plot(score_list, pred_score_list, OUT_DIR + "\\compare\\residual_vs_true.png")##预测分数减真实分数与真实分数的残差图
plot_score_distribution(score_list, pred_score_list, OUT_DIR + "\\compare\\score_distribution.png")##真实分数和预测分数的分布直方图
plot_error_by_bins(score_list, pred_score_list, "rmse", OUT_DIR + "\\compare\\rmse_by_bins.png")##各分数段的RMSE柱状图
plot_error_by_bins(score_list, pred_score_list, "mae", OUT_DIR + "\\compare\\mae_by_bins.png")##各分数段的MAE柱状图