import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# 只需要修改这里
# =========================

IMAGE_DIR = r"D:\biyesheji\PyTorch-Neural-Image-Assessment-master\test\image"       # 测试集图片文件夹，里面是 jpg 图片
PRED_DIR = r"D:\biyesheji\PyTorch-Neural-Image-Assessment-master\test\_eval_out"        # 模型预测评分 txt 文件夹
GT_DIR = r"D:\biyesheji\PyTorch-Neural-Image-Assessment-master\test\score"            # 真实评分 txt 文件夹
OUTPUT_DIR = r"D:\biyesheji\PyTorch-Neural-Image-Assessment-master\test"       # 输出文件夹

NUM_IMAGES = 20                    # 输出前 20 张
TARGET_WIDTH = 1600                # 输出图片宽度
DPI = 300                          # 论文图片建议 300 dpi

# =========================
# 下面不用改
# =========================


def natural_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.stem)
    ]


def load_score_distribution(txt_path):
    probs = np.loadtxt(txt_path, dtype=np.float64).reshape(-1)

    if probs.size != 10:
        raise ValueError(f"{txt_path} 中不是 10 个概率值，而是 {probs.size} 个。")

    total = probs.sum()
    if total <= 0:
        raise ValueError(f"{txt_path} 中概率和小于等于 0。")

    return probs / total


def distribution_to_score(probs):
    score_levels = np.arange(1, 11, dtype=np.float64)
    return float(np.sum(score_levels * probs))


def get_font_path():
    """
    优先使用论文中常见的 Times New Roman / DejaVu Serif。
    如果系统没有这些字体，会自动退回默认字体。
    """
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesnewroman.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            return font_path

    return None


FONT_PATH = get_font_path()


def make_font(size):
    if FONT_PATH is not None:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_font(draw, text, max_width, max_height, start_size, min_size=12):
    """
    自动寻找不会超出指定宽高的字体大小。
    """
    size = start_size

    while size >= min_size:
        font = make_font(size)
        w, h = text_size(draw, text, font)

        if w <= max_width and h <= max_height:
            return font

        size -= 1

    return make_font(min_size)


def draw_center_text(draw, box, text, font, fill=(0, 0, 0)):
    """
    在 box 中居中画文字，保证不会从左侧或右侧跑出。
    """
    left, top, right, bottom = box
    w, h = text_size(draw, text, font)

    x = left + (right - left - w) / 2
    y = top + (bottom - top - h) / 2

    draw.text((x, y), text, font=font, fill=fill)


def create_annotated_image(image_path, gt_score, pred_score, output_path):
    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    if original_width != TARGET_WIDTH:
        scale = TARGET_WIDTH / original_width
        target_height = int(original_height * scale)
        image = image.resize(
            (TARGET_WIDTH, target_height),
            Image.Resampling.LANCZOS
        )
    else:
        target_height = original_height

    canvas_width = TARGET_WIDTH

    # 底部区域不要太矮，否则文字容易拥挤
    caption_height = max(240, int(target_height * 0.22))

    canvas_height = target_height + caption_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)

    border_width = max(2, canvas_width // 700)
    line_width = max(2, canvas_width // 900)

    # 外边框
    draw.rectangle(
        [0, 0, canvas_width - 1, canvas_height - 1],
        outline=(25, 25, 25),
        width=border_width
    )

    # 图片和标注区域分隔线
    draw.line(
        [(0, target_height), (canvas_width, target_height)],
        fill=(25, 25, 25),
        width=line_width
    )

    diff = pred_score - gt_score

    # =========================
    # 底部标注区域布局
    # =========================

    margin_x = int(canvas_width * 0.045)
    caption_top = target_height
    caption_bottom = canvas_height

    title_top = caption_top + int(caption_height * 0.06)
    title_bottom = caption_top + int(caption_height * 0.28)

    table_top = caption_top + int(caption_height * 0.34)
    table_bottom = caption_bottom - int(caption_height * 0.08)

    usable_left = margin_x
    usable_right = canvas_width - margin_x
    usable_width = usable_right - usable_left

    col_width = usable_width / 3

    col1 = (
        int(usable_left),
        int(table_top),
        int(usable_left + col_width),
        int(table_bottom)
    )

    col2 = (
        int(usable_left + col_width),
        int(table_top),
        int(usable_left + 2 * col_width),
        int(table_bottom)
    )

    col3 = (
        int(usable_left + 2 * col_width),
        int(table_top),
        int(usable_right),
        int(table_bottom)
    )

    columns = [col1, col2, col3]

    # 三列之间的分隔线
    for x in [
        int(usable_left + col_width),
        int(usable_left + 2 * col_width)
    ]:
        draw.line(
            [(x, table_top), (x, table_bottom)],
            fill=(180, 180, 180),
            width=line_width
        )

    # 标题
    title_text = f"Image ID: {image_path.stem}"
    title_font = fit_font(
        draw=draw,
        text=title_text,
        max_width=int(usable_width),
        max_height=int(title_bottom - title_top),
        start_size=int(canvas_width * 0.030),
        min_size=14
    )

    draw_center_text(
        draw,
        (usable_left, title_top, usable_right, title_bottom),
        title_text,
        title_font,
        fill=(40, 40, 40)
    )

    # 每列使用两行：指标名称 + 数值
    labels = [
        "Ground Truth",
        "Prediction",
        "Prediction - Ground Truth"
    ]

    values = [
        f"{gt_score:.3f}",
        f"{pred_score:.3f}",
        f"{diff:+.3f}"
    ]

    label_area_ratio = 0.45

    for col, label, value in zip(columns, labels, values):
        left, top, right, bottom = col

        inner_margin = int(col_width * 0.04)

        label_box = (
            left + inner_margin,
            top,
            right - inner_margin,
            int(top + (bottom - top) * label_area_ratio)
        )

        value_box = (
            left + inner_margin,
            int(top + (bottom - top) * label_area_ratio),
            right - inner_margin,
            bottom
        )

        label_font = fit_font(
            draw=draw,
            text=label,
            max_width=label_box[2] - label_box[0],
            max_height=label_box[3] - label_box[1],
            start_size=int(canvas_width * 0.026),
            min_size=12
        )

        value_font = fit_font(
            draw=draw,
            text=value,
            max_width=value_box[2] - value_box[0],
            max_height=value_box[3] - value_box[1],
            start_size=int(canvas_width * 0.038),
            min_size=14
        )

        draw_center_text(
            draw,
            label_box,
            label,
            label_font,
            fill=(70, 70, 70)
        )

        draw_center_text(
            draw,
            value_box,
            value,
            value_font,
            fill=(0, 0, 0)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(DPI, DPI))


def collect_matched_files():
    image_dir = Path(IMAGE_DIR)
    pred_dir = Path(PRED_DIR)
    gt_dir = Path(GT_DIR)

    if not image_dir.exists():
        raise FileNotFoundError(f"图片文件夹不存在：{image_dir}")

    if not pred_dir.exists():
        raise FileNotFoundError(f"预测评分文件夹不存在：{pred_dir}")

    if not gt_dir.exists():
        raise FileNotFoundError(f"真实评分文件夹不存在：{gt_dir}")

    image_files = {}

    for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        for image_path in image_dir.glob(pattern):
            image_files[image_path.stem] = image_path

    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.txt")}

    common_names = (
        set(image_files.keys())
        & set(pred_files.keys())
        & set(gt_files.keys())
    )

    common_names = sorted(
        common_names,
        key=lambda name: natural_key(Path(name))
    )

    if len(common_names) == 0:
        raise RuntimeError("没有找到图片、预测评分、真实评分三者名字都对应的文件。")

    matched_files = []

    for name in common_names:
        matched_files.append(
            (
                image_files[name],
                pred_files[name],
                gt_files[name]
            )
        )

    return matched_files


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_files = collect_matched_files()
    selected_files = matched_files[:NUM_IMAGES]

    print(f"共找到 {len(matched_files)} 组匹配文件。")
    print(f"开始输出前 {len(selected_files)} 张图片。")
    print(f"输出文件夹：{output_dir.resolve()}")

    for index, (image_path, pred_path, gt_path) in enumerate(selected_files, start=1):
        pred_probs = load_score_distribution(pred_path)
        gt_probs = load_score_distribution(gt_path)

        pred_score = distribution_to_score(pred_probs)
        gt_score = distribution_to_score(gt_probs)

        output_name = f"{index:02d}_{image_path.stem}_SCI.png"
        output_path = output_dir / output_name

        create_annotated_image(
            image_path=image_path,
            gt_score=gt_score,
            pred_score=pred_score,
            output_path=output_path
        )

        print(
            f"[{index:02d}] {image_path.name} | "
            f"GT={gt_score:.3f}, "
            f"Pred={pred_score:.3f}, "
            f"Diff={pred_score - gt_score:+.3f}"
        )

    print("全部完成。")


if __name__ == "__main__":
    main()