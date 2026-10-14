"""Create the deterministic six-stage MovieLens project roadmap PNG."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "movielens_roadmap.png"
W, H = 1920, 1360
BG, NAVY, BLUE, TEAL, PURPLE, INDIGO, ROSE = "#f3f7fb", "#173e64", "#2f76b4", "#2b8c97", "#8063aa", "#5b72b4", "#ab4b76"
BORDER, TEXT, MUTED, WHITE = "#c8dced", "#234d72", "#587995", "#ffffff"
FONT = r"C:\Windows\Fonts\NotoSansSC-VF.ttf"
BOLD = r"C:\Windows\Fonts\simhei.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap(draw, text, f, width):
    lines, current = [], ""
    for char in text:
        if draw.textbbox((0, 0), current + char, font=f)[2] <= width:
            current += char
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def paragraph(draw, x, y, label, content, color):
    draw.text((x, y), label, font=font(18, True), fill=color)
    tx = x + 74
    lines = wrap(draw, content, font(18), 445)
    for i, line in enumerate(lines[:3]):
        draw.text((tx, y + i * 27), line, font=font(18), fill=MUTED)
    return y + max(28, len(lines[:3]) * 27)


def main():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    rounded(draw, (55, 38, 1865, 135), 22, NAVY)
    draw.text((90, 58), "MovieLens 分布式增量电影推荐系统", font=font(34, True), fill=WHITE)
    draw.text((90, 103), "从数据完整性到分布式、流式、图推荐与端到端消融的六阶段路线", font=font(18), fill="#dcecf8")
    draw.text((1620, 72), "可复现 → 可扩展", font=font(21, True), fill=WHITE)

    rounded(draw, (55, 168, 1865, 275), 18, "#e8f3fc", BORDER, 2)
    rounded(draw, (80, 192, 205, 228), 18, BLUE)
    draw.text((106, 198), "统一协议", font=font(16, True), fill=WHITE)
    draw.text((230, 190), "评分≥4为正反馈 · 按用户时间80/10/10划分 · 固定100负样本", font=font(21, True), fill=TEXT)
    draw.text((230, 228), "核心指标：NDCG@10 · 辅助指标：Recall@10、吞吐量与推荐延迟", font=font(17), fill=MUTED)

    cards = [
        ("阶段一  数据检查与基线", BLUE, "已完成", "建立可复现的评估基准", "四表完整性与主外键校验；80/10/10用户时序切分；seed=20260907固定100负样本", "质量报告、划分文件、Top10结果、NDCG@10基线", "1–2周"),
        ("阶段二  相似度与召回", BLUE, "planned", "降低全量相似度计算成本", "余弦/Jaccard精确计算；LSH近似候选；比较召回率、精度与延迟", "相似度模块、LSH代码、精度-效率报告", "2–3周"),
        ("阶段三  分布式推荐", TEAL, "planned", "把32M级数据处理搬到Spark", "Spark ALS或协同过滤；分布式数据处理；规模与节点吞吐实验", "Spark作业、算法源码、扩展性日志", "3–4周"),
        ("阶段四  流式增量", INDIGO, "planned", "模拟评分回放并更新兴趣状态", "时间戳回放；滑动窗口/衰减热度；受影响用户与物品增量更新", "流处理模拟器、状态日志、延迟报告", "3–4周"),
        ("阶段五  图推荐实验", PURPLE, "planned", "利用高阶用户-电影关系", "二部图；Personalized PageRank或随机游走；图特征消融", "图构建脚本、链接分析、对比实验", "2–3周"),
        ("阶段六  集成与消融", ROSE, "planned", "量化每个模块对最终效果的贡献", "LSH召回 + 粗排 + 精排；移除增量/图/LSH的消融；P99评估", "端到端系统、消融表、项目报告", "3–4周"),
    ]
    positions = [(55, 315), (675, 315), (1295, 315), (1295, 700), (675, 700), (55, 700)]
    for (title, color, status, goal, practice, deliverable, period), (x, y) in zip(cards, positions):
        rounded(draw, (x, y, x + 570, y + 315), 18, WHITE, BORDER, 2)
        draw.rectangle((x, y, x + 570, y + 62), fill=color)
        draw.text((x + 24, y + 16), title, font=font(23, True), fill=WHITE)
        yy = paragraph(draw, x + 25, y + 84, "目标", goal, color) + 13
        yy = paragraph(draw, x + 25, yy, "关键实践", practice, color) + 13
        paragraph(draw, x + 25, yy, "交付物", deliverable, color)
        draw.text((x + 25, y + 284), "周期  " + period, font=font(16, True), fill=color)

    def arrow(x1, y1, x2, y2, label):
        draw.line((x1, y1, x2, y2), fill="#7e9abd", width=4)
        if y1 == y2:
            direction = 1 if x2 > x1 else -1
            points = [(x2, y2), (x2 - 16 * direction, y2 - 10), (x2 - 16 * direction, y2 + 10)]
        else:
            direction = 1 if y2 > y1 else -1
            points = [(x2, y2), (x2 - 10, y2 - 16 * direction), (x2 + 10, y2 - 16 * direction)]
        draw.polygon(points, fill="#7e9abd")
        draw.text(((x1 + x2) // 2 - 25, (y1 + y2) // 2 - 27), label, font=font(15, True), fill="#6282a5")

    arrow(625, 472, 660, 472, "先修")
    arrow(1245, 472, 1280, 472, "核心依赖")
    arrow(1580, 640, 1580, 680, "进入系统层")
    arrow(1290, 857, 1255, 857, "图特征")
    arrow(670, 857, 635, 857, "最终交付")

    rounded(draw, (55, 1045, 1865, 1155), 18, "#e8f3fc", BORDER, 2)
    draw.text((80, 1065), "贯穿始终的支撑理论", font=font(20, True), fill=TEXT)
    for label, x, end in [("数据工程", 80, 200), ("推荐评价", 230, 360), ("分布式计算", 390, 550), ("流式与状态", 580, 740), ("图算法", 770, 890), ("工程可复现", 920, 1080)]:
        rounded(draw, (x, 1102, end, 1134), 16, "#5a84a9")
        draw.text((x + 15, 1108), label, font=font(15, True), fill=WHITE)
    draw.text((1110, 1105), "NDCG@10 / Recall@10 · 数据泄漏防护 · 资源与延迟 · 统一候选集", font=font(16), fill=MUTED)

    rounded(draw, (55, 1190, 1865, 1318), 18, NAVY)
    draw.text((82, 1212), "执行节奏", font=font(20, True), fill=WHITE)
    draw.text((82, 1252), "每阶段提交：脚本 + 数据清单/派生结果 + 日志 + 曲线 + README；先通过复现验收，再进入下一阶段。", font=font(17), fill="#e4eff8")
    draw.text((1635, 1240), "总计约 16–20 周", font=font(20, True), fill=WHITE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, dpi=(150, 150))
    print(OUT)


if __name__ == "__main__":
    main()
