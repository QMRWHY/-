"""

CE-DIoU Loss Function for YOLOv8 (纯 NumPy 版本，无需 PyTorch)

==============================================================

在传统 DIoU 的基础上，增加四个角点的对齐惩罚项，使模型更精准地

学习目标的形状和方向，特别适合无人机航拍等小目标、形态多变的场景。


CE-DIoU = IoU - ( ρ²/c² + α/4 * Σ dᵢ² )

Loss    = 1 - CE-DIoU


其中 d₁、d₂、d₃、d₄ 分别为预测框与真实框四个角点的欧氏距离平方。

"""


import numpy as np



def box_corners(boxes: np.ndarray) -> np.ndarray:

    """

    从 [x1, y1, x2, y2] 格式的边界框提取四个角点坐标。

    Args:

        boxes: (N, 4)，格式为 [x1, y1, x2, y2]

    Returns:

        corners: (N, 4, 2)，每个框的左上、右上、左下、右下角坐标

                 维度 1 的顺序：[左上, 右上, 左下, 右下]

    """

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    corners = np.stack([

        np.stack([x1, y1], axis=-1),   # 左上

        np.stack([x2, y1], axis=-1),   # 右上

        np.stack([x1, y2], axis=-1),   # 左下

        np.stack([x2, y2], axis=-1),   # 右下

    ], axis=-2)  # (N, 4, 2)

    return corners



def box_corners_dist_sq(corners_pred: np.ndarray, corners_gt: np.ndarray) -> np.ndarray:

    """

    计算两两组角点之间的欧氏距离平方。

    Args:

        corners_pred: (N, 4, 2) 预测框四个角点

        corners_gt:   (N, 4, 2) 真实框四个角点

    Returns:

        dists: (N, 4) 四个角点的距离平方 d₁~d₄

    """

    diff = corners_pred - corners_gt                       # (N, 4, 2)

    dists_sq = np.sum(diff ** 2, axis=-1)                  # (N, 4)

    return dists_sq



def ce_diou_loss(

    pred_boxes: np.ndarray,

    gt_boxes:   np.ndarray,

    alpha:      float = 0.5,

    eps:        float = 1e-7,

) -> float:

    """

    计算 CE-DIoU（Corner-Enhanced DIoU）损失。


    CE-DIoU = IoU - ( ρ²/c² + α/4 * Σ dᵢ² )

    Loss    = 1 - CE-DIoU


    Args:

        pred_boxes: (N, 4) 预测边界框 [x1, y1, x2, y2]

        gt_boxes:   (N, 4) 真实边界框 [x1, y1, x2, y2]

        alpha:      角点惩罚项的权重系数

        eps:        防止除零的极小值

    Returns:

        loss: 标量损失（mean over batch）

    """

    # ---------- 1. IoU ----------

    x1_p, y1_p, x2_p, y2_p = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

    x1_g, y1_g, x2_g, y2_g = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]


    inter_x1 = np.maximum(x1_p, x1_g)

    inter_y1 = np.maximum(y1_p, y1_g)

    inter_x2 = np.minimum(x2_p, x2_g)

    inter_y2 = np.minimum(y2_p, y2_g)


    inter_w = np.maximum(inter_x2 - inter_x1, 0)

    inter_h = np.maximum(inter_y2 - inter_y1, 0)

    inter_area = inter_w * inter_h


    pred_area = (x2_p - x1_p) * (y2_p - y1_p)

    gt_area   = (x2_g - x1_g) * (y2_g - y1_g)

    union_area = pred_area + gt_area - inter_area


    iou = inter_area / (union_area + eps)                  # (N,)


    # ---------- 2. 中心点距离项 ρ²/c² ----------

    cx_p = (x1_p + x2_p) * 0.5

    cy_p = (y1_p + y2_p) * 0.5

    cx_g = (x1_g + x2_g) * 0.5

    cy_g = (y1_g + y2_g) * 0.5


    rho_sq = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2       # (N,)


    # 最小闭包区域对角线 c

    enclose_x1 = np.minimum(x1_p, x1_g)

    enclose_y1 = np.minimum(y1_p, y1_g)

    enclose_x2 = np.maximum(x2_p, x2_g)

    enclose_y2 = np.maximum(y2_p, y2_g)

    c = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2  # (N,)


    center_penalty = rho_sq / (c + eps)                    # (N,)


    # ---------- 3. 角点惩罚项 α/4 * Σ dᵢ² ----------

    corners_pred = box_corners(pred_boxes)                  # (N, 4, 2)

    corners_gt   = box_corners(gt_boxes)                   # (N, 4, 2)

    d_sum = np.sum(box_corners_dist_sq(corners_pred, corners_gt), axis=-1)  # (N,)


    corner_penalty = (alpha / 4.0) * d_sum                 # (N,)


    # ---------- 4. CE-DIoU & Loss ----------

    ce_diou = iou - (center_penalty + corner_penalty)      # (N,)

    loss = 1.0 - ce_diou                                   # (N,)


    return float(np.mean(loss))



def ce_diou_batch(

    pred_boxes: np.ndarray,

    gt_boxes:   np.ndarray,

    alpha:      float = 0.5,

    eps:        float = 1e-7,

) -> np.ndarray:

    """

    逐样本返回 CE-DIoU Loss（不取 mean），用于逐样本分析。

    """

    x1_p, y1_p, x2_p, y2_p = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

    x1_g, y1_g, x2_g, y2_g = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]


    inter_x1 = np.maximum(x1_p, x1_g)

    inter_y1 = np.maximum(y1_p, y1_g)

    inter_x2 = np.minimum(x2_p, x2_g)

    inter_y2 = np.minimum(y2_p, y2_g)


    inter_w = np.maximum(inter_x2 - inter_x1, 0)

    inter_h = np.maximum(inter_y2 - inter_y1, 0)

    inter_area = inter_w * inter_h


    pred_area = (x2_p - x1_p) * (y2_p - y1_p)

    gt_area   = (x2_g - x1_g) * (y2_g - y1_g)

    union_area = pred_area + gt_area - inter_area


    iou = inter_area / (union_area + eps)


    cx_p = (x1_p + x2_p) * 0.5

    cy_p = (y1_p + y2_p) * 0.5

    cx_g = (x1_g + x2_g) * 0.5

    cy_g = (y1_g + y2_g) * 0.5


    rho_sq = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2


    enclose_x1 = np.minimum(x1_p, x1_g)

    enclose_y1 = np.minimum(y1_p, y1_g)

    enclose_x2 = np.maximum(x2_p, x2_g)

    enclose_y2 = np.maximum(y2_p, y2_g)

    c = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2


    center_penalty = rho_sq / (c + eps)


    corners_pred = box_corners(pred_boxes)

    corners_gt   = box_corners(gt_boxes)

    d_sum = np.sum(box_corners_dist_sq(corners_pred, corners_gt), axis=-1)

    corner_penalty = (alpha / 4.0) * d_sum


    ce_diou = iou - (center_penalty + corner_penalty)

    loss = 1.0 - ce_diou


    return loss



# ─────────────────────────────────────────────────────────────

# 快速测试

# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    np.random.seed(42)


    # 模拟一批预测框和真实框 (N=8, 4)

    pred_boxes = np.array([

        [10, 10, 60, 60],

        [25, 30, 55, 70],

        [80, 20, 90, 40],

        [15, 15, 45, 45],

        [30, 40, 80, 90],

        [10, 20, 50, 60],

        [60, 10, 80, 30],

        [20, 30, 70, 80],

    ], dtype=np.float32)


    gt_boxes = np.array([

        [15, 15, 55, 55],

        [20, 25, 60, 75],

        [85, 25, 95, 35],

        [10, 10, 50, 50],

        [25, 35, 85, 95],

        [ 5, 15, 55, 65],

        [55,  5, 85, 35],

        [15, 25, 75, 85],

    ], dtype=np.float32)


    # ---------- 基础函数测试 ----------

    corners_pred = box_corners(pred_boxes)

    corners_gt   = box_corners(gt_boxes)

    d_sum = box_corners_dist_sq(corners_pred, corners_gt).sum(axis=-1)

    print("角点距离平方 d₁~d₄ 求和 (前4个样本):", d_sum[:4])


    loss_fn = lambda p, g: ce_diou_loss(p, g, alpha=0.5)

    loss = loss_fn(pred_boxes, gt_boxes)

    print(f"\nCE-DIoU Loss (α=0.5): {loss:.6f}")


    # ---------- α 敏感性展示 ----------

    print("\n── α 敏感性测试 ──")

    for alpha in [0.0, 0.25, 0.5, 1.0, 2.0]:

        l = ce_diou_loss(pred_boxes, gt_boxes, alpha=alpha)

        print(f"  α = {alpha:.2f}  →  Loss = {l:.6f}")

    # α=0 时退化为普通 DIoU loss


    # ---------- 与普通 DIoU 对比 ----------

    print("\n── 与普通 DIoU 对比 ──")

    l_diou = ce_diou_loss(pred_boxes, gt_boxes, alpha=0.0)

    l_ce   = ce_diou_loss(pred_boxes, gt_boxes, alpha=0.5)

    print(f"  DIoU Loss (α=0): {l_diou:.6f}")

    print(f"  CE-DIoU Loss (α=0.5): {l_ce:.6f}")

    print(f"  差值: {l_ce - l_diou:.6f}  (角点惩罚贡献)")


    # ---------- 逐样本分析 ----------

    print("\n── 逐样本 Loss 分解 ──")

    per_sample = ce_diou_batch(pred_boxes, gt_boxes, alpha=0.5)

    for i in range(len(pred_boxes)):

        print(f"  样本 {i}: Loss = {per_sample[i]:.6f}")