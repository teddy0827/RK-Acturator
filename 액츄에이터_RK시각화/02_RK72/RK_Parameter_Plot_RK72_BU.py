"""
ASML Scanner Overlay 보정 모델: RK Parameter Grid Deformation 시각화 (0~7차, K1~K72)
dx / dy 분리 출력 (2개 창)

출처:
  K1~K20  : Mulkens et al., SPIE Proc. 8683 (2013) Table 1  — 수식 & 액추에이터 확인
  K21~K42 : RSC Nanoscale Advances (2025, D5NA00682A)        — 수식 확인
  K43~K72 : 다항식 패턴 유추
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 유틸리티 함수 =====
def _vpow(var, p):
    """변수 거듭제곱 문자열: _vpow('x',3) → 'x3', p=0 → ''"""
    if p == 0:
        return ""
    if p == 1:
        return var
    return f"{var}{p}"


def _formula(direction, k, xp, yp):
    """수식 문자열 생성"""
    if direction == 'dx':
        parts = [s for s in [_vpow('x', xp), _vpow('y', yp)] if s]
    else:
        parts = [s for s in [_vpow('y', yp), _vpow('x', xp)] if s]
    term = '*'.join(parts) if parts else '1'
    return f"{direction} = K{k}*{term}"


def _ordinal(n):
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    if n == 3:
        return "3rd"
    return f"{n}th"


def _phys_name(order, xp, yp, direction):
    """물리적 의미 이름 생성 (K21+ 용)"""
    o = _ordinal(order)
    if direction == 'dx':
        if yp == 0:
            return f"{o} Ord Dist X"
        if xp == 0:
            return f"{o} Ord Flow X"
        return f"{o} Ord Mixed X"
    else:
        if xp == 0:
            return f"{o} Ord Dist Y"
        if yp == 0:
            return f"{o} Ord Flow Y"
        return f"{o} Ord Mixed Y"


def _make_dx(xp, yp):
    """dx 변위 함수 생성"""
    def f(x, y):
        v = np.ones_like(x)
        if xp > 0:
            v = v * x**xp
        if yp > 0:
            v = v * y**yp
        return (v, np.zeros_like(y))
    return f


def _make_dy(xp, yp):
    """dy 변위 함수 생성"""
    def f(x, y):
        v = np.ones_like(x)
        if xp > 0:
            v = v * x**xp
        if yp > 0:
            v = v * y**yp
        return (np.zeros_like(x), v)
    return f


# ===== K1~K20: 논문 확인 (SPIE 2013 Table 1) =====
# (K이름, 물리적의미, 수식, 함수, 액추에이터, 차수)
# K21 이후는 액추에이터를 ""(빈 문자열)로 설정
params = [
    # --- 0차 ---
    ("K1",  "Translation X",    "dx = K1",       lambda x, y: (np.ones_like(x), np.zeros_like(y)),  "Scan",      0),
    ("K2",  "Translation Y",    "dy = K2",       lambda x, y: (np.zeros_like(x), np.ones_like(y)),  "Scan",      0),
    # --- 1차 ---
    ("K3",  "Magnification X",  "dx = K3*x",     lambda x, y: (x, np.zeros_like(y)),               "Lens",      1),
    ("K4",  "Magnification Y",  "dy = K4*y",     lambda x, y: (np.zeros_like(x), y),               "Scan",      1),
    ("K5",  "Rotation X",       "dx = -K5*y",    lambda x, y: (-y, np.zeros_like(y)),              "Scan",      1),
    ("K6",  "Rotation Y",       "dy = K6*x",     lambda x, y: (np.zeros_like(x), x),               "Scan",      1),
    # --- 2차 ---
    ("K7",  "2nd Ord Mag X",    "dx = K7*x2",    lambda x, y: (x**2, np.zeros_like(y)),            "Lens",      2),
    ("K8",  "2nd Ord Mag Y",    "dy = K8*y2",    lambda x, y: (np.zeros_like(x), y**2),            "Scan",      2),
    ("K9",  "Trapezoid X",      "dx = K9*x*y",   lambda x, y: (x*y, np.zeros_like(y)),             "Scan Lens", 2),
    ("K10", "Trapezoid Y",      "dy = K10*y*x",  lambda x, y: (np.zeros_like(x), y*x),             "Scan",      2),
    ("K11", "Bow X",            "dx = K11*y2",   lambda x, y: (y**2, np.zeros_like(y)),             "Scan",      2),
    ("K12", "Bow Y",            "dy = K12*x2",   lambda x, y: (np.zeros_like(x), x**2),            "Lens",      2),
    # --- 3차 ---
    ("K13", "3rd Ord Mag X",    "dx = K13*x3",   lambda x, y: (x**3, np.zeros_like(y)),            "Lens",      3),
    ("K14", "3rd Ord Mag Y",    "dy = K14*y3",   lambda x, y: (np.zeros_like(x), y**3),            "Scan",      3),
    ("K15", "Accordion X",      "dx = K15*x2*y", lambda x, y: (x**2*y, np.zeros_like(y)),          "Scan Lens", 3),
    ("K16", "Accordion Y",      "dy = K16*y2*x", lambda x, y: (np.zeros_like(x), y**2*x),          "Scan",      3),
    ("K17", "C-shape X",        "dx = K17*x*y2", lambda x, y: (x*y**2, np.zeros_like(y)),          "Scan Lens", 3),
    ("K18", "C-shape Y",        "dy = K18*y*x2", lambda x, y: (np.zeros_like(x), y*x**2),          "Scan Lens", 3),
    ("K19", "3rd Ord Flow X",   "dx = K19*y3",   lambda x, y: (y**3, np.zeros_like(y)),             "Scan",      3),
    ("K20", "3rd Ord Flow Y",   "dy = K20*x3",   lambda x, y: (np.zeros_like(x), x**3),            "— (None)",  3),
]

# ===== K21~K72: 프로그래밍 생성 (4~7차) — 액추에이터 미기재("") =====
for order in range(4, 8):
    for i in range(order + 1):
        k_start = order * (order + 1) + 1
        k_dx = k_start + 2 * i
        k_dy = k_dx + 1

        xp_dx, yp_dx = order - i, i
        xp_dy, yp_dy = i, order - i

        params.append((
            f"K{k_dx}", _phys_name(order, xp_dx, yp_dx, 'dx'),
            _formula('dx', k_dx, xp_dx, yp_dx),
            _make_dx(xp_dx, yp_dx), "", order
        ))
        params.append((
            f"K{k_dy}", _phys_name(order, xp_dy, yp_dy, 'dy'),
            _formula('dy', k_dy, xp_dy, yp_dy),
            _make_dy(xp_dy, yp_dy), "", order
        ))


# ===== dx / dy 분리 =====
# params 리스트에서 짝수 인덱스(0,2,4,...)=dx, 홀수 인덱스(1,3,5,...)=dy
dx_params = params[0::2]   # K1, K3, K5, K7, ... K71  (36개)
dy_params = params[1::2]   # K2, K4, K6, K8, ... K72  (36개)


# ===== 시각화 설정 =====
N_GRID = 7
N_FINE = 50
SCALE  = 0.18
GRID_RANGE = 1.0
PLOT_MARGIN = 0.35

grid_vals = np.linspace(-GRID_RANGE, GRID_RANGE, N_GRID)
fine_vals = np.linspace(-GRID_RANGE, GRID_RANGE, N_FINE)

ACT_COLORS = {
    'Scan':      ('#2e7d32', '#e8f5e9'),
    'Lens':      ('#1565c0', '#e3f2fd'),
    'Scan Lens': ('#e65100', '#fff3e0'),
    '— (None)':  ('#c62828', '#ffebee'),
}

ORDER_BG = ['#ffffff', '#fcfcfc', '#f8f8f8', '#f5f5f5',
            '#fff8f0', '#f0fff4', '#f5f0ff', '#fff0f0']


def draw_single(ax, k_name, phys_name, formula, func, actuator, order):
    """하나의 K 파라미터에 대한 격자 변형을 그린다."""
    ax.set_facecolor(ORDER_BG[order])
    ax.set_xlim(-GRID_RANGE - PLOT_MARGIN, GRID_RANGE + PLOT_MARGIN)
    ax.set_ylim(-GRID_RANGE - PLOT_MARGIN, GRID_RANGE + PLOT_MARGIN)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # --- 원래 격자 (회색) ---
    for v in grid_vals:
        ax.plot([-GRID_RANGE, GRID_RANGE], [v, v], color='#cccccc', linewidth=0.6)
        ax.plot([v, v], [-GRID_RANGE, GRID_RANGE], color='#cccccc', linewidth=0.6)
    gx0, gy0 = np.meshgrid(grid_vals, grid_vals)
    ax.plot(gx0, gy0, 'o', color='#bbbbbb', markersize=1.5, zorder=2)

    # --- 변형된 격자 (검정) ---
    for v in grid_vals:
        xs = fine_vals.copy()
        ys = np.full_like(xs, v)
        ddx, ddy = func(xs, ys)
        ax.plot(xs + ddx * SCALE, ys + ddy * SCALE, color='#111111', linewidth=1.0, zorder=3)
    for v in grid_vals:
        ys = fine_vals.copy()
        xs = np.full_like(ys, v)
        ddx, ddy = func(xs, ys)
        ax.plot(xs + ddx * SCALE, ys + ddy * SCALE, color='#111111', linewidth=1.0, zorder=3)
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    ddx, ddy = func(gx, gy)
    ax.plot(gx + ddx * SCALE, gy + ddy * SCALE, 'o', color='#111111', markersize=1.8, zorder=4)

    # --- 제목 ---
    ax.set_title(f"{k_name} — {phys_name}\n{formula}",
                 fontsize=6, fontweight='bold', pad=3)

    # --- 액추에이터 배지 (K1~K20만 표시) ---
    if actuator:
        txt_c, bg_c = ACT_COLORS.get(actuator, ('#333', '#eee'))
        ax.text(0, -GRID_RANGE - 0.22, actuator, ha='center', va='center',
                fontsize=5, fontweight='bold', color=txt_c,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=bg_c,
                          edgecolor=txt_c, linewidth=0.6))

    # --- 좌표축 라벨 ---
    ax.text(GRID_RANGE + 0.28, -GRID_RANGE - 0.32, 'x(Slit)',
            fontsize=4, color='#999', ha='right')
    ax.text(-GRID_RANGE - 0.28, GRID_RANGE + 0.28, 'y(Scan)',
            fontsize=4, color='#999', ha='left')


def draw_pyramid(fig, param_list, title_text, save_path):
    """피라미드 레이아웃: 항목을 한 칸씩 띄워 트리 형태로 배치
    0차:                  1개  → col 7
    1차:                2개    → cols 6, 8
    2차:              3개      → cols 5, 7, 9
    ...
    7차: 8개                   → cols 0, 2, 4, 6, 8, 10, 12, 14
    """
    DATA_COLS = 15   # 8개 항목을 2칸 간격: 0,2,4,...,14
    TOTAL_COLS = DATA_COLS + 1  # col 0 = 라벨
    CENTER = 7       # 데이터 영역 중심 (0~14의 중앙)

    fig.suptitle(title_text, fontsize=12, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(8, TOTAL_COLS, figure=fig, hspace=0.55, wspace=0.05,
                           left=0.03, right=0.97, top=0.93, bottom=0.05,
                           width_ratios=[0.4] + [1] * DATA_COLS)

    # 차수별 그룹화
    order_groups = {}
    for p in param_list:
        order_groups.setdefault(p[5], []).append(p)

    for order in range(8):
        items = order_groups[order]
        n_items = len(items)   # = order + 1

        # 한 칸 건너 배치: 중앙(CENTER)에서 양쪽으로 확장
        # positions = [CENTER - order + 2*j  for j in range(n_items)]
        for j, p in enumerate(items):
            col = 1 + (CENTER - order + 2 * j)   # +1 for label column
            ax = fig.add_subplot(gs[order, col])
            draw_single(ax, p[0], p[1], p[2], p[3], p[4], p[5])

        # 차수 라벨
        ax_label = fig.add_subplot(gs[order, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, f"{order}차",
                      transform=ax_label.transAxes, fontsize=9, fontweight='bold',
                      color='#555', ha='center', va='center')

    # 범례
    legend_handles = [
        mpatches.Patch(facecolor='#cccccc', edgecolor='#aaa', label='원래 격자 (이상적 Shot)'),
        mpatches.Patch(facecolor='#111111', edgecolor='#111', label='변형된 격자 (해당 K 적용)'),
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='#2e7d32', label='Scan (Wafer/Reticle Stage)'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#1565c0', label='Lens (Projection Lens)'),
        mpatches.Patch(facecolor='#fff3e0', edgecolor='#e65100', label='Scan Lens (동적 렌즈 조정)'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='#c62828', label='— (액추에이터 없음)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=9, frameon=True, fancybox=True, edgecolor='#ccc',
               bbox_to_anchor=(0.5, 0.005))

    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#f5f5f5')
    print(f"저장 완료: {save_path}")


# ===== 메인: dx / dy 각각 별도 창으로 출력 =====
SAVE_DIR = r"G:\내 드라이브\10_액츄에이터_RK시각화"

# --- dx 창 ---
fig_dx = plt.figure(figsize=(12, 8), facecolor='#f5f5f5')
draw_pyramid(
    fig_dx, dx_params,
    "ASML Scanner Overlay: dx 방향 RK Parameter (0~7차)\n"
    "dx = K1 + K3*x - K5*y + K7*x2 + K9*x*y + K11*y2 + ... + K71*y7\n"
    "(K1~K19: SPIE 2013 확인  |  K21~: 액추에이터 미확인)",
    f"{SAVE_DIR}\\RK_Parameter_Plot_dx.png"
)

# --- dy 창 ---
fig_dy = plt.figure(figsize=(12, 8), facecolor='#f5f5f5')
draw_pyramid(
    fig_dy, dy_params,
    "ASML Scanner Overlay: dy 방향 RK Parameter (0~7차)\n"
    "dy = K2 + K4*y + K6*x + K8*y2 + K10*y*x + K12*x2 + ... + K72*x7\n"
    "(K2~K20: SPIE 2013 확인  |  K22~: 액추에이터 미확인)",
    f"{SAVE_DIR}\\RK_Parameter_Plot_dy.png"
)

plt.show()
