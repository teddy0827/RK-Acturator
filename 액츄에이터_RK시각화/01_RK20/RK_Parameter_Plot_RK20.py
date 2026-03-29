"""
ASML Scanner Overlay 보정 모델: RK Parameter Grid Deformation 시각화
출처: "High Order Field-to-Field Corrections for Imaging and Overlay"
      (Mulkens et al., ASML, SPIE Proc. 8683, 2013)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 20개 K Parameter 정의 =====
params = [
    # (K이름, 물리적의미, 수식표현, dx함수, dy함수, 액추에이터)
    ("K1",  "Translation X",            "dx = K1",            lambda x,y: (np.ones_like(x), np.zeros_like(y)),  "Scan"),
    ("K2",  "Translation Y",            "dy = K2",            lambda x,y: (np.zeros_like(x), np.ones_like(y)),  "Scan"),
    ("K3",  "Magnification X (Mx)",     "dx = K3·x",          lambda x,y: (x, np.zeros_like(y)),               "Lens"),
    ("K4",  "Magnification Y (My)",     "dy = K4·y",          lambda x,y: (np.zeros_like(x), y),               "Scan"),
    ("K5",  "Rotation X (Rx)",          "dx = −K5·y",         lambda x,y: (-y, np.zeros_like(y)),              "Scan"),
    ("K6",  "Rotation Y (Ry)",          "dy = K6·x",          lambda x,y: (np.zeros_like(x), x),               "Scan"),
    ("K7",  "2nd Order Mag X (D2x)",    "dx = K7·x²",         lambda x,y: (x**2, np.zeros_like(y)),            "Lens"),
    ("K8",  "2nd Order Mag Y",          "dy = K8·y²",         lambda x,y: (np.zeros_like(x), y**2),            "Scan"),
    ("K9",  "Trapezoid X",              "dx = K9·x·y",        lambda x,y: (x*y, np.zeros_like(y)),             "Scan Lens"),
    ("K10", "Trapezoid Y",              "dy = K10·y·x",       lambda x,y: (np.zeros_like(x), y*x),             "Scan"),
    ("K11", "Bow X",                    "dx = K11·y²",        lambda x,y: (y**2, np.zeros_like(y)),             "Scan"),
    ("K12", "Bow Y (D2y)",              "dy = K12·x²",        lambda x,y: (np.zeros_like(x), x**2),            "Lens"),
    ("K13", "3rd Order Mag X (D3x)",    "dx = K13·x³",        lambda x,y: (x**3, np.zeros_like(y)),            "Lens"),
    ("K14", "3rd Order Mag Y",          "dy = K14·y³",        lambda x,y: (np.zeros_like(x), y**3),            "Scan"),
    ("K15", "Accordion X",              "dx = K15·x²·y",      lambda x,y: (x**2*y, np.zeros_like(y)),          "Scan Lens"),
    ("K16", "Accordion Y",              "dy = K16·y²·x",      lambda x,y: (np.zeros_like(x), y**2*x),          "Scan"),
    ("K17", "C-shape Distortion X",     "dx = K17·x·y²",      lambda x,y: (x*y**2, np.zeros_like(y)),          "Scan Lens"),
    ("K18", "C-shape Distortion Y",     "dy = K18·y·x²",      lambda x,y: (np.zeros_like(x), y*x**2),          "Scan Lens"),
    ("K19", "3rd Order Flow X",         "dx = K19·y³",        lambda x,y: (y**3, np.zeros_like(y)),             "Scan"),
    ("K20", "3rd Order Flow Y (D3y)",   "dy = K20·x³",        lambda x,y: (np.zeros_like(x), x**3),            "— (None)"),
]

# ===== 설정 =====
N_GRID = 7          # 격자 선 수
N_FINE = 50         # 곡선 보간 포인트 수
SCALE  = 0.18       # 변형 과장 계수
GRID_RANGE = 1.0    # 격자 범위 [-1, 1]
PLOT_MARGIN = 0.35  # 플롯 여백

# 격자 좌표
grid_vals = np.linspace(-GRID_RANGE, GRID_RANGE, N_GRID)
fine_vals = np.linspace(-GRID_RANGE, GRID_RANGE, N_FINE)


def draw_single(ax, k_name, phys_name, formula, func, actuator):
    """하나의 K 파라미터에 대한 격자 변형을 그린다."""
    ax.set_xlim(-GRID_RANGE - PLOT_MARGIN, GRID_RANGE + PLOT_MARGIN)
    ax.set_ylim(-GRID_RANGE - PLOT_MARGIN, GRID_RANGE + PLOT_MARGIN)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # --- 원래 격자 (회색) ---
    for v in grid_vals:
        ax.plot([-GRID_RANGE, GRID_RANGE], [v, v], color='#cccccc', linewidth=0.8)
        ax.plot([v, v], [-GRID_RANGE, GRID_RANGE], color='#cccccc', linewidth=0.8)
    # 원래 격자점
    gx_orig, gy_orig = np.meshgrid(grid_vals, grid_vals)
    ax.plot(gx_orig, gy_orig, 'o', color='#bbbbbb', markersize=2.5, zorder=2)

    # --- 변형된 격자 (검정) ---
    # 가로선 (constant y)
    for v in grid_vals:
        xs = fine_vals.copy()
        ys = np.full_like(xs, v)
        ddx, ddy = func(xs, ys)
        ax.plot(xs + ddx * SCALE, ys + ddy * SCALE, color='#111111', linewidth=1.3, zorder=3)

    # 세로선 (constant x)
    for v in grid_vals:
        ys = fine_vals.copy()
        xs = np.full_like(ys, v)
        ddx, ddy = func(xs, ys)
        ax.plot(xs + ddx * SCALE, ys + ddy * SCALE, color='#111111', linewidth=1.3, zorder=3)

    # 변형된 격자점
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    ddx, ddy = func(gx, gy)
    ax.plot(gx + ddx * SCALE, gy + ddy * SCALE, 'o', color='#111111', markersize=3, zorder=4)

    # --- 제목 ---
    ax.set_title(f"{k_name}  —  {phys_name}\n{formula}",
                 fontsize=9, fontweight='bold', pad=6)

    # --- 액추에이터 배지 ---
    act_colors = {
        'Scan':      ('#2e7d32', '#e8f5e9'),
        'Lens':      ('#1565c0', '#e3f2fd'),
        'Scan Lens': ('#e65100', '#fff3e0'),
        '— (None)':  ('#c62828', '#ffebee'),
    }
    txt_color, bg_color = act_colors.get(actuator, ('#333', '#eee'))
    ax.text(0, -GRID_RANGE - 0.22, actuator, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=txt_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color,
                      edgecolor=txt_color, linewidth=0.8))

    # 좌표축 라벨
    ax.text(GRID_RANGE + 0.28, -GRID_RANGE - 0.32, 'x(Slit)', fontsize=6, color='#999', ha='right')
    ax.text(-GRID_RANGE - 0.28, GRID_RANGE + 0.28, 'y(Scan)', fontsize=6, color='#999', ha='left')


# ===== 메인 플롯: 4x5 그리드 =====
fig, axes = plt.subplots(4, 5, figsize=(20, 16), facecolor='#f5f5f5')
fig.suptitle("ASML Scanner Overlay 보정 모델: RK Parameter Grid Deformation\n"
             "(ASML SPIE Proc. 8683, 2013 — Table 1 기준)",
             fontsize=16, fontweight='bold', y=0.98)

for idx, (k_name, phys_name, formula, func, actuator) in enumerate(params):
    row, col = divmod(idx, 5)
    ax = axes[row][col]
    draw_single(ax, k_name, phys_name, formula, func, actuator)

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

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(r"C:\Users\oth08\OneDrive\바탕 화면\메모\액츄에이터\RK_Parameter_Plot.png",
            dpi=200, bbox_inches='tight', facecolor='#f5f5f5')
plt.show()
print("저장 완료: RK_Parameter_Plot.png")
