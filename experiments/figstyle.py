"""Shared figure style: Morandi palette (normal-vision separable; markers/hatches as secondary encoding), serif CJK + Times-metric Latin."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass

_avail = {f.name for f in font_manager.fontManager.ttflist}
LATIN = next((f for f in ("Liberation Serif", "Times New Roman") if f in _avail), "DejaVu Serif")  # Linux / Windows
CJK = next((f for f in ("Noto Serif CJK SC", "SimSun", "Songti SC") if f in _avail), "DejaVu Serif")

plt.rcParams.update({
    "font.family": [LATIN, CJK],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": LATIN, "mathtext.it": f"{LATIN}:italic",
    "mathtext.bf": f"{LATIN}:bold", "mathtext.cal": f"{LATIN}:italic", "mathtext.sf": CJK, "axes.unicode_minus": True, "savefig.dpi": 600, "hatch.linewidth": 0.5, "figure.dpi": 150,
})

# fixed categorical order (never cycled): method identity follows the method across all figures
C = {"GNN-IL+": "#4E7896", "GNN-IL+d": "#A3BCCF", "SA": "#C9A24E", "GA": "#9A7BB0", "SA-fast": "#6F9B73", "Sequential": "#C0705A", "GNN-IL": "#8C8C8C"}
MK = {"GNN-IL+": "o", "GNN-IL+d": "h", "SA": "s", "GA": "D", "SA-fast": "^", "Sequential": "v", "GNN-IL": "X"}
HATCH = {"GNN-IL+": "", "GNN-IL+d": "||||", "SA": "///", "GA": "\\\\\\", "SA-fast": "...", "Sequential": "xx", "GNN-IL": "--"}
INK, MUTED, GRID = "#333333", "#7A7A7A", "#E3E3E3"
CM = 1 / 2.54


def panel_label(ax, s):
    ax.text(0.5, -0.28, s, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def zh(s):
    """Mathtext has no font fallback: inside a string that contains math, route CJK runs through \\mathsf (= CJK serif)."""
    import re
    if '$' not in s:
        return s
    parts = s.split('$')
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'([\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+)', lambda m: '$\\mathsf{' + m.group(1) + '}$', parts[i])
    return '$'.join(parts)
