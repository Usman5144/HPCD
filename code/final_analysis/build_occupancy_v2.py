import os
#!/usr/bin/env python3

from pathlib import Path
from decimal import Decimal
import csv
import hashlib

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(os.environ.get("HPCD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

# Correct frozen source for token-pooled OVERALL occupancy.
SRC = ROOT / "results/final_analysis_v1/occupancy/occupancy_by_dataset_v1.csv"

OUT_CSV = ROOT / "results/final_analysis_v1/figure_data/figure_occupancy_v2.csv"
OUT_PDF = ROOT / "results/final_analysis_v1/figures/figure_occupancy_v2.pdf"
OUT_SVG = ROOT / "results/final_analysis_v1/figures/figure_occupancy_v2.svg"
OUT_PNG = ROOT / "results/final_analysis_v1/figures/figure_occupancy_v2.png"

EXPECTED_SRC_SHA = (
    "1367d1e14e57d5acac045f4e705af5c5"
    "914983eb4b79e53611748a18f558adcf"
)

METHODS = [
    "Human",
    "Original",
    "Vanilla",
    "CoPA",
    "HPCD",
]

METHOD_MAP = {
    "Human": "Human",
    "Original": "Original",
    "Vanilla": "Vanilla",
    "CoPA-3B": "CoPA",
    "CoPA": "CoPA",
    "HPCD": "HPCD",
}

ROUNDED_REFERENCE = {
    "Human": [
        0.1677, 0.1673, 0.1643,
        0.1714, 0.1689, 0.1604,
    ],
    "Original": [
        0.1374, 0.1648, 0.1636,
        0.1633, 0.1610, 0.2099,
    ],
    "Vanilla": [
        0.1463, 0.1642, 0.1649,
        0.1694, 0.1699, 0.1854,
    ],
    "CoPA": [
        0.1588, 0.1784, 0.1803,
        0.1798, 0.1718, 0.1308,
    ],
    "HPCD": [
        0.1461, 0.1650, 0.1697,
        0.1683, 0.1666, 0.1843,
    ],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 1. VERIFY FROZEN SOURCE
# ============================================================

if not SRC.exists():
    raise FileNotFoundError(f"Missing frozen source:\n{SRC}")

actual_src_sha = sha256(SRC)

print("\n" + "=" * 88)
print("FROZEN OCCUPANCY SOURCE")
print("=" * 88)
print(SRC)
print("SHA256:", actual_src_sha)

if actual_src_sha != EXPECTED_SRC_SHA:
    raise RuntimeError(
        "\nSOURCE SHA256 MISMATCH\n"
        f"Expected: {EXPECTED_SRC_SHA}\n"
        f"Actual:   {actual_src_sha}\n"
        "\nSTOPPING."
    )


# ============================================================
# 2. LOAD RAW HIGH-PRECISION OVERALL VALUES
# ============================================================

with open(SRC, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

data = {method: {} for method in METHODS}

for row in rows:
    dataset = row["dataset"].strip()
    raw_method = row["method"].strip()

    if dataset.lower() != "overall":
        continue

    if raw_method == "Temperature Control":
        continue

    if raw_method not in METHOD_MAP:
        continue

    method = METHOD_MAP[raw_method]

    if method not in METHODS:
        continue

    state = int(row["state"])
    probability_text = row["probability"].strip()

    if state in data[method]:
        raise RuntimeError(
            f"Duplicate row: method={method}, state={state}"
        )

    data[method][state] = {
        "text": probability_text,
        "value": Decimal(probability_text),
    }


# ============================================================
# 3. STRUCTURAL VALIDATION
# ============================================================

for method in METHODS:
    states = sorted(data[method])

    if states != [0, 1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"{method}: expected states 0..5, got {states}"
        )


# ============================================================
# 4. VALIDATE AGAINST FROZEN ROUNDED REFERENCES
# ============================================================

print("\n" + "=" * 88)
print("ROUNDED OVERALL OCCUPANCY VALIDATION")
print("=" * 88)

for method in METHODS:
    got = [
        round(float(data[method][state]["value"]), 4)
        for state in range(6)
    ]

    print(f"{method:<10} {got}")

    if got != ROUNDED_REFERENCE[method]:
        raise RuntimeError(
            f"\n{method}: rounded frozen values mismatch.\n"
            f"Got:      {got}\n"
            f"Expected: {ROUNDED_REFERENCE[method]}\n"
            "\nSTOPPING."
        )


# ============================================================
# 5. EXACT SUMS
# ============================================================

sums = {}

for method in METHODS:
    total = sum(
        (data[method][state]["value"] for state in range(6)),
        Decimal("0"),
    )
    sums[method] = total

    if abs(total - Decimal("1")) > Decimal("1e-9"):
        raise RuntimeError(
            f"{method}: occupancy sum not ~1: {total}"
        )


# ============================================================
# 6. WRITE EXACT 30-ROW V2 SOURCE CSV
# ============================================================

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

output_rows = []

for method in METHODS:
    for state in range(6):
        output_rows.append({
            "method": method,
            "state": f"S{state + 1}",
            "occupancy": data[method][state]["text"],
        })

if len(output_rows) != 30:
    raise RuntimeError(
        f"Expected exactly 30 rows, got {len(output_rows)}"
    )

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["method", "state", "occupancy"],
    )
    writer.writeheader()
    writer.writerows(output_rows)


# ============================================================
# 7. VERIFY V2 VALUES == FROZEN SOURCE EXACTLY
# ============================================================

with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
    check_rows = list(csv.DictReader(f))

max_abs_diff = Decimal("0")

for row in check_rows:
    method = row["method"]
    state = int(row["state"][1:]) - 1

    src_value = data[method][state]["value"]
    v2_value = Decimal(row["occupancy"])

    diff = abs(src_value - v2_value)

    if diff > max_abs_diff:
        max_abs_diff = diff

if max_abs_diff != Decimal("0"):
    raise RuntimeError(
        "\nV2 DATA DO NOT EXACTLY MATCH FROZEN SOURCE.\n"
        f"Maximum absolute difference: {max_abs_diff}\n"
        "\nSTOPPING BEFORE PLOTTING."
    )


# ============================================================
# 8. PUBLICATION FIGURE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(3.30, 2.55))

x = [1, 2, 3, 4, 5, 6]

styles = {
    "Human": {
        "color": "#000000",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 1.9,
        "markersize": 4.2,
        "zorder": 5,
    },
    "Original": {
        "color": "#777777",
        "linestyle": "-.",
        "marker": "x",
        "linewidth": 1.5,
        "markersize": 4.3,
        "zorder": 2,
    },
    "Vanilla": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 1.55,
        "markersize": 4.0,
        "zorder": 3,
    },
    "CoPA": {
        "color": "#D55E00",
        "linestyle": ":",
        "marker": "^",
        "linewidth": 1.6,
        "markersize": 4.3,
        "zorder": 3,
    },
    "HPCD": {
        "color": "#009E73",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 1.8,
        "markersize": 4.1,
        "zorder": 4,
    },
}

for method in METHODS:
    y = [
        float(data[method][state]["value"])
        for state in range(6)
    ]

    ax.plot(
        x,
        y,
        label=method,
        **styles[method],
    )

ax.set_xlim(0.75, 6.25)
ax.set_ylim(0.00, 0.23)

ax.set_xticks(
    x,
    ["S1", "S2", "S3", "S4", "S5", "S6"],
)

ax.set_yticks([
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
])

ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.2f")
)

ax.set_xlabel("Token-choice state")
ax.set_ylabel("Occupancy probability")

ax.grid(
    axis="y",
    linewidth=0.45,
    alpha=0.28,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

ax.tick_params(
    width=0.6,
    length=2.8,
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=3,
    frameon=False,
    handlelength=2.2,
    columnspacing=0.9,
    handletextpad=0.45,
    borderaxespad=0.0,
)

fig.subplots_adjust(
    left=0.19,
    right=0.985,
    bottom=0.19,
    top=0.80,
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.025,
)

fig.savefig(
    OUT_SVG,
    bbox_inches="tight",
    pad_inches=0.025,
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.025,
)

plt.close(fig)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 88)
print("OCCUPANCY V2 COMPLETE")
print("=" * 88)

print("\nPDF:")
print(OUT_PDF.relative_to(ROOT))

print("\nSVG:")
print(OUT_SVG.relative_to(ROOT))

print("\nPNG:")
print(OUT_PNG.relative_to(ROOT))

print("\nSOURCE CSV:")
print(OUT_CSV.relative_to(ROOT))

print("\n" + "=" * 88)
print("SHA256")
print("=" * 88)

for path in [
    OUT_CSV,
    OUT_PDF,
    OUT_SVG,
    OUT_PNG,
]:
    print(
        sha256(path),
        path.relative_to(ROOT),
    )

print("\n" + "=" * 88)
print("EXACT OCCUPANCY SUMS")
print("=" * 88)

for method in METHODS:
    print(
        f"{method:<10} {format(sums[method], 'f')}"
    )

print("\n" + "=" * 88)
print("MAXIMUM DIFFERENCE FROM FROZEN SOURCE")
print("=" * 88)
print(format(max_abs_diff, "f"))

print("\nOCCUPANCY_V2=PASS")

