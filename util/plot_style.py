from __future__ import annotations
import matplotlib.pyplot as plt
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
OKABE_ITO_SEQUENCE = (
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
)

METHOD_COLORS = {
    "Baseline": OKABE_ITO["vermillion"],
    "EveNet": OKABE_ITO["blue"],
    "EveNet-Pretrain": OKABE_ITO["blue"],
    "EveNet-Scratch": OKABE_ITO["bluish_green"],
    "Pretrain": OKABE_ITO["blue"],
    "Scratch": OKABE_ITO["bluish_green"],
    "Truth": OKABE_ITO["orange"],
    "TruthNeutrino": OKABE_ITO["orange"],
    "EveNet-Truth": OKABE_ITO["orange"],
}
METHOD_COLOR_CYCLE = (
    OKABE_ITO["vermillion"],
    OKABE_ITO["blue"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
)

PROCESS_COLOR_CYCLE = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#EE7733",
    "#009988",
    "#BBBBBB",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#CC6677",
    "#332288",
)
PROCESS_COLORS = {
    "data94": OKABE_ITO["black"],
    "data": OKABE_ITO["black"],
    "Ztautau": OKABE_ITO["blue"],
    "Ztautau_pipi": plt.cm.tab20.colors[0],
    "Ztautau_pirho":  plt.cm.tab20.colors[1],
    "Ztautau_rhopi":  plt.cm.tab20.colors[2],
    "Ztautau_pie":  plt.cm.tab20.colors[3],
    "Ztautau_pimu":  plt.cm.tab20.colors[4],
    "Ztautau_rhoe":  plt.cm.tab20.colors[5],
    "Ztautau_rhomu":  plt.cm.tab20.colors[6],
    "Ztautau_rhorho":  plt.cm.tab20.colors[7],
    "Ztautau_ee":  plt.cm.tab20.colors[8],
    "Ztautau_mumu":  plt.cm.tab20.colors[9],
    "Ztautau_emu":  plt.cm.tab20.colors[10],
    "Ztautau_mue":  plt.cm.tab20.colors[11],
    "Ztautau_piother":  plt.cm.tab20.colors[12],
    "Ztautau_others": "#B8B8B8",
    "unselected": "#D0D0D0",
    "Zll": "#8C564B",
    "Zqq": "#7F7F7F",
}

PROCESS_LATEX_LABELS = {
    "data94": "Data",
    "data": "Data",
    "Ztautau": r"$Z\to\tau\tau$",
    "Ztautau_pipi": r"$\tau\tau\to\pi\pi$",
    "Ztautau_pirho": r"$\tau\tau\to\pi\rho$",
    "Ztautau_rhopi": r"$\tau\tau\to\rho\pi$",
    "Ztautau_pie": r"$\tau\tau\to\pi e$",
    "Ztautau_epi": r"$\tau\tau\to e\pi$",
    "Ztautau_pimu": r"$\tau\tau\to\pi\mu$",
    "Ztautau_mupi": r"$\tau\tau\to\mu\pi$",
    "Ztautau_rhoe": r"$\tau\tau\to\rho e$",
    "Ztautau_erho": r"$\tau\tau\to e\rho$",
    "Ztautau_rhomu": r"$\tau\tau\to\rho\mu$",
    "Ztautau_murho": r"$\tau\tau\to\mu\rho$",
    "Ztautau_rhorho": r"$\tau\tau\to\rho\rho$",
    "Ztautau_ee": r"$\tau\tau\to ee$",
    "Ztautau_mumu": r"$\tau\tau\to\mu\mu$",
    "Ztautau_emu": r"$\tau\tau\to e\mu$",
    "Ztautau_mue": r"$\tau\tau\to\mu e$",
    "Ztautau_piother": r"$\tau\tau\to\pi+\mathrm{other}$",
    "Ztautau_others": r"$Z\to\tau\tau$ other",
    "Zll": r"$Z\to\ell\ell$",
    "Zqq": r"$Z\to q\bar{q}$",
}
CHANNEL_LATEX_LABELS = {
    "pipi": PROCESS_LATEX_LABELS["Ztautau_pipi"],
    "pirho": PROCESS_LATEX_LABELS["Ztautau_pirho"],
    "rhopi": PROCESS_LATEX_LABELS["Ztautau_rhopi"],
    "rhorho": PROCESS_LATEX_LABELS["Ztautau_rhorho"],
    "pie": PROCESS_LATEX_LABELS["Ztautau_pie"],
    "epi": PROCESS_LATEX_LABELS["Ztautau_epi"],
    "pimu": PROCESS_LATEX_LABELS["Ztautau_pimu"],
    "mupi": PROCESS_LATEX_LABELS["Ztautau_mupi"],
    "rhoe": PROCESS_LATEX_LABELS["Ztautau_rhoe"],
    "erho": PROCESS_LATEX_LABELS["Ztautau_erho"],
    "rhomu": PROCESS_LATEX_LABELS["Ztautau_rhomu"],
    "murho": PROCESS_LATEX_LABELS["Ztautau_murho"],
    "ee": PROCESS_LATEX_LABELS["Ztautau_ee"],
    "mumu": PROCESS_LATEX_LABELS["Ztautau_mumu"],
    "emu": PROCESS_LATEX_LABELS["Ztautau_emu"],
    "mue": PROCESS_LATEX_LABELS["Ztautau_mue"],
    "hadhad": r"$\tau_{\mathrm{had}}\tau_{\mathrm{had}}$",
    "baseline": "baseline",
}


def method_color(method: str, method_index: int) -> str:
    return METHOD_COLORS.get(method, METHOD_COLOR_CYCLE[method_index % len(METHOD_COLOR_CYCLE)])


def process_color(process_name: str, process_index: int = 0) -> str:
    return PROCESS_COLORS.get(process_name, PROCESS_COLOR_CYCLE[process_index % len(PROCESS_COLOR_CYCLE)])


def process_latex_label(sample_name: str) -> str:
    return PROCESS_LATEX_LABELS.get(sample_name, sample_name.replace("_", r"\_"))


def channel_latex_label(name: str) -> str:
    channel = name.removeprefix("Ztautau_")
    if channel in CHANNEL_LATEX_LABELS:
        return CHANNEL_LATEX_LABELS[channel]
    if name in PROCESS_LATEX_LABELS:
        return PROCESS_LATEX_LABELS[name]
    return name.replace("_", r"\_")
