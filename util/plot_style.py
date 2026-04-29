from __future__ import annotations

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
    "Ztautau_pipi": "#4477AA",
    "Ztautau_pirho": "#EE6677",
    "Ztautau_rhopi": "#228833",
    "Ztautau_pie": "#CCBB44",
    "Ztautau_pimu": "#66CCEE",
    "Ztautau_rhoe": "#AA3377",
    "Ztautau_rhomu": "#EE7733",
    "Ztautau_rhorho": "#009988",
    "Ztautau_ee": "#882255",
    "Ztautau_mumu": "#44AA99",
    "Ztautau_emu": "#999933",
    "Ztautau_mue": "#AA4499",
    "Ztautau_piother": "#DDCC77",
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
    "Ztautau_pimu": r"$\tau\tau\to\pi\mu$",
    "Ztautau_rhoe": r"$\tau\tau\to\rho e$",
    "Ztautau_rhomu": r"$\tau\tau\to\rho\mu$",
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
    "pimu": PROCESS_LATEX_LABELS["Ztautau_pimu"],
    "rhoe": PROCESS_LATEX_LABELS["Ztautau_rhoe"],
    "rhomu": PROCESS_LATEX_LABELS["Ztautau_rhomu"],
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
