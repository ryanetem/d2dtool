import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import py3Dmol
import re
import os

st.set_page_config(page_title="D2D Tool", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; max-width: 1100px; }

    body, p, li, span, label,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"],
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600; font-size: 1.5rem !important;
    }
    h1 { text-align: center; margin-bottom: 0.6rem; }
    h2 { margin-top: 1.6rem; }
    p, li { font-size: 1.02rem; line-height: 1.65; }

    .stTabs [data-baseweb="tab-list"] { justify-content: center !important; }
    .stTabs [data-baseweb="tab"],
    .stTabs [aria-selected="true"],
    .stTabs button[role="tab"],
    .stTabs button[role="tab"] p { color: #000000 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #888888 !important; }
    .stTabs [data-baseweb="tab-border"] { background-color: transparent !important; }

    .stSelectbox label,
    .stSelectbox div,
    .stSelectbox span,
    .stSelectbox input,
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] *,
    [role="listbox"] *,
    [data-baseweb="popover"] * { color: #000000 !important; }

    .stMultiSelect [data-baseweb="tag"] {
        background-color: #E0E0E0 !important;
        color: #000000 !important;
    }
    .stMultiSelect [data-baseweb="tag"] span,
    .stMultiSelect [data-baseweb="tag"] [role="presentation"] {
        color: #333333 !important;
    }

    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_NOZOOM = {"scrollZoom": False, "dragmode": "pan", "displayModeBar": False}

RESIDUE_DISTANCES = {
    1: 20.84, 2: 21.37, 3: 17.5, 4: 18.95, 5: 13.2, 6: 16.26, 7: 18.85,
    8: 15.22, 9: 10.58, 10: 9.82, 11: 10.5, 12: 8.08, 13: 7.84, 14: 5.13,
    15: 7.68, 16: 7.43, 17: 10.78, 18: 12.11, 19: 6.07, 20: 10.83, 21: 14.44,
    22: 16.91, 23: 20.38, 24: 19.21, 25: 23.47, 26: 24.31, 27: 25.38,
    28: 24.54, 29: 17.18, 30: 19.81, 31: 18.51, 32: 14.65, 33: 10.93,
    34: 9.81, 35: 16.5, 36: 18.11, 37: 13.66, 38: 17.36, 39: 19.95,
    40: 19.06, 41: 22.05, 42: 18.74, 43: 15.32, 44: 15.64, 45: 15.06,
    46: 19.86, 47: 20.19, 48: 17.38, 49: 16.81, 50: 16.05, 51: 18.29,
    52: 17.92, 53: 15.21, 54: 13.57, 55: 17.65, 56: 15.88, 57: 18.75,
    58: 21.34, 59: 20.33, 60: 19.5, 61: 21.76, 62: 20.89, 63: 17.07,
    64: 18.23, 65: 20.38, 66: 18.38, 67: 14.67, 68: 18.35, 69: 19.7,
    70: 15.77, 71: 16.44, 72: 11.38, 73: 14.37, 74: 11.08, 75: 10.66,
    76: 4.45, 77: 10.53, 78: 10.14, 79: 12.57, 80: 14.6, 81: 12.6,
    82: 15.59, 83: 18.75, 84: 18.39, 85: 19.51, 86: 21.7, 87: 22.78,
    88: 23.69, 89: 24.18, 90: 24.07, 91: 24.71, 92: 24.3, 93: 26.09,
    94: 25.12, 95: 21.8, 96: 21.32, 97: 24.34, 98: 18.69, 99: 16.51,
    100: 22.69, 101: 23.31, 102: 17.27, 103: 18.62, 104: 22.73, 105: 21.35,
    106: 16.6, 107: 20.86, 108: 23.3, 109: 21.04, 110: 19.54, 111: 16.0,
    112: 13.45, 113: 13.39, 114: 8.23, 115: 11.61, 116: 7.13, 117: 9.65,
    118: 5.81, 119: 5.61, 120: 3.23, 121: 9.8, 122: 11.5, 123: 14.66,
    124: 17.11, 125: 19.25, 126: 15.87, 127: 14.32, 128: 18.91, 129: 19.87,
    130: 15.94, 131: 14.17, 132: 9.2, 133: 14.03, 134: 16.06, 135: 17.46,
    136: 18.39, 137: 14.13, 138: 14.52, 139: 17.81, 140: 17.3, 141: 11.42,
    142: 16.87, 143: 19.13, 144: 13.46, 145: 15.37, 146: 18.51, 147: 20.55,
    148: 16.08, 149: 16.23, 150: 21.66, 151: 22.91, 152: 19.77, 153: 20.98,
    154: 23.1, 155: 19.41, 156: 16.43, 157: 15.81, 158: 12.41, 159: 11.36,
    160: 7.01, 161: 7.3, 162: 3.76, 163: 1.33, 164: 0.0, 165: 1.34,
    166: 3.0, 167: 3.14, 168: 3.6, 169: 5.51, 170: 7.25, 171: 6.9,
    172: 8.57, 173: 10.4, 174: 11.89, 175: 11.69, 176: 13.42, 177: 11.69,
    178: 6.43, 179: 9.76, 180: 11.59, 181: 13.39, 182: 14.2, 183: 16.27,
    184: 14.96, 185: 12.34, 186: 15.68, 187: 12.58, 188: 10.14, 189: 12.22,
    190: 12.47, 191: 9.09, 192: 8.54, 193: 10.61, 194: 10.81, 195: 5.05,
    196: 7.78, 197: 11.04, 198: 10.96, 199: 9.61, 200: 12.65, 201: 14.63,
    202: 13.05, 203: 13.53, 204: 17.24, 205: 18.1, 206: 17.49, 207: 20.19,
    208: 22.08, 209: 21.25, 210: 23.91, 211: 19.58, 212: 20.64, 213: 18.09,
    214: 15.23, 215: 12.86, 216: 8.89, 217: 6.01, 218: 2.79, 219: 4.67,
    220: 3.34, 221: 5.1, 222: 6.52, 223: 7.61, 224: 10.68, 225: 11.54,
    226: 15.62, 227: 17.2, 228: 19.64, 229: 22.59, 230: 23.64, 231: 24.33,
    232: 22.56, 233: 19.48, 234: 20.14, 235: 20.85, 236: 17.89, 237: 16.02,
    238: 17.65, 239: 16.01, 240: 10.63, 241: 13.4, 242: 12.64, 243: 10.26,
    244: 13.09, 245: 11.95, 246: 8.97, 247: 9.06, 248: 11.84, 249: 11.96,
    250: 9.92, 251: 13.34, 252: 13.95, 253: 15.66, 254: 18.45, 255: 16.05,
    256: 18.41, 257: 21.53, 258: 22.9, 259: 26.45, 260: 27.76, 261: 27.17,
    262: 28.95, 263: 26.41, 264: 25.5, 265: 26.2, 266: 22.31, 267: 19.63,
    268: 18.6, 269: 14.52, 270: 14.45, 271: 12.69, 272: 9.08, 273: 9.55,
    274: 7.28, 275: 4.46, 276: 6.97, 277: 9.36, 278: 8.13, 279: 11.74,
    280: 14.41, 281: 12.45, 282: 12.91, 283: 16.45, 284: 12.66, 285: 9.5,
    286: 9.65, 287: 9.03, 288: 5.45, 289: 6.42, 290: 4.54, 291: 5.39,
    292: 3.46, 293: 5.65, 294: 4.05, 295: 0.0, 296: 4.41, 297: 6.07,
    298: 6.96, 299: 11.24, 300: 11.51, 301: 9.31, 302: 12.65, 303: 14.43,
    304: 13.09, 305: 16.71, 306: 19.6, 307: 19.97, 308: 22.68, 309: 25.41,
    310: 23.2, 311: 21.07, 312: 22.5, 313: 18.73, 314: 18.86, 315: 22.3,
    316: 21.91, 317: 19.86, 318: 22.03, 319: 25.01, 320: 24.43, 321: 21.16,
    322: 22.86, 323: 21.55, 324: 17.5, 325: 3.27, 326: 4.94, 327: 7.93,
    328: 10.66, 329: 6.18, 330: 8.59, 331: 12.27, 332: 11.81, 333: 8.81,
    334: 11.3, 335: 14.06, 336: 14.68, 337: 11.39, 338: 14.38, 339: 17.71,
    340: 16.6, 341: 13.13, 342: 16.66, 343: 18.6, 344: 16.95, 345: 14.47,
    346: 17.61, 347: 17.0, 348: 12.66, 349: 13.36, 350: 14.07, 351: 9.27,
    352: 5.07, 353: 0.0, 354: 5.05, 355: 5.94, 356: 5.55, 357: 9.59,
    358: 10.78, 359: 8.39, 360: 12.5, 361: 14.82, 362: 12.04, 363: 14.61,
    364: 18.05, 365: 16.41, 366: 12.35, 367: 14.78, 368: 17.79, 369: 14.31,
    370: 13.54, 371: 16.13, 372: 16.17, 373: 12.68, 374: 12.4, 375: 14.33,
    376: 10.59, 377: 7.13, 378: 8.45, 379: 7.91, 380: 4.68, 381: 6.91,
    382: 9.37, 383: 5.66, 384: 7.84, 385: 11.29, 386: 12.19, 387: 9.12,
    388: 11.78, 389: 14.93, 390: 12.85, 391: 10.55, 392: 13.17, 393: 14.2,
    394: 11.02, 395: 12.55, 396: 13.75, 397: 9.94, 398: 10.65, 399: 10.97,
    400: 6.55, 401: 6.69, 402: 6.99, 403: 4.32, 404: 7.14, 405: 9.31,
    406: 7.64, 407: 11.0, 408: 13.11, 409: 11.14, 410: 13.98, 411: 16.66,
    412: 14.62, 413: 11.23, 414: 9.0, 415: 3.27, 416: 6.25, 417: 8.84,
    418: 9.59, 419: 6.8, 420: 9.45, 421: 12.8, 422: 11.46, 423: 13.64,
    424: 16.97, 425: 17.64,
}

def structural_region(pos):
    if pd.isna(pos):
        return "NA"
    pos = int(pos)
    if pos in (164, 295, 353):
        return "Catalytic"
    d = RESIDUE_DISTANCES.get(pos)
    if d is None:
        return "NA"
    if d <= 5:
        return "Active Site Pocket"
    if d <= 10:
        return "Near Active Site"
    return "Distant"

def get_distance(pos):
    if pd.isna(pos):
        return np.nan
    return RESIDUE_DISTANCES.get(int(pos), np.nan)

REGION_ORDER = ["Catalytic", "Active Site Pocket", "Near Active Site", "Distant"]
REGION_COLORS = {
    "Active Site Pocket": "#FF0000",
    "Near Active Site": "#FFA500",
    "Distant": "#ADD8E6",
}
WT_COLOR = "#888888"

REGION_DESCRIPTIONS = {
    "Catalytic": (
        "Catalytic residues are amino acids that directly participate in the chemical "
        "reaction. These include E164 (acid/base catalyst, donates a proton to the leaving "
        "group), E353 (nucleophile, attacks anomeric carbon of the substrate), and Y295 "
        "(orients E353 via a hydrogen bond). Mutations here almost always destroy activity."
    ),
    "Active Site Pocket": (
        "Active site pocket residues are within 5 \u00c5 of the catalytic triad. "
        "They make direct contact with the substrate and determine the geometry, bonding, "
        "and contact of key structures in the protein. Mutations here don't necessarily "
        "destroy activity, but may have other chaotic (likely inhibiting) effects."
    ),
    "Near Active Site": (
        "Near active site residues are 5\u201310 \u00c5 from the catalytic triad. "
        "They don't directly touch the substrate, but they support the architecture of "
        "the active site. Mutations here have chaotic effects."
    ),
    "Distant": (
        "Distant residues are more than 10 \u00c5 from the catalytic triad. These might "
        "be on the protein surface or in parts of the protein simply too far from catalytic "
        "sites. Mutations here are more tolerable for activity."
    ),
}

AA_PROPS = {
    'A': {'group': 'Nonpolar', 'charge': 'Neutral'},
    'R': {'group': 'Basic',    'charge': 'Positive'},
    'N': {'group': 'Polar',    'charge': 'Neutral'},
    'D': {'group': 'Acidic',   'charge': 'Negative'},
    'C': {'group': 'Polar',    'charge': 'Neutral'},
    'E': {'group': 'Acidic',   'charge': 'Negative'},
    'Q': {'group': 'Polar',    'charge': 'Neutral'},
    'G': {'group': 'Nonpolar', 'charge': 'Neutral'},
    'H': {'group': 'Basic',    'charge': 'Positive'},
    'I': {'group': 'Nonpolar', 'charge': 'Neutral'},
    'L': {'group': 'Nonpolar', 'charge': 'Neutral'},
    'K': {'group': 'Basic',    'charge': 'Positive'},
    'M': {'group': 'Nonpolar', 'charge': 'Neutral'},
    'F': {'group': 'Aromatic', 'charge': 'Neutral'},
    'P': {'group': 'Nonpolar', 'charge': 'Neutral'},
    'S': {'group': 'Polar',    'charge': 'Neutral'},
    'T': {'group': 'Polar',    'charge': 'Neutral'},
    'W': {'group': 'Aromatic', 'charge': 'Neutral'},
    'Y': {'group': 'Aromatic', 'charge': 'Neutral'},
    'V': {'group': 'Nonpolar', 'charge': 'Neutral'},
}
AA_GROUPS = sorted(set(v['group'] for v in AA_PROPS.values()))
CHARGES = ["Negative", "Neutral", "Positive"]

AA_NAMES = {
    'A': 'Alanine',       'R': 'Arginine',       'N': 'Asparagine',     'D': 'Aspartic acid',
    'C': 'Cysteine',      'E': 'Glutamic acid',  'Q': 'Glutamine',      'G': 'Glycine',
    'H': 'Histidine',     'I': 'Isoleucine',     'L': 'Leucine',        'K': 'Lysine',
    'M': 'Methionine',    'F': 'Phenylalanine',  'P': 'Proline',        'S': 'Serine',
    'T': 'Threonine',     'W': 'Tryptophan',     'Y': 'Tyrosine',       'V': 'Valine',
}

METRICS = {
    "kcat/KM (1/(mM min))": "Catalytic Efficiency",
    "kcat (1/min)": "Turnover (kcat)",
    "Substrate Affinity (1/mM)": "Substrate Affinity (1/KM)",
    "T50 (degrees C)": "T50",
    "Tm (degrees C)": "Tm",
    "Yield (mg/mL)": "Yield",
}

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "BglB_characterization_data_curated.csv"))
    df = df.rename(columns={"Variant": "Mutant"})
    df["KM (mM)"] = pd.to_numeric(df["KM (mM)"], errors="coerce")
    df["Substrate Affinity (1/mM)"] = np.where(
        df["KM (mM)"] > 0, 1.0 / df["KM (mM)"], np.nan
    )
    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    def parse(v):
        v = str(v).strip()
        if v == "WT":
            return pd.Series({"orig_aa": None, "position": None, "new_aa": None, "is_wt": True})
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", v)
        if m:
            return pd.Series({"orig_aa": m.group(1), "position": int(m.group(2)),
                              "new_aa": m.group(3), "is_wt": False})
        return pd.Series({"orig_aa": None, "position": None, "new_aa": None, "is_wt": False})
    df = pd.concat([df, df["Mutant"].apply(parse)], axis=1)
    df["region"] = df["position"].apply(structural_region)
    df["distance_A"] = df["position"].apply(get_distance)
    df["orig_group"] = df["orig_aa"].map(lambda x: AA_PROPS.get(x, {}).get("group", ""))
    df["new_group"]  = df["new_aa"].map(lambda x: AA_PROPS.get(x, {}).get("group", ""))
    df["orig_charge"] = df["orig_aa"].map(lambda x: AA_PROPS.get(x, {}).get("charge", ""))
    df["new_charge"]  = df["new_aa"].map(lambda x: AA_PROPS.get(x, {}).get("charge", ""))
    return df

df = load_data()

@st.cache_data
def mutant_summary(_df):
    expressed = _df[_df["Expressed?"] == "yes"]
    agg_cols = {c: (c, "mean") for c in METRICS if c in expressed.columns}
    grp = expressed.groupby("Mutant").agg(n=("Mutant", "size"), **agg_cols).reset_index()
    meta = _df.drop_duplicates("Mutant")[["Mutant","orig_aa","new_aa","position","is_wt",
           "region","distance_A","orig_group","new_group","orig_charge","new_charge"]]
    grp = grp.merge(meta, on="Mutant", how="left")
    return grp

summary = mutant_summary(df)
mutants = summary[~summary["is_wt"]].copy()

BGLB_SEQ = dict(zip(
    df.dropna(subset=["position", "orig_aa"]).drop_duplicates("position")["position"].astype(int),
    df.dropna(subset=["position", "orig_aa"]).drop_duplicates("position")["orig_aa"],
))

wt_rows = df[(df["is_wt"]) & (df["Expressed?"] == "yes")]
WT_REF = {c: wt_rows[c].mean() for c in METRICS}

st.title("BglB Mutant Data", anchor=False)

st.markdown(
    "Mutations are classified based on their proximity to the active site "
    "([PDB 2JIE](https://www.rcsb.org/structure/2JIE)). Distances computed as "
    "minimum heavy-atom distance to the catalytic triad (E164, Y295, E353)."
)

for region in REGION_ORDER:
    st.markdown(REGION_DESCRIPTIONS[region])

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "BglB.pdb")):
    pdb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BglB.pdb")
    with open(pdb_path) as _f:
        _pdb_text = _f.read()

    _catalytic = [164, 295, 353]
    _pocket = sorted([p for p, d in RESIDUE_DISTANCES.items()
                      if 0 < d <= 5 and p not in _catalytic])
    _near = sorted([p for p, d in RESIDUE_DISTANCES.items() if 5 < d <= 10])

    _viewer = py3Dmol.view(width=800, height=620)
    _viewer.addModel(_pdb_text, "pdb")
    _viewer.setBackgroundColor("#FFFFFF")
    _viewer.setStyle({}, {"cartoon": {"color": "#808080"}})

    _viewer.addStyle({"resi": _near},
                     {"cartoon": {"color": "#FFA500"}})
    _viewer.addStyle({"resi": _near, "atom": "CA"},
                     {"sphere": {"color": "#FFA500", "radius": 1.0}})

    _viewer.addStyle({"resi": _pocket},
                     {"cartoon": {"color": "#FF0000"}})
    _viewer.addStyle({"resi": _pocket, "atom": "CA"},
                     {"sphere": {"color": "#FF0000", "radius": 1.0}})

    _viewer.addStyle({"resi": _catalytic},
                     {"cartoon": {"color": "#0000FF"},
                      "stick":   {"color": "#0000FF"}})

    _viewer.addStyle({"hetflag": True}, {"stick": {"color": "#00AA00"}})
    _viewer.setStyle({"resn": "HOH"}, {})

    _viewer.zoomTo()
    components.html(_viewer._make_html(), height=640)

    st.caption(
        "BglB structure. Drag, scroll, and click to move around. "
        "Catalytic triad in blue, active site pocket in red, "
        "near active site in orange, distant in gray, substrate in green."
    )

tabs = st.tabs(["Graph data", "Rankings"])

with tabs[0]:
    st.subheader("Distance calculator", anchor=False)
    st.markdown(
        "Enter a residue position to determine its distance from the active site. "
        "For example, 399 or W399N."
    )
    query = st.text_input(
        "Position lookup",
        label_visibility="collapsed",
        key="lookup_query",
    )
    if query:
        q = query.strip().upper()
        full_m = re.match(r'^([A-Z])(\d+)([A-Z])$', q)
        pos_m = re.match(r'^(\d+)$', q)
        if full_m:
            typed_orig, pos, typed_new = full_m.group(1), int(full_m.group(2)), full_m.group(3)
        elif pos_m:
            typed_orig, pos, typed_new = None, int(pos_m.group(1)), None
        else:
            typed_orig = typed_new = None
            pos = None
            st.markdown("*Invalid*")

        if pos is not None:
            if pos in RESIDUE_DISTANCES:
                d = RESIDUE_DISTANCES[pos]
                zone = structural_region(pos)
                actual_aa = BGLB_SEQ.get(pos)
                aa_full = AA_NAMES.get(actual_aa, actual_aa) if actual_aa else None

                lines = []
                if actual_aa:
                    lines.append(f"**Position {pos}** contains **{aa_full} ({actual_aa})**.")
                else:
                    lines.append(f"**Position {pos}**")
                if pos in (164, 295, 353):
                    lines.append("Distance to catalytic triad: **0 Å**")
                else:
                    lines.append(f"Distance to catalytic triad: **{d:.2f} Å**")
                lines.append(f"Zone: **{zone}**")
                st.markdown("  \n".join(lines))
            else:
                st.markdown("*Invalid*")

    st.subheader("Analyze mutant data", anchor=False)

    c1, c2 = st.columns(2)
    with c1:
        sort_by = st.radio("Sort by", ["Amino acid type", "Charge change"],
                            horizontal=True, key="sort_by")
    with c2:
        metric = st.selectbox("Enzyme parameters", list(METRICS.keys()),
                               format_func=lambda x: METRICS[x], key="metric")

    if sort_by == "Amino acid type":
        col_orig, col_new = "orig_group", "new_group"
        opts = sorted(AA_GROUPS)
        key_from, key_to = "from_aa", "to_aa"
    else:
        col_orig, col_new = "orig_charge", "new_charge"
        opts = CHARGES
        key_from, key_to = "from_charge", "to_charge"

    wt_val = WT_REF.get(metric)
    if wt_val is None or pd.isna(wt_val):
        wt_val = 0.0

    region_means = {r: np.nan for r in REGION_COLORS.keys()}
    region_counts = {r: 0 for r in REGION_COLORS.keys()}

    sel_left = st.session_state.get(key_from, [])
    sel_right = st.session_state.get(key_to, [])

    if sel_left and sel_right:
        d = mutants.dropna(subset=[metric, col_orig, col_new, "region"])
        d = d[d[col_orig].isin(sel_left) & d[col_new].isin(sel_right)]
        for r in REGION_COLORS.keys():
            sub = d[d["region"] == r]
            if len(sub) > 0:
                region_means[r] = sub[metric].mean()
                region_counts[r] = len(sub)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Wildtype"], y=[wt_val],
        name="Wildtype", marker_color=WT_COLOR, marker_line_width=0,
        hovertemplate="Wildtype<br>" + METRICS[metric] + ": %{y:.2f}<extra></extra>"))
    for region in ["Active Site Pocket", "Near Active Site", "Distant"]:
        y = region_means[region]
        n = region_counts[region]
        fig.add_trace(go.Bar(
            x=[region], y=[y if not pd.isna(y) else None],
            name=region, marker_color=REGION_COLORS[region], marker_line_width=0,
            hovertemplate=(f"{region}<br>" + METRICS[metric] +
                           f": %{{y:.2f}}<br>n={n}<extra></extra>") if n > 0 else
                          f"{region}<br>No data<extra></extra>"))

    bar_vals = [v for v in [wt_val] + list(region_means.values()) if not pd.isna(v)]
    y_max = max(bar_vals) * 1.15 if bar_vals else 1
    fig.update_layout(
        height=440,
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=14, color="#000000"),
        xaxis=dict(
            categoryorder="array",
            categoryarray=["Wildtype", "Active Site Pocket", "Near Active Site", "Distant"],
            tickfont=dict(size=13, color="#000000"),
            showgrid=False,
            showline=True,
            linecolor="#888888",
            linewidth=1,
            ticks="",
        ),
        yaxis=dict(
            title=dict(text=f"Mean {METRICS[metric]}", font=dict(size=13, color="#000000")),
            tickfont=dict(size=12, color="#000000"),
            range=[0, y_max],
            gridcolor="#eeeeee",
            zeroline=False,
            showline=False,
            ticks="",
        ),
        bargap=0.35,
    )
    chart_state = st.plotly_chart(
        fig, use_container_width=True, config=PLOTLY_NOZOOM,
        on_select="rerun", key="region_chart"
    )

    if metric == "Substrate Affinity (1/mM)":
        st.markdown("*A higher KM means lower substrate affinity.*")

    clicked_region = None
    try:
        pts = chart_state.selection.points
        if pts:
            trace_idx = pts[0].get("curve_number", 0)
            regions_in_order = ["Wildtype", "Active Site Pocket", "Near Active Site", "Distant"]
            if 0 <= trace_idx < len(regions_in_order):
                clicked_region = regions_in_order[trace_idx]
    except (AttributeError, IndexError, KeyError, TypeError):
        pass

    if clicked_region and clicked_region != "Wildtype" and sel_left and sel_right:
        mut_list = mutants.dropna(subset=[metric, col_orig, col_new, "region"])
        mut_list = mut_list[
            mut_list[col_orig].isin(sel_left) &
            mut_list[col_new].isin(sel_right) &
            (mut_list["region"] == clicked_region)
        ].sort_values(metric, ascending=False)

        if len(mut_list) > 0:
            st.markdown(
                f"**{clicked_region}** mutations contributing to this bar (n={len(mut_list)}):"
            )
            show_df = mut_list[["Mutant", "distance_A", metric]].copy()
            show_df.columns = ["Mutant", "Distance (Å)", METRICS[metric]]
            st.dataframe(
                show_df.reset_index(drop=True),
                use_container_width=True, hide_index=True,
                height=min(400, len(mut_list) * 36 + 50)
            )

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    sel_l, sel_arrow, sel_r = st.columns([5, 1, 5])
    with sel_l:
        st.multiselect("From", opts, key=key_from)
    with sel_arrow:
        st.markdown(
            "<div style='text-align:center; padding-top: 2.0rem; "
            "font-size: 1.6rem; color: #555;'>→</div>",
            unsafe_allow_html=True)
    with sel_r:
        st.multiselect("To", opts, key=key_to)

with tabs[1]:
    st.header("Mutant Rankings", anchor=False)
    var_search = st.text_input("Search mutant", value="", key="rank_search")
    rank_metric = st.selectbox("Rank by", list(METRICS.keys()),
                                format_func=lambda x: METRICS[x], key="rank_m")
    rank_df = summary.dropna(subset=[rank_metric]).copy()
    if var_search:
        rank_df = rank_df[rank_df["Mutant"].str.contains(var_search, case=False, na=False)]
    rank_df = rank_df.sort_values(rank_metric, ascending=False)
    wt_val_rank = WT_REF.get(rank_metric, None)

    display = rank_df[["Mutant", "region", "distance_A", "n", rank_metric]].copy()
    display.columns = ["Mutant", "Structural Region", "Distance (Å)", "Replicates", METRICS[rank_metric]]
    display = display.reset_index(drop=True)
    display.index = display.index + 1
    display.index.name = "Rank"

    def color_cell(val):
        if pd.isna(val) or wt_val_rank is None or wt_val_rank == 0:
            return ""
        ratio = val / wt_val_rank
        if ratio >= 1:
            return f"background-color: rgba(255,165,0,{min((ratio-1)/2,1)*0.55})"
        return f"background-color: rgba(173,216,230,{min(1-ratio,1)*0.85})"

    styled = display.style.map(color_cell, subset=[METRICS[rank_metric]])
    styled = styled.format({METRICS[rank_metric]: "{:.2f}", "Distance (Å)": "{:.1f}"})
    table_height = max(700, len(rank_df)*35+50)
    st.dataframe(styled, use_container_width=True, height=min(table_height, 2000))

st.markdown(
    "<div style='text-align:center; color:#000; font-size:0.85rem; margin-top: 30px;'>"
    "Data from <a href='https://d2d.ucdavis.edu' style='color:#0000EE;'>d2d.ucdavis.edu</a>"
    " &amp; Structure from <a href='https://www.rcsb.org/structure/2JIE' style='color:#0000EE;'>"
    "PDB 2JIE</a> (Isorna et al. 2007)</div>",
    unsafe_allow_html=True)
