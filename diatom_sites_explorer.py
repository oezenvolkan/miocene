# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25",
#     "pandas>=2.0",
#     "numpy",
#     "altair>=5.0",
# ]
# ///
"""Interactive explorer for Miocene diatom occurrences at Southern Ocean drill sites.

Run with:   marimo run diatom_sites_explorer.py      (app view)
       or:  marimo edit diatom_sites_explorer.py     (to see/edit the code)
With uv:    uvx marimo run --sandbox diatom_sites_explorer.py

Keep neptune_miocene_diatoms_southern_ocean.csv in the same folder as this file.
The basemap is loaded from the vega-datasets CDN, so the map needs an internet connection.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Miocene Southern Ocean diatom sites",
)


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Miocene diatoms at Southern Ocean sites

    Diatom occurrences from the Neptune database, Miocene only (5.333–23.03 Ma), at DSDP/ODP holes whose
    Miocene paleolatitude was about 46°S or further south. Pick a site to see its age range, sample counts
    and how much of its diatom record is pennate. The map can show each hole at its present-day position
    or at its median Miocene paleoposition (from Neptune's paleolatitude/longitude).
    """)
    return


@app.cell(hide_code=True)
def _():
    # --- Settings -------------------------------------------------------------
    DATA_FILE_NAME = "/Users/strabo/Documents/Bets/miocene-so-diatoms/neptune_miocene_diatoms_southern_ocean.csv"
    WORLD_URL = "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/world-110m.json"

    # Miocene subdivisions (ICS 2020), as (name, younger bound, older bound) in Ma
    SUBEPOCHS = [
        ("Late Miocene", 5.333, 11.63),
        ("Middle Miocene", 11.63, 15.98),
        ("Early Miocene", 15.98, 23.03),
    ]

    GROUP_CORE = "≥50°S"
    GROUP_BORDER = "46–50°S (borderline)"
    GROUP_COLORS = {GROUP_CORE: "#2a78d6", GROUP_BORDER: "#eb6834"}

    FORM_ORDER = ["Raphid pennate", "Araphid pennate", "Pennate (unassigned)", "Centric", "Other"]
    FORM_COLORS = ["#1baf7a", "#6250d6", "#e87ba4", "#898781", "#c3c2b7"]

    INK = "#0b0b0b"
    INK_2 = "#52514e"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    SURFACE = "#fcfcfb"
    return (
        DATA_FILE_NAME,
        FORM_COLORS,
        FORM_ORDER,
        GRID,
        GROUP_COLORS,
        GROUP_CORE,
        INK,
        INK_2,
        MUTED,
        SUBEPOCHS,
        SURFACE,
        WORLD_URL,
    )


@app.cell(hide_code=True)
def _(DATA_FILE_NAME, Path, mo, pd):
    _folder = mo.notebook_dir() or Path.cwd()
    _path = Path(_folder) / DATA_FILE_NAME
    mo.stop(
        not _path.exists(),
        mo.callout(
            mo.md(f"**Data file not found.** Put `{DATA_FILE_NAME}` in the same folder as this notebook."),
            kind="danger",
        ),
    )
    raw = pd.read_csv(_path)
    return (raw,)


@app.cell(hide_code=True)
def _(FORM_ORDER, np, pd, raw):
    # --- Derived tables: occurrences, samples, sites ---------------------------
    occ = raw.copy()
    _pen = occ["diatom_form"].eq("pennate")
    occ["form_label"] = np.select(
        [
            _pen & occ["pennate_group"].eq("raphid"),
            _pen & occ["pennate_group"].eq("araphid"),
            _pen,
            occ["diatom_form"].eq("centric"),
        ],
        FORM_ORDER[:4],
        default="Other",
    )
    occ["is_pennate"] = _pen

    # One sample = one hole + depth
    samples = (
        occ.groupby(["hole_id", "sample_depth_mbsf"])
        .agg(
            age=("sample_age_ma", "median"),
            core=("core", "first"),
            section=("section_number", "first"),
            n_occ=("is_pennate", "size"),
            n_pen=("is_pennate", "sum"),
        )
        .reset_index()
    )
    samples["pen_pct"] = 100 * samples["n_pen"] / samples["n_occ"]

    def _circular_mean(deg):
        deg = deg.dropna()
        if deg.empty:
            return np.nan
        rad = np.deg2rad(deg)
        return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))

    _g = occ.groupby("hole_id")
    sites = pd.DataFrame(
        {
            "site_group": _g["site_group"].first(),
            "region": _g["region"].first(),
            "present_lat": _g["present_lat"].first(),
            "present_lon": _g["present_lon"].first(),
            "paleo_lat": _g["paleo_latitude"].median(),
            "paleo_lat_min": _g["paleo_latitude"].min(),
            "paleo_lat_max": _g["paleo_latitude"].max(),
            "paleo_lon": _g["paleo_longitude"].agg(_circular_mean),
            "age_min": _g["sample_age_ma"].min(),
            "age_max": _g["sample_age_ma"].max(),
            "occurrences": _g.size(),
            "pennate_occ": _g["is_pennate"].sum(),
            "raphid_occ": _g["form_label"].agg(lambda s: (s == "Raphid pennate").sum()),
            "araphid_occ": _g["form_label"].agg(lambda s: (s == "Araphid pennate").sum()),
            "sources": _g["source_citation"].nunique(),
        }
    )
    sites["taxa"] = occ.drop_duplicates(["hole_id", "resolved_genus", "species"]).groupby("hole_id").size()
    sites["pennate_taxa"] = (
        occ[occ["is_pennate"]].drop_duplicates(["hole_id", "resolved_genus", "species"]).groupby("hole_id").size()
    )
    _sg = samples.groupby("hole_id")
    sites["samples"] = _sg.size()
    sites["samples_with_pennates"] = _sg["n_pen"].agg(lambda s: int((s > 0).sum()))
    sites = sites.fillna({"pennate_taxa": 0}).reset_index()
    sites["pct_pennate"] = 100 * sites["pennate_occ"] / sites["occurrences"]
    sites["pct_samples_pennate"] = 100 * sites["samples_with_pennates"] / sites["samples"]

    def _natural_key(hole):
        leg, site = hole.split("_")
        digits = "".join(ch for ch in site if ch.isdigit())
        return (int(leg), int(digits), site)

    sites["_order"] = sites["hole_id"].map(_natural_key)
    sites = sites.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return occ, samples, sites


@app.cell(hide_code=True)
def _(mo):
    position = mo.ui.radio(
        options={"Present-day": "present", "Miocene paleoposition": "paleo"},
        value="Present-day",
        inline=True,
        label="Map positions:",
    )
    include_border = mo.ui.checkbox(value=True, label="Include borderline sites (46–50°S)")
    return include_border, position


@app.cell(hide_code=True)
def _(GROUP_CORE, include_border, mo, sites):
    visible_sites = sites if include_border.value else sites[sites["site_group"] == GROUP_CORE]
    _options = {f"{r.hole_id} · {r.region}": r.hole_id for r in visible_sites.itertuples()}
    # Start on the ≥50°S site with the most samples
    _core = visible_sites[visible_sites["site_group"] == GROUP_CORE]
    _start = _core.loc[_core["samples"].idxmax(), "hole_id"]
    site_picker = mo.ui.dropdown(
        options=_options,
        value=next(k for k, v in _options.items() if v == _start),
        label="Site",
        searchable=True,
        full_width=True,
    )
    return site_picker, visible_sites


@app.cell(hide_code=True)
def _(include_border, mo, position, site_picker):
    mo.vstack([site_picker, mo.hstack([position, include_border], justify="start", align="center", gap=3)], gap=0.75)
    return


@app.cell(hide_code=True)
def _(occ, samples, site_picker, sites):
    sel_id = site_picker.value
    sel_site = sites.set_index("hole_id").loc[sel_id]
    sel_occ = occ[occ["hole_id"] == sel_id]
    sel_samples = samples[samples["hole_id"] == sel_id].sort_values("age")
    return sel_id, sel_occ, sel_samples, sel_site


@app.cell(hide_code=True)
def _(
    GROUP_COLORS,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    WORLD_URL,
    alt,
    np,
    pd,
    position,
    sel_id,
    visible_sites,
):
    # --- Map ---------------------------------------------------------------
    _lat_col, _lon_col = ("present_lat", "present_lon") if position.value == "present" else ("paleo_lat", "paleo_lon")
    _pts = visible_sites.assign(lat=visible_sites[_lat_col], lon=visible_sites[_lon_col]).dropna(subset=["lat", "lon"])
    _pts = _pts.assign(
        age_range=_pts.apply(lambda r: f"{r.age_min:.2f}–{r.age_max:.2f} Ma", axis=1),
        pen_label=_pts["pct_pennate"].map(lambda v: f"{v:.0f}%"),
        pos_label=_pts.apply(lambda r: f"{abs(r.lat):.1f}°S, {abs(r.lon):.1f}°{'E' if r.lon >= 0 else 'W'}", axis=1),
    )
    # Holes plotted at (nearly) the same spot: list them in the tooltip
    _key = list(zip(_pts["lat"].round(1), _pts["lon"].round(1)))
    _pts["same_spot"] = [
        ", ".join(h for h, k2 in zip(_pts["hole_id"], _key) if k2 == k and h != hole) or "—"
        for hole, k in zip(_pts["hole_id"], _key)
    ]

    _tooltip = [
        alt.Tooltip("hole_id:N", title="Hole"),
        alt.Tooltip("region:N", title="Region"),
        alt.Tooltip("site_group:N", title="Group"),
        alt.Tooltip("pos_label:N", title="Plotted at"),
        alt.Tooltip("age_range:N", title="Age range"),
        alt.Tooltip("samples:Q", title="Samples"),
        alt.Tooltip("pen_label:N", title="Pennate share"),
        alt.Tooltip("same_spot:N", title="Same location"),
    ]
    _enc = dict(longitude="lon:Q", latitude="lat:Q")

    _ocean = alt.Chart(alt.sphere()).mark_geoshape(fill="#eef3f8", stroke="#c3c2b7", strokeWidth=1)
    _graticule = alt.Chart(alt.graticule(step=[30, 10])).mark_geoshape(
        filled=False, stroke="#d5dbe2", strokeWidth=0.6
    )
    _land = alt.Chart(alt.topo_feature(WORLD_URL, "countries")).mark_geoshape(
        fill="#e2e0d8", stroke=SURFACE, strokeWidth=0.4
    )
    _ring = pd.DataFrame({"lon": np.arange(-180, 181, 2.0), "lat": -50.0})
    _ring_line = (
        alt.Chart(_ring)
        .mark_line(strokeDash=[5, 4], color=INK_2, strokeWidth=1.2)
        .encode(order="lon:Q", **_enc)
    )
    _ring_text = (
        alt.Chart(pd.DataFrame({"lon": [-135.0], "lat": [-50.0], "t": ["50°S"]}))
        .mark_text(color=INK_2, fontSize=11, dx=10, dy=-8, align="left")
        .encode(text="t:N", **_enc)
    )
    _points = (
        alt.Chart(_pts)
        .mark_circle(size=110, opacity=0.95, stroke=SURFACE, strokeWidth=1.5)
        .encode(
            color=alt.Color(
                "site_group:N",
                scale=alt.Scale(domain=list(GROUP_COLORS), range=list(GROUP_COLORS.values())),
                legend=alt.Legend(title="Site group", orient="bottom", direction="horizontal",
                                  values=[g for g in GROUP_COLORS if g in set(_pts["site_group"])]),
            ),
            tooltip=_tooltip,
            **_enc,
        )
    )
    _sel_pts = _pts[_pts["hole_id"] == sel_id]
    _sel_ring = (
        alt.Chart(_sel_pts)
        .mark_point(shape="circle", size=520, filled=False, stroke=INK, strokeWidth=2.5)
        .encode(tooltip=_tooltip, **_enc)
    )
    _sel_text = (
        alt.Chart(_sel_pts)
        .mark_text(fontSize=12, fontWeight="bold", color=INK, dy=-20)
        .encode(text="hole_id:N", **_enc)
    )
    if position.value == "present":
        _title = alt.Title("Present-day positions", subtitle="hover a site for details",
                           anchor="start", fontSize=13, color=INK, subtitleColor=MUTED)
    else:
        _title = alt.Title("Median Miocene paleopositions", subtitle="coastlines are present-day; hover a site for details",
                           anchor="start", fontSize=13, color=INK, subtitleColor=MUTED)
    site_map = (
        alt.layer(_ocean, _graticule, _land, _ring_line, _ring_text, _points, _sel_ring, _sel_text)
        .project(type="orthographic", rotate=[0, 90, 0], clipAngle=52, scale=310, translate=[250, 250])
        .properties(width=500, height=500, title=_title)
        .configure_view(stroke=None)
        .configure_legend(labelColor=INK_2, titleColor=INK_2)
    )
    return (site_map,)


@app.cell(hide_code=True)
def _(GROUP_CORE, mo, sel_id, sel_site):
    # --- Headline statistics for the selected site --------------------------
    def _lat(v):
        return f"{abs(v):.1f}°S"

    def _lon(v):
        return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"

    _s = sel_site
    _pen_total = max(_s.pennate_occ, 1)
    _header = mo.md(
        f"### {sel_id}\n\n"
        f"**{_s.region}** · {_s.site_group}<br>"
        f"Present-day position {_lat(_s.present_lat)}, {_lon(_s.present_lon)}<br>"
        f"Miocene paleolatitude {_lat(_s.paleo_lat_max)}–{_lat(_s.paleo_lat_min)} (median {_lat(_s.paleo_lat)})"
    )
    _cards = [
        mo.stat(f"{_s.age_min:.2f}–{_s.age_max:.2f}", label="Age range (Ma)", caption="youngest–oldest sample", bordered=True),
        mo.stat(f"{_s.samples:,}", label="Samples", caption=f"{_s.occurrences:,} occurrence records", bordered=True),
        mo.stat(f"{_s.pct_pennate:.0f}%", label="Pennate share", caption=f"{_s.pennate_occ:,} of {_s.occurrences:,} records", bordered=True),
        mo.stat(f"{_s.pct_samples_pennate:.0f}%", label="Samples with pennates", caption=f"{_s.samples_with_pennates} of {_s.samples}", bordered=True),
        mo.stat(f"{100 * _s.raphid_occ / _pen_total:.0f}% / {100 * _s.araphid_occ / _pen_total:.0f}%", label="Raphid / araphid", caption="share of pennate records", bordered=True),
        mo.stat(f"{_s.taxa}", label="Taxa", caption=f"{int(_s.pennate_taxa)} pennate", bordered=True),
    ]
    _warnings = []
    if _s.samples < 10:
        _warnings.append(f"Only {_s.samples} samples, so the percentages rest on very few samples.")
    if _s.taxa < 15:
        _warnings.append(
            f"Short taxon list ({_s.taxa} taxa): the source studies may have recorded only selected marker species."
        )
    if _s.site_group != GROUP_CORE:
        _warnings.append("Borderline site: median Miocene paleolatitude between 46°S and 50°S.")
    _note = mo.callout(mo.md("  \n".join(_warnings)), kind="warn") if _warnings else mo.md("")
    site_panel = mo.vstack(
        [_header, mo.hstack(_cards[:2], widths="equal"), mo.hstack(_cards[2:4], widths="equal"),
         mo.hstack(_cards[4:], widths="equal"), _note],
        gap=0.75,
    )
    return (site_panel,)


@app.cell(hide_code=True)
def _(mo, site_map, site_panel):
    mo.hstack([site_map, site_panel], widths=[1.05, 1], align="start", gap=1.5)
    return


@app.cell(hide_code=True)
def _(FORM_COLORS, FORM_ORDER, INK, INK_2, MUTED, SURFACE, alt, sel_occ):
    # --- Composition of the record: centric vs pennate subgroups -------------
    _comp = sel_occ.groupby("form_label").size().reindex(FORM_ORDER, fill_value=0).reset_index(name="n")
    _comp.columns = ["form", "n"]
    _comp["pct"] = 100 * _comp["n"] / _comp["n"].sum()
    _comp["end"] = _comp["pct"].cumsum()
    _comp["start"] = _comp["end"] - _comp["pct"]
    _comp["mid"] = (_comp["start"] + _comp["end"]) / 2
    _comp["label"] = _comp["pct"].map(lambda v: f"{v:.0f}%")
    _comp["text_color"] = ["#ffffff", "#ffffff", INK, "#ffffff", INK]
    _comp = _comp[_comp["n"] > 0]

    _bar = (
        alt.Chart(_comp)
        .mark_bar(height=34, stroke=SURFACE, strokeWidth=2)
        .encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=[0, 100]), title="% of occurrence records",
                    axis=alt.Axis(grid=False, tickCount=5, labelColor=MUTED, titleColor=INK_2, domainColor="#c3c2b7")),
            x2="end:Q",
            color=alt.Color("form:N", scale=alt.Scale(domain=FORM_ORDER, range=FORM_COLORS),
                            legend=alt.Legend(title=None, orient="bottom", columns=5, labelColor=INK_2)),
            tooltip=[alt.Tooltip("form:N", title="Group"), alt.Tooltip("n:Q", title="Records"),
                     alt.Tooltip("label:N", title="Share")],
        )
    )
    _text = (
        alt.Chart(_comp[_comp["pct"] >= 6])
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(x="mid:Q", text="label:N", color=alt.Color("text_color:N", scale=None))
    )
    comp_chart = (
        alt.layer(_bar, _text)
        .properties(width=620, height=60, title=alt.Title("Composition of the diatom record", anchor="start", color=INK, fontSize=13))
        .configure_view(stroke=None)
    )
    return (comp_chart,)


@app.cell(hide_code=True)
def _(FORM_COLORS, FORM_ORDER, GRID, INK, INK_2, MUTED, alt, mo, sel_occ):
    # --- Pennate genera at the selected site ---------------------------------
    _pen = sel_occ[sel_occ["is_pennate"]]
    if _pen.empty:
        genera_chart = mo.md("_No pennate taxa recorded at this site._")
    else:
        _gen = _pen.groupby(["resolved_genus", "form_label"]).size().reset_index(name="n")
        _gen["pct"] = 100 * _gen["n"] / _gen["n"].sum()
        _gen["label"] = _gen.apply(lambda r: f"{'<1' if r.pct < 1 else f'{r.pct:.0f}'}%  ({r.n})", axis=1)
        _order = _gen.sort_values("n", ascending=False)["resolved_genus"].tolist()
        _max = float(_gen["pct"].max())
        _dom = max(_max * 1.3, 10)
        _step = 5 if _dom <= 25 else (10 if _dom <= 60 else 20)
        _ticks = [v for v in range(0, 101, _step) if v <= _dom]
        _base = alt.Chart(_gen).encode(
            y=alt.Y("resolved_genus:N", sort=_order, title=None,
                    axis=alt.Axis(labelColor=INK_2, labelFontStyle="italic", ticks=False, domain=False)),
        )
        _bars = _base.mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("pct:Q", title="% of pennate records", scale=alt.Scale(domain=[0, _dom], nice=False),
                    axis=alt.Axis(gridColor=GRID, labelColor=MUTED, titleColor=INK_2, domain=False, values=_ticks)),
            color=alt.Color("form_label:N", scale=alt.Scale(domain=FORM_ORDER[:3], range=FORM_COLORS[:3]),
                            legend=alt.Legend(title=None, orient="bottom", labelColor=INK_2)),
            tooltip=[alt.Tooltip("resolved_genus:N", title="Genus"), alt.Tooltip("form_label:N", title="Group"),
                     alt.Tooltip("n:Q", title="Records"), alt.Tooltip("pct:Q", title="% of pennate records", format=".1f")],
        )
        _labels = _base.mark_text(align="left", dx=5, color=INK_2, fontSize=11).encode(x="pct:Q", text="label:N")
        genera_chart = (
            alt.layer(_bars, _labels)
            .properties(width=560, height=max(22 * len(_order), 60),
                        title=alt.Title("Pennate genera", subtitle="share of the site's pennate records (count in brackets)",
                                        anchor="start", color=INK, fontSize=13, subtitleColor=MUTED))
            .configure_view(stroke=None)
        )
    return (genera_chart,)


@app.cell(hide_code=True)
def _(GRID, INK, INK_2, MUTED, SUBEPOCHS, SURFACE, alt, pd, sel_samples):
    # --- Pennate share through time -----------------------------------------
    _bands = pd.DataFrame(SUBEPOCHS, columns=["name", "young", "old"])
    _bands["mid"] = (_bands["young"] + _bands["old"]) / 2
    _bands["shade"] = ["#f3f2ee", "#ffffff", "#f3f2ee"]
    _x = alt.X("age:Q", title="Age (Ma)", scale=alt.Scale(domain=[5.333, 23.03], nice=False),
               axis=alt.Axis(labelColor=MUTED, titleColor=INK_2, grid=False, domainColor="#c3c2b7", values=list(range(6, 24, 2))))
    _band_rect = (
        alt.Chart(_bands)
        .mark_rect()
        .encode(x=alt.X("young:Q", scale=alt.Scale(domain=[5.333, 23.03], nice=False)), x2="old:Q",
                color=alt.Color("shade:N", scale=None))
    )
    _band_text = (
        alt.Chart(_bands)
        .mark_text(baseline="top", dy=4, color=MUTED, fontSize=11)
        .encode(x="mid:Q", y=alt.value(0), text="name:N")
    )
    _ts = sel_samples.assign(depth=sel_samples["sample_depth_mbsf"])
    _y = alt.Y("pen_pct:Q", title="Pennate share of records in sample (%)", scale=alt.Scale(domain=[0, 105], nice=False),
               axis=alt.Axis(gridColor=GRID, labelColor=MUTED, titleColor=INK_2, domain=False, values=[0, 25, 50, 75, 100]))
    _line = alt.Chart(_ts).mark_line(color=INK_2, strokeWidth=2).encode(x=_x, y=_y)
    _dots = (
        alt.Chart(_ts)
        .mark_circle(size=70, color=INK_2, stroke=SURFACE, strokeWidth=1.5, opacity=1)
        .encode(
            x=_x, y=_y,
            tooltip=[alt.Tooltip("age:Q", title="Age (Ma)", format=".3f"),
                     alt.Tooltip("depth:Q", title="Depth (mbsf)", format=".2f"),
                     alt.Tooltip("n_occ:Q", title="Records in sample"),
                     alt.Tooltip("n_pen:Q", title="Pennate records"),
                     alt.Tooltip("pen_pct:Q", title="Pennate share (%)", format=".0f")],
        )
    )
    time_chart = (
        alt.layer(_band_rect, _band_text, _line, _dots)
        .properties(width=640, height=260,
                    title=alt.Title("Pennate share through the Miocene", subtitle="one point per sample; hover for details",
                                    anchor="start", color=INK, fontSize=13, subtitleColor=MUTED))
        .configure_view(stroke=None)
    )
    return (time_chart,)


@app.cell(hide_code=True)
def _(SUBEPOCHS, pd, sel_samples):
    # --- Statistics by Miocene subdivision ----------------------------------
    _rows = []
    for _name, _young, _old in SUBEPOCHS:
        _in = sel_samples[(sel_samples["age"] >= _young) & (sel_samples["age"] < _old if _old < 23.03 else sel_samples["age"] <= _old)]
        _n_occ = int(_in["n_occ"].sum())
        _rows.append(
            {
                "Interval": _name,
                "Ages (Ma)": f"{_young:g}–{_old:g}",
                "Samples": len(_in),
                "Records": _n_occ,
                "Pennate records (%)": round(100 * _in["n_pen"].sum() / _n_occ, 1) if _n_occ else None,
                "Samples with pennates (%)": round(100 * (_in["n_pen"] > 0).mean(), 1) if len(_in) else None,
            }
        )
    subepoch_table = pd.DataFrame(_rows)
    return (subepoch_table,)


@app.cell(hide_code=True)
def _(comp_chart, genera_chart, mo, sel_occ, subepoch_table, time_chart):
    _sources = (
        sel_occ.groupby("source_citation").size().sort_values(ascending=False).reset_index(name="records")
    )
    _records = sel_occ[
        ["sample_depth_mbsf", "sample_age_ma", "core", "section_number", "sect_interval_top",
         "resolved_genus", "species", "taxon_abundance", "diatom_form", "pennate_group", "source_citation"]
    ].sort_values(["sample_age_ma", "resolved_genus", "species"])
    mo.ui.tabs(
        {
            "Composition": mo.vstack([comp_chart, genera_chart], gap=1.5),
            "Through time": mo.vstack(
                [time_chart, mo.md("#### By Miocene subdivision"),
                 mo.ui.table(subepoch_table, selection=None, pagination=False, show_download=False, show_search=False)],
                gap=1,
            ),
            "Sources & records": mo.vstack(
                [mo.md("#### Source studies"),
                 mo.ui.table(_sources, selection=None, wrapped_columns=["source_citation"], show_search=False),
                 mo.md("#### All occurrence records for this site"),
                 mo.ui.table(_records, selection=None, page_size=15)],
                gap=1,
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All sites compared
    """)
    return


@app.cell(hide_code=True)
def _(GRID, GROUP_COLORS, INK, INK_2, MUTED, alt, sel_id, visible_sites):
    _d = visible_sites.assign(
        selected=visible_sites["hole_id"].eq(sel_id),
        label=visible_sites["pct_pennate"].map(lambda v: f"{v:.0f}%"),
    )
    _order = _d.sort_values("pct_pennate", ascending=False)["hole_id"].tolist()
    _base = alt.Chart(_d).encode(
        y=alt.Y("hole_id:N", sort=_order, title=None, axis=alt.Axis(labelColor=INK_2, ticks=False, domain=False)),
    )
    _bars = _base.mark_bar(cornerRadiusEnd=4, height=13).encode(
        x=alt.X("pct_pennate:Q", title="Pennate share of records (%)", scale=alt.Scale(domain=[0, 70], nice=False),
                axis=alt.Axis(gridColor=GRID, labelColor=MUTED, titleColor=INK_2, domain=False,
                              values=[0, 10, 20, 30, 40, 50, 60, 70])),
        color=alt.Color("site_group:N", scale=alt.Scale(domain=list(GROUP_COLORS), range=list(GROUP_COLORS.values())),
                        legend=alt.Legend(title="Site group", orient="bottom", labelColor=INK_2, titleColor=INK_2)),
        opacity=alt.condition("datum.selected", alt.value(1), alt.value(0.55)),
        stroke=alt.condition("datum.selected", alt.value(INK), alt.value(None)),
        strokeWidth=alt.condition("datum.selected", alt.value(2), alt.value(0)),
        tooltip=[alt.Tooltip("hole_id:N", title="Hole"), alt.Tooltip("region:N", title="Region"),
                 alt.Tooltip("samples:Q", title="Samples"), alt.Tooltip("label:N", title="Pennate share")],
    )
    _labels = _base.mark_text(align="left", dx=4, fontSize=10.5, color=INK_2).encode(x="pct_pennate:Q", text="label:N")
    all_sites_chart = (
        alt.layer(_bars, _labels)
        .properties(width=520, height=18 * len(_d),
                    title=alt.Title("Pennate share by site", subtitle="selected site outlined", anchor="start",
                                    color=INK, fontSize=13, subtitleColor=MUTED))
        .configure_view(stroke=None)
    )
    all_sites_chart
    return


@app.cell(hide_code=True)
def _(mo, visible_sites):
    _t = visible_sites[
        ["hole_id", "region", "site_group", "present_lat", "present_lon", "paleo_lat", "age_min", "age_max",
         "samples", "occurrences", "taxa", "pct_pennate", "pct_samples_pennate", "sources"]
    ].rename(
        columns={
            "hole_id": "Hole", "region": "Region", "site_group": "Group", "present_lat": "Lat (today)",
            "present_lon": "Lon (today)", "paleo_lat": "Paleolat (median)", "age_min": "Youngest (Ma)",
            "age_max": "Oldest (Ma)", "samples": "Samples", "occurrences": "Records", "taxa": "Taxa",
            "pct_pennate": "Pennate records (%)", "pct_samples_pennate": "Samples with pennates (%)",
            "sources": "Source studies",
        }
    )
    mo.ui.table(
        _t.round({"Lat (today)": 2, "Lon (today)": 2, "Paleolat (median)": 2, "Youngest (Ma)": 3, "Oldest (Ma)": 3,
                  "Pennate records (%)": 1, "Samples with pennates (%)": 1}),
        selection=None,
        pagination=False,
        freeze_columns_left=["Hole"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How the numbers are worked out": mo.md(r"""
    - **Records** are Neptune occurrences: one taxon recorded in one sample. Neptune stores presences only, so there are no recorded absences.
    - **Samples** are unique hole + depth combinations.
    - **Pennate share** = pennate records ÷ all records at the site (or in the sample). It is a share of *taxa recorded*, not of valves counted.
    - **Centric / pennate** is assigned by genus (`resolved_genus`). Raphid: *Denticulopsis, Fragilariopsis, Nitzschia, Crucidenticula, Alveus, Pleurosigma, Raphidodiscus*. Araphid: *Thalassionema, Thalassiothrix, Cavitatus, Rhaphoneis*. *Rouxia* is pennate with no raphid/araphid assignment. *Mediaria* (placement unconfirmed) and the non-diatoms *Actiniscus, Foliactiniscus* and *Carduifolia* are grouped as **Other**.
    - **Site groups** use the median Miocene paleolatitude: ≥50°S (≤ −49.5°) and 46–50°S borderline.
    - **Paleoposition** on the map = median Neptune paleolatitude and circular-mean paleolongitude of the site's Miocene samples.
    - Some source studies listed only selected marker species, so low counts can reflect what was recorded rather than what was in the sediment.
    """)
        }
    )
    return


if __name__ == "__main__":
    app.run()
