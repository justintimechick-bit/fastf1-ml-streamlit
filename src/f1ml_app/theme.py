"""Pit-wall design tokens and constructor accents; no network or artifact mutation."""

import json
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st

from f1ml_app.settings import project_root

LIGHT = {
    "surface": "#f4f6f8",
    "card": "#ffffff",
    "raised": "#e9eef3",
    "ink": "#101820",
    "ink2": "#394c60",
    "muted": "#526579",
    "grid": "#cbd3dc",
    "neutral": "#687b8e",
    "on_neutral": "#ffffff",
    "accent": "#bd2035",
    "on_accent": "#ffffff",
    "locked": "#006c91",
    "verified": "#157445",
    "warning": "#8b5700",
    "series": ("#007ba6", "#b84e00", "#157445"),
}
DARK = {
    "surface": "#080d12",
    "card": "#121b25",
    "raised": "#1b2937",
    "ink": "#f4f7fa",
    "ink2": "#c3cfdb",
    "muted": "#98aabd",
    "grid": "#304152",
    "neutral": "#8094a8",
    "on_neutral": "#080d12",
    "accent": "#ff5263",
    "on_accent": "#080d12",
    "locked": "#65d9f5",
    "verified": "#67e8a5",
    "warning": "#ffd071",
    "series": ("#65d9f5", "#ffac63", "#67e8a5"),
}

# Presentation accents keyed by configs/team_mapping.json's canonical team_id.
# Keep colors out of the identity document: its digest seals ingestion history.
TEAM_COLORS = {
    "red_bull": "#4781FF",
    "ferrari": "#FF4655",
    "mclaren": "#FF9B31",
    "mercedes": "#3DDFC5",
    "aston_martin": "#45C69C",
    "alpine": "#F49BD4",
    "williams": "#67C5FF",
    "haas": "#F0F3F6",
    "racing_bulls": "#A5A0FF",
    "sauber": "#88E64B",
    "audi": "#FF795B",
    "cadillac": "#D8BE87",
}
TEAM_FALLBACK = "#98AABD"
TEAM_NAMES = {
    "red_bull": "Red Bull Racing",
    "ferrari": "Ferrari",
    "mclaren": "McLaren",
    "mercedes": "Mercedes",
    "aston_martin": "Aston Martin",
    "alpine": "Alpine",
    "williams": "Williams",
    "haas": "Haas",
    "racing_bulls": "Racing Bulls",
    "sauber": "Sauber",
    "audi": "Audi",
    "cadillac": "Cadillac",
}


def _team_key(value: str) -> str:
    return re.sub(r"[\s_-]+", "_", value.strip().casefold())


@lru_cache(maxsize=8)
def _team_aliases(root: str) -> dict[str, str]:
    aliases = {_team_key(team): team for team in TEAM_COLORS}
    try:
        document = json.loads((Path(root) / "configs/team_mapping.json").read_text())
        for rule in document["rules"]:
            team = rule["team_id"]
            for name in [team, rule["source_id"], *rule["source_names"]]:
                aliases[_team_key(name)] = team
    except (OSError, ValueError, KeyError, TypeError):
        pass  # Installed wheels may not have the project configs alongside them.
    return aliases


def team_color(team: str | None) -> str:
    """Resolve a canonical ID or configured alias to a decorative constructor accent."""
    key = _team_key(str(team or ""))
    canonical = _team_aliases(str(project_root())).get(key, key)
    return TEAM_COLORS.get(canonical, TEAM_FALLBACK)


def team_name(team: str | None) -> str:
    key = _team_key(str(team or ""))
    canonical = _team_aliases(str(project_root())).get(key, key)
    return TEAM_NAMES.get(
        canonical, str(team).replace("_", " ").title() if team else "Unknown team"
    )


def constructor_ids() -> list[str]:
    """Canonical constructor identities from the reviewed mapping, including older teams."""
    return sorted(set(_team_aliases(str(project_root())).values()), key=team_name)


def tokens() -> dict:
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - no browser context in scripts/tests
        kind = None
    return LIGHT if kind == "light" else DARK


CSS = """
<style>
:root{
  --f1-carbon:#080d12;--f1-surface:#080d12;--f1-card:#121b25;--f1-raised:#1b2937;
  --f1-ink:#f4f7fa;--f1-ink2:#c3cfdb;--f1-muted:#98aabd;--f1-border:#304152;
  --f1-accent:#ff5263;--f1-on-accent:#080d12;
  --f1-locked:#65d9f5;--f1-verified:#67e8a5;--f1-warning:#ffd071;
  --f1-numeric:"SFMono-Regular",Consolas,"Liberation Mono",monospace;
}
/* Stable Streamlit hooks only; no generated emotion class names. */
[data-testid="stAppViewContainer"]{font-variant-numeric:tabular-nums lining-nums}
/* Double Streamlit's large (2rem) logo while preserving the artwork's proportions. */
[data-testid="stLogo"],[data-testid="stHeaderLogo"]{
  height:4rem;width:auto;max-width:100%;object-fit:contain}
[data-testid="stSidebarHeader"]{height:auto;min-height:4.5rem}
[data-testid="stHeader"]:has([data-testid="stHeaderLogo"]){height:5rem}
[data-testid="stAppViewContainer"]:has([data-testid="stHeaderLogo"]) .block-container{
  padding-top:6rem}
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"]{color:var(--f1-on-accent)}
.block-container{padding-top:3.5rem;padding-bottom:1.25rem}
[data-testid="stMetric"]{background:var(--f1-card);border:1px solid var(--f1-border);
  border-radius:6px;padding:.75rem .9rem;transition:border-color 120ms ease}
[data-testid="stMetricLabel"],.f1-micro{font-size:.7rem;font-weight:650;
  letter-spacing:.09em;text-transform:uppercase;color:var(--f1-muted)}
[data-testid="stMetricValue"],[data-testid="stMetricDelta"],.f1-num,
.f1-tower .pos,.f1-tower .val,.f1-tag,.f1-meta,.f1-team-value{
  font-variant-numeric:tabular-nums lining-nums;font-feature-settings:"tnum" 1,"lnum" 1}
[data-testid="stMetricValue"],.f1-tower .pos,.f1-tower .val,.f1-team-value{
  font-family:var(--f1-numeric);letter-spacing:-.025em}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:var(--f1-border)}
.f1-hero{display:flex;align-items:flex-end;gap:1rem;color:var(--f1-ink);
  border-left:3px solid var(--f1-accent);padding:.15rem 0 .15rem .9rem;margin:.4rem 0 .5rem;
  min-width:0}
.f1-logo{display:block;width:336px;height:112px;max-width:100%;object-fit:contain;flex:none}
.f1-heading{flex:1;min-width:0}
.f1-title{font-family:"Arial Narrow","Roboto Condensed",system-ui,sans-serif;
  font-stretch:condensed;font-size:1.5rem;font-weight:750;line-height:1.15;
  margin:0 0 .3rem;padding:0;letter-spacing:-.025em;color:var(--f1-ink)}
.f1-meta{font-size:.83rem;line-height:1.45;color:var(--f1-ink2);max-width:78ch}
.f1-kicker,.f1-status{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center}
.f1-kicker{margin-top:.5rem}
.f1-status{justify-content:flex-end;max-width:28rem}
.f1-tag{display:inline-flex;align-items:center;gap:.35rem;padding:3px 7px;
  border-radius:3px;font-size:.68rem;line-height:1.4;font-weight:650;
  border:1px solid var(--f1-border);color:var(--f1-ink2);overflow-wrap:anywhere}
.f1-tag.solid{background:var(--f1-raised);color:var(--f1-ink)}
.f1-tag.locked{color:var(--f1-locked)}
.f1-tag.verified{color:var(--f1-verified)}
.f1-tag.warn{color:var(--f1-warning)}
.f1-tower{display:flex;flex-direction:column;gap:4px;margin:.25rem 0 .4rem}
.f1-tower .row,.f1-team-row{display:flex;align-items:center;gap:.65rem;padding:.5rem .65rem;
  border:1px solid var(--f1-border);border-radius:4px;background:var(--f1-card);
  color:var(--f1-ink);transition:border-color 120ms ease,background 120ms ease}
.f1-tower .pos,.f1-team-rank{min-width:2.2rem;height:1.65rem;display:inline-flex;
  align-items:center;justify-content:center;font-weight:700;background:var(--f1-raised);
  border-radius:3px;font-size:.85rem}
.f1-tower .drv,.f1-team-driver{flex:1;min-width:0;font-weight:600;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.f1-tower .val,.f1-team-value{color:var(--f1-ink2);text-align:right;white-space:nowrap}
.f1-team-row{border-left:3px solid var(--team-accent);margin:.25rem 0}
.f1-team-pill{display:inline-flex;align-items:center;gap:.4rem;padding:2px 7px;
  border:1px solid var(--f1-border);border-radius:3px;font-size:.75rem;
  background:var(--f1-card);color:var(--f1-ink)}
.f1-team-pill:before{content:"";display:block;width:4px;height:.85rem;
  background:var(--team-accent);border:1px solid var(--f1-border)}
.f1-team-caption,.f1-card-caption{display:block;font-size:.78rem;line-height:1.4;
  font-weight:400;color:var(--f1-muted)}
.f1-empty{border:1px dashed var(--f1-border);background:var(--f1-card);
  border-radius:6px;padding:1rem;margin:.4rem 0 .8rem;color:var(--f1-ink)}
.f1-empty .h{font-weight:700;font-size:1rem;margin-bottom:.3rem}
.f1-empty .b{color:var(--f1-ink2);line-height:1.5}
.f1-brand{font-weight:800;letter-spacing:.1em;font-size:.78rem}
.f1-brand span{color:var(--f1-accent)}
.f1-card{border:1px solid var(--f1-border);border-radius:6px;padding:.9rem;
  background:var(--f1-card);color:var(--f1-ink);transition:border-color 120ms ease}
.f1-card-title{font-weight:650;margin-bottom:.4rem}
.f1-tooltip{cursor:help;border-bottom:1px dotted var(--f1-muted)}
.f1-tooltip:focus-visible{outline:2px solid var(--f1-locked);outline-offset:3px}
.st-key-f1_context_ribbon{position:sticky;top:2.9rem;z-index:50;
  background:var(--f1-surface);padding:.25rem 0}
/* Top navigation: a radio (stable test contract) drawn as a tab bar. */
.st-key-lab_page [role="radiogroup"]{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:0;width:100%;padding:0;border-bottom:1px solid var(--f1-border)}
.st-key-lab_page label[data-testid="stRadioOption"]{display:flex;align-items:center;
  justify-content:center;min-height:46px;box-sizing:border-box;margin:0;padding:.55rem .8rem;
  border:0;border-bottom:3px solid transparent;border-radius:0;background:transparent;
  color:var(--f1-muted);cursor:pointer;transition:color 120ms ease,border-color 120ms ease}
.st-key-lab_page label[data-testid="stRadioOption"] p{font-weight:600;font-size:.95rem}
.st-key-lab_page label[data-testid="stRadioOption"]:hover{color:var(--f1-ink);
  border-bottom-color:var(--f1-border)}
.st-key-lab_page label[data-testid="stRadioOption"]:has(input:checked){
  color:var(--f1-ink);border-bottom-color:var(--f1-accent)}
.st-key-lab_page label[data-testid="stRadioOption"]:has(input:checked) p{font-weight:750}
.st-key-lab_page label[data-testid="stRadioOption"]:has(input:focus-visible){
  outline:2px solid var(--f1-locked);outline-offset:-2px}
/* Hide the radio dots on tab-like and segmented radios (labels stay readable). */
.st-key-lab_page label[data-testid="stRadioOption"]>div>div:not([data-testid]),
.st-key-lab_forecast_tab label[data-testid="stRadioOption"]>div>div:not([data-testid]),
.st-key-lab_live_task label[data-testid="stRadioOption"]>div>div:not([data-testid]){
  display:none}
/* Prediction switch: segmented control look. */
.st-key-lab_forecast_tab [role="radiogroup"],.st-key-lab_live_task [role="radiogroup"]{
  display:inline-flex;flex-wrap:wrap;gap:0;padding:3px;border:1px solid var(--f1-border);
  border-radius:8px;background:var(--f1-card)}
.st-key-lab_forecast_tab label[data-testid="stRadioOption"],
.st-key-lab_live_task label[data-testid="stRadioOption"]{margin:0;padding:.4rem .95rem;
  border-radius:6px;color:var(--f1-ink2);cursor:pointer}
.st-key-lab_forecast_tab label[data-testid="stRadioOption"]:has(input:checked),
.st-key-lab_live_task label[data-testid="stRadioOption"]:has(input:checked){
  background:var(--f1-accent);color:var(--f1-on-accent)}
.st-key-lab_forecast_tab label[data-testid="stRadioOption"]:has(input:checked) p,
.st-key-lab_live_task label[data-testid="stRadioOption"]:has(input:checked) p{
  color:var(--f1-on-accent);font-weight:700}
.st-key-lab_forecast_tab label[data-testid="stRadioOption"]:has(input:focus-visible),
.st-key-lab_live_task label[data-testid="stRadioOption"]:has(input:focus-visible){
  outline:2px solid var(--f1-locked);outline-offset:1px}
/* Race setup card and refresh card. */
.st-key-f1_setup,.st-key-f1_refresh_card{background:var(--f1-card);
  border-top:3px solid var(--f1-accent)!important}
/* Scoreboard: the best model gets an accent border and a text badge in its label. */
[class*="st-key-f1_score_best"] [data-testid="stMetric"]{border-color:var(--f1-verified);
  box-shadow:inset 0 3px 0 var(--f1-verified)}
[class*="st-key-f1_score_base"] [data-testid="stMetric"]{border-style:dashed;
  background:transparent}
[class*="st-key-f1_model_card"]{height:100%}
/* Timestamps are long; keep them on one line inside the metric tiles. */
.st-key-f1_cutoffs [data-testid="stMetricValue"]{font-size:1.15rem;line-height:2.2rem}
.f1-model-comparison{font-size:.76rem;line-height:1.6;color:var(--f1-ink2);
  margin:.5rem 0;overflow-wrap:anywhere}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.65rem}
[data-testid="stSidebar"] h3{font-size:1rem;padding:.55rem 0 .3rem}
.f1-context{display:flex;align-items:center;flex-wrap:wrap;gap:.45rem 1.4rem;
  padding:.55rem .85rem;border:1px solid var(--f1-border);border-radius:6px;
  background:var(--f1-card);color:var(--f1-ink);font-variant-numeric:tabular-nums}
.f1-context-field{display:flex;flex-direction:column;gap:.1rem;min-width:0}
.f1-context-field:first-child{flex:0 1 auto}
.f1-context-field:first-child .f1-context-value{font-size:.92rem;font-weight:700}
.f1-context-value{font-size:.8rem;line-height:1.35;overflow-wrap:anywhere}
.f1-context-artifact{display:flex;flex-direction:column;gap:.1rem;min-width:0;
  margin-left:auto;max-width:22rem}
.f1-context-artifact code{font-family:var(--f1-numeric);font-size:.68rem;color:var(--f1-muted);
  background:transparent;padding:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.f1-context-badges{display:flex;gap:.35rem;flex-wrap:wrap}
.f1-context-pending{flex-basis:100%;font-size:.78rem;color:var(--f1-warning);
  border-top:1px dashed var(--f1-border);padding-top:.4rem}
.f1-spotlight{border:1px solid var(--f1-border);border-top:3px solid var(--team-accent);
  border-radius:5px;background:var(--f1-card);padding:.85rem;min-height:12rem;
  color:var(--f1-ink);font-variant-numeric:tabular-nums}
.f1-spotlight-driver{font-size:1.1rem;font-weight:700;line-height:1.3;margin:.4rem 0}
.f1-spotlight .f1-team-pill{margin-bottom:.65rem}
.f1-probability-label{display:flex;justify-content:space-between;gap:.5rem;
  color:var(--f1-ink2);font-size:.8rem;margin:.3rem 0}
.f1-probability-track{height:6px;background:var(--f1-raised);border:1px solid var(--f1-border);
  border-radius:2px;overflow:hidden;margin-bottom:.65rem}
.f1-probability-fill{height:100%;background:var(--team-accent)}
.f1-challenger{margin:.5rem 0 .75rem;display:flex;align-items:center;gap:1rem;
  flex-wrap:wrap;border:1px solid var(--f1-border);border-left:3px solid var(--team-accent);
  border-radius:4px;padding:.75rem;background:var(--f1-card);color:var(--f1-ink)}
.f1-challenger-probability{flex:1;min-width:10rem}
@media(hover:hover){
  .f1-card:hover,.f1-team-row:hover,.f1-tower .row:hover,[data-testid="stMetric"]:hover{
    border-color:var(--f1-muted)}
}
@media(max-width:900px){
  .f1-hero{flex-wrap:wrap;gap:.65rem}
  .f1-status{max-width:100%;justify-content:flex-start;flex-basis:100%}
}
@media(max-width:700px){
  .st-key-lab_page [role="radiogroup"]{grid-template-columns:repeat(2,minmax(0,1fr))}
  .f1-context-artifact{margin-left:0;max-width:100%}
  .st-key-f1_context_ribbon{position:static}
}
@media(max-width:540px){
  .block-container{padding-top:3.25rem;padding-bottom:1rem}
  .f1-hero{padding:.75rem;gap:.6rem}
  .f1-logo{width:264px;height:88px}
  .f1-title{font-size:1.2rem}
  .f1-meta{font-size:.78rem}
  .f1-team-row{flex-wrap:wrap}
}
@media(prefers-reduced-motion:reduce){
  .f1-card,.f1-team-row,.f1-tower .row,[data-testid="stMetric"]{transition:none}
}
</style>
"""

# Retained as a public constant for existing callers. Theme overrides are now variables.
DARK_CSS = "<style>:root{color-scheme:dark}</style>"


def inject():
    st.html(CSS)
    t = tokens()
    names = {"grid": "border"}
    variables = ";".join(
        f"--f1-{names.get(key, key).replace('_', '-')}:{t[key]}"
        for key in (
            "surface",
            "card",
            "raised",
            "ink",
            "ink2",
            "muted",
            "grid",
            "accent",
            "on_accent",
            "locked",
            "verified",
            "warning",
        )
    )
    st.html(f"<style>:root{{{variables}}}</style>")
    if t is DARK:
        st.html(DARK_CSS)
