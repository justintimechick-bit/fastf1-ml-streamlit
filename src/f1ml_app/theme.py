"""Design tokens. Chart hues are the validated categorical slots 1–3 (light and dark steps)."""

import streamlit as st

LIGHT = {
    "surface": "#fbfbfa",
    "ink": "#141413",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "neutral": "#aeaca4",
    "on_neutral": "#141413",
    "series": ("#2a78d6", "#eb6834", "#1baf7a"),
}
DARK = {
    "surface": "#141416",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "neutral": "#7d7b75",
    "on_neutral": "#ffffff",
    "series": ("#3987e5", "#d95926", "#199e70"),
}


def tokens() -> dict:
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - outside a browser session (tests), use light
        kind = None
    return DARK if kind == "dark" else LIGHT


# Only classes owned by this app are styled; Streamlit internals are left alone.
CSS = """
<style>
.f1-hero{position:relative;overflow:hidden;background:#141419;color:#f3f2ef;
  border-radius:14px;padding:20px 150px 18px 28px;margin:0 0 .4rem 0}
.f1-hero:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:#d7263d}
.f1-hero:after{content:"";position:absolute;right:0;top:0;bottom:0;width:140px;opacity:.13;
  background:repeating-conic-gradient(#ffffff 0 25%,transparent 0 50%) 0 0/14px 14px;
  -webkit-mask-image:linear-gradient(to left,#000 10%,transparent);
  mask-image:linear-gradient(to left,#000 10%,transparent)}
.f1-kicker{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#bdbcb6;
  font-weight:650;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.f1-title{font-size:1.85rem;font-weight:720;line-height:1.15;margin:.35rem 0 .3rem;
  letter-spacing:-.012em;color:#ffffff}
.f1-meta{font-size:.92rem;color:#d6d5d0;font-variant-numeric:tabular-nums}
.f1-tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.68rem;
  font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  border:1px solid rgba(255,255,255,.28);color:#f3f2ef}
.f1-tag.solid{background:#f3f2ef;color:#141419;border-color:#f3f2ef}
.f1-tag.warn{background:#fab219;color:#141413;border-color:#fab219}
.f1-tower{display:flex;flex-direction:column;gap:6px;margin:.25rem 0 .2rem}
.f1-tower .row{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:8px;
  border:1px solid rgba(128,128,128,.22)}
.f1-tower .pos{min-width:30px;height:26px;border-radius:6px;background:#141419;color:#fff;
  display:inline-flex;align-items:center;justify-content:center;font-weight:700;
  font-variant-numeric:tabular-nums;font-size:.85rem}
.f1-tower .drv{flex:1;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.f1-tower .val{font-variant-numeric:tabular-nums;opacity:.85}
.f1-empty{border:1px dashed rgba(128,128,128,.45);border-radius:12px;padding:18px 20px;
  margin:.4rem 0 .8rem}
.f1-empty .h{font-weight:700;font-size:1.05rem;margin-bottom:.3rem}
.f1-empty .b{opacity:.86;line-height:1.5}
.f1-brand{font-weight:800;letter-spacing:.16em;text-transform:uppercase;font-size:.78rem}
.f1-brand span{color:#d7263d}
</style>
"""


DARK_CSS = """
<style>
.f1-hero{background:#25252d;box-shadow:inset 0 0 0 1px rgba(255,255,255,.07)}
.f1-tower .pos{background:#f3f2ef;color:#141419}
</style>
"""


def inject():
    st.html(CSS)
    if tokens() is DARK:  # the dark page is already near-black; lift the dark accents
        st.html(DARK_CSS)
