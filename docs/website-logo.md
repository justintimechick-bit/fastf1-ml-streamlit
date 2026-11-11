# Pit-wall telemetry design

The app uses the generated `src/f1ml_app/assets/logo-ml.png` lockup: FastF1 plus
a neural network emblem. `ui.BRAND_LOGO` is the shared asset path used by the
navigation logo and the optional logo in the shared horizontal header. Navigation
uses Streamlit's large logo size with a CSS height of 4rem, twice the default
large size. The optional header image is 336 × 112px on desktop and 264 × 88px
on mobile, preserving the artwork's 3:1 aspect ratio. Page headers show the title, subtitle,
and wrapping status badges without repeating the navigation logo.

The default theme is dark: carbon `#080d12`, card `#121b25`, raised surface
`#1b2937`, hairline `#304152`, text `#f4f7fa`, and red accent `#ff5263`.
Locked predictions use cyan, verified receipts green, and warnings amber.
An explicit light theme remains available. `theme.tokens()` keeps its existing
chart keys and adds surface and status tokens. `theme.inject()` supplies matching
`--f1-*` CSS variables, compact page padding, and tabular numeric styling.
Metric values and timing-tower numbers use a local monospace font stack;
headings use a condensed sans-serif stack without remote font downloads.

`ui.hero(kicker, title, meta)` keeps its existing signature. The active real-data
lab uses `ui.brand_header(title, subtitle, dataset_version=..., verified_run=...)`.
The config badge reads `experiment_id` from `configs/real-20261105.toml`;
the separate dataset badge identifies the actual loaded dataset directory.
Only pass runs returned by the verified saved-run loader. The prediction lock
badge requires both a seal timestamp and `sealed_before_target_scoring = true`
from that run's `prediction_lock.json`. A prediction lock does not establish an
untouched holdout; the exploratory evaluation badge preserves that distinction.

Constructor accents are presentation tokens in `theme.TEAM_COLORS`, keyed by
canonical `team_id` from `configs/team_mapping.json`. `theme.team_color(name)`
also resolves configured source IDs and aliases, ignoring case and separator
differences, and returns `#98AABD` for unknown teams. Colors deliberately remain
outside the mapping document because its checksum identifies ingestion history.
These are interface accents, not a claim of official team brand specifications.

Reusable helpers (plain text inputs are HTML-escaped):

```python
from f1ml_app import ui

st.html(ui.team_pill("McLaren"))
st.html(ui.team_row("Driver name", "mclaren", "72.4%", rank=2,
                    caption="Race podium", tooltip="Independent podium probability"))
ui.metric_card("Rank MAE", "2.18", tooltip="Lower is better", caption="Known ranks only")
ui.info_card("Prediction cutoff", "Before qualifying", label="Availability",
             tooltip="Only earlier published information is used")
```

The prediction table also offers a collapsed constructor timing tower. Constructor
colors decorate a stripe and pill; readable team names remain visible. Cards have
quiet border hover states; tooltips can receive keyboard focus, and reduced-motion
preferences disable transitions.

## Logo artwork

`src/f1ml_app/assets/logo-ml.png` combines the supplied FastF1 banner with a
generated neural network emblem and a plus sign. It is the active website logo.
The previous Pit Wall mark remains in `src/f1ml_app/assets/logo.svg` as an unused
asset. SVG and PNG assets are included in source and wheel packages.

Original banner: `src/f1ml_app/assets/banner.png`, copied unchanged from
`~/Downloads/banner.png`. It is not rendered as a full-width page banner.

The PNG lockup was created with the built-in image generation tool. Final prompt:

```text
Use case: compositing, logo-brand.
Asset type: transparent horizontal website logo for an F1 machine learning app.
Input image 1 is the existing FastF1 banner and the edit target.
Primary request: combine the provided FastF1 banner with a plus sign and a new matching machine learning logo to its right, producing one clean, horizontal logo lockup.
Keep the original FastF1 car, bar chart and exact "FastF1" lettering intact and fully legible on the left. Preserve their shapes, proportions, colors and relative layout. Add a generous gap, a simple slate gray plus sign, then an original minimalist neural network emblem: three connected layers of circular nodes, with an angular fast forward silhouette, suggesting machine learning. Match the existing flat graphic style, with slate gray connections, orange and red accents, and a small green accent. The neural network emblem should be about as tall as the FastF1 wordmark so both parts have balanced visual weight.
Composition: one very wide horizontal lockup, tightly framed with modest clear padding; no extra title or subtitle. Genuine transparent background. No glow, shadows, gradients, mockup surfaces or watermark. Do not add any text beyond the source FastF1 lettering.
```
