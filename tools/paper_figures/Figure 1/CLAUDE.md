# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

No build step. Serve the directory over HTTP so the browser can load the `.jsx` files:

```bash
python3 -m http.server 8080
# open http://localhost:8080/Figure%201.html
```

Opening `Figure 1.html` as a `file://` URL will fail — browsers block cross-origin script fetches for local files. Babel standalone transpiles all JSX in-browser at runtime.

## Directory structure

```
Figure 1/
├── Figure 1.html       # sole entry point
├── CLAUDE.md
├── src/                # JSX source files
│   ├── design-canvas.jsx
│   ├── figure-shared.jsx
│   └── figure-clean.jsx
└── scripts/            # asset generation
    ├── generate_icons.py   # produces amplitude.svg + phase.svg
    ├── generate_tcav.py    # produces tcav.svg
    ├── amplitude.svg
    ├── phase.svg
    ├── tcav.svg
    └── venv/
```

## Architecture

`Figure 1.html` is the sole entry point. It loads React 18, ReactDOM, and Babel standalone from CDN, then pulls in the `src/` `.jsx` files as `<script type="text/babel">` tags. Each file exposes its components as **bare globals** (no `export`) — this is intentional; Babel standalone shares the window scope across all script tags.

**Load order matters** (later scripts may reference globals from earlier ones):

| File | Globals exposed |
|---|---|
| `src/design-canvas.jsx` | `DesignCanvas`, `DCSection`, `DCArtboard`, `DCPostIt` |
| `src/figure-shared.jsx` | `FIG_FONTS`, `DEFAULT_STAGE_COLORS`, `tint`, `EEGTrace`, `TokenGrid`, `SAEColumn`, `SpectrumCurve`, `Arrow`, `StageHeader`, `StageBody`, `StageCard`, `FigCaption` |
| `src/figure-clean.jsx` | `FigureClean` |

## State & tweaks wiring

`App` in `Figure 1.html` owns all state:

```js
const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
```

`useTweaks` (from `tweaks-panel.jsx`) syncs state via a `window.postMessage` protocol (`__edit_mode_set_keys`, `__edit_mode_available`, etc.) — this is a bridge for a live-edit host that can rewrite the `/*EDITMODE-BEGIN*/…/*EDITMODE-END*/` block in the HTML file on disk. Do not remove those markers.

The `colors` object passed to every figure component is derived from `t.palette` against the `PALETTES` map (defined inline in the HTML):

```js
{ spectral, sae, tcav, steer }  // one hex per pipeline stage
```

## Key design patterns

**`src/design-canvas.jsx`** — pan/zoom canvas with persistent artboard layout. State is read/written to a `.design-canvas.state.json` sidecar via `window.omelette.writeFile()` (a host bridge). Pan/zoom is ref-based and written directly to the DOM to avoid React re-renders at 60fps. Artboard drag-reorder uses live CSS `transform` on siblings during drag, then commits the new order on `pointerup`.

**`src/figure-shared.jsx`** — pure SVG schematic primitives with no state. `EEGTrace` and `TokenGrid` use deterministic pseudo-random generation (seeded sine waves / index-based sparse patterns) so the visuals are stable across re-renders. `tint(hex, alpha)` converts hex to `rgba` and is used throughout for tinted backgrounds.

**`src/figure-clean.jsx`** — the submission figure. Four-column grid (`COL_W = 280`). Inline math uses `<M>` (upright) and `<Mi>` (italic) span wrappers that set `fontFamily` directly. Trapezoid shapes (encoder/decoder) are drawn as SVG rounded-polygon paths via `roundedPolyPath(pts, r)`.
