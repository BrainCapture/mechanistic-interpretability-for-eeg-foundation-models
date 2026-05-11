// Figure 1 — clean four-strip layout. Diagrammatic, minimal prose.
//
// Style notes:
//   - Header bar is a 2-row layout (numeral row + title row) so the title
//     never wraps, regardless of column width.
//   - Boxes use HTML/CSS only; arrows between boxes are drawn with absolute
//     positioning so they always start at box edges, not through the boxes.
//   - "Math" labels are not actually italic-on-tint; we use upright math font
//     so subscripts stop overlapping at small sizes on tinted backgrounds.

const COL_W = 280;

// ── Atoms ────────────────────────────────────────────────────────────────────

// Math span — upright math glyphs (no italic) so subscripts don't collide on
// tinted backgrounds. Subscripts use real <sub> tags.
function M({ children }) {
  return (
    <span style={{ fontFamily: FIG_FONTS.math, fontStyle: 'normal' }}>
      {children}
    </span>
  );
}

// Italic math (used sparingly, on white only)
function Mi({ children }) {
  return (
    <span style={{ fontFamily: FIG_FONTS.math, fontStyle: 'italic' }}>
      {children}
    </span>
  );
}

// Box — rounded rectangle with a primary label and optional sub-label.
// The two lines are clearly separated with margin so they never overlap.
function Box({
  children, sub,
  w, h,
  fill = '#fff', border = '#9aa3a8', color = '#111',
  style = {},
}) {
  return (
    <div style={{
      width: w, height: h,
      padding: '6px 10px',
      background: fill,
      border: `1px solid ${border}`,
      borderRadius: 8,
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      fontFamily: FIG_FONTS.sans,
      fontSize: 14,
      color, lineHeight: 1.25, textAlign: 'center', boxSizing: 'border-box',
      ...style,
    }}>
      <div style={{ fontWeight: 500 }}>{children}</div>
      {sub && (
        <div style={{ fontSize: 12, color: '#666', marginTop: 3, lineHeight: 1.25 }}>{sub}</div>
      )}
    </div>
  );
}

// Trapezoid — encoder/decoder shape. orient: 'shrink' (big→small, encoder), 'expand' (small→big, decoder).
// 'shrinkV' (big top → small bottom), 'expandV' (small top → big bottom) for vertical flows.
function Trapezoid({ w = 80, h = 56, orient = 'shrink', fill = '#fff', stroke = '#666', label, sub, color = '#111' }) {
  const pad = 0.32; // taper amount
  const r = 6; // corner rounding
  let p; // four corners (TL, TR, BR, BL)
  if (orient === 'shrink') {
    // horizontal: left tall, right short — (0,0)→(w,h*pad)→(w,h*(1-pad))→(0,h)
    p = [[0, 0], [w, h * pad], [w, h * (1 - pad)], [0, h]];
  } else if (orient === 'expand') {
    p = [[0, h * pad], [w, 0], [w, h], [0, h * (1 - pad)]];
  } else if (orient === 'shrinkV') {
    // top wide, bottom narrow
    p = [[0, 0], [w, 0], [w * (1 - pad), h], [w * pad, h]];
  } else { // expandV
    p = [[w * pad, 0], [w * (1 - pad), 0], [w, h], [0, h]];
  }
  // Build a rounded-corner path through the four points.
  const path = roundedPolyPath(p, r);
  return (
    <div style={{ position: 'relative', width: w, height: h, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <svg width={w} height={h} style={{ position: 'absolute', inset: 0, overflow: 'visible' }}>
        <path d={path} fill={fill} stroke={stroke} strokeWidth={2} strokeLinejoin="round" />
      </svg>
      <div style={{ position: 'relative', textAlign: 'center', fontFamily: FIG_FONTS.sans, fontSize: 18, color, lineHeight: 1.2, padding: '0 6px' }}>
        <div style={{ fontWeight: 600 }}>{label}</div>
        {sub && <div style={{ fontSize: 14, color: '#666', marginTop: 2 }}>{sub}</div>}
      </div>
    </div>
  );
}

// Build an SVG path that traces a polygon with rounded corners.
function roundedPolyPath(pts, r) {
  const n = pts.length;
  const v = (a, b) => [b[0] - a[0], b[1] - a[1]];
  const len = (u) => Math.hypot(u[0], u[1]);
  const norm = (u) => { const L = len(u) || 1; return [u[0] / L, u[1] / L]; };
  let d = '';
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const curr = pts[i];
    const next = pts[(i + 1) % n];
    const inDir = norm(v(prev, curr));   // direction arriving at curr
    const outDir = norm(v(curr, next));  // direction leaving curr
    const inLen = len(v(prev, curr));
    const outLen = len(v(curr, next));
    const rr = Math.min(r, inLen / 2, outLen / 2);
    const start = [curr[0] - inDir[0] * rr, curr[1] - inDir[1] * rr];
    const end = [curr[0] + outDir[0] * rr, curr[1] + outDir[1] * rr];
    if (i === 0) d += `M ${start[0].toFixed(2)} ${start[1].toFixed(2)} `;
    else d += `L ${start[0].toFixed(2)} ${start[1].toFixed(2)} `;
    d += `Q ${curr[0].toFixed(2)} ${curr[1].toFixed(2)} ${end[0].toFixed(2)} ${end[1].toFixed(2)} `;
  }
  d += 'Z';
  return d;
}

// Mini icon — 5 bars following ~1/f (one per band)
function MiniSpectrum({ w = 56, h = 24, color = '#222', bars = 5 }) {
  const gap = 2;
  const padX = 2;
  const barW = (w - padX * 2 - gap * (bars - 1)) / bars;
  // heights ~ 1/f, normalised to fit
  const raw = Array.from({ length: bars }, (_, i) => 1 / (i + 1));
  const max = raw[0];
  const usableH = h - 4;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <line x1={padX - 0.5} y1={h - 2} x2={w - padX + 0.5} y2={h - 2} stroke="#bbb" strokeWidth={0.6} />
      {raw.map((r, i) => {
        const bh = Math.max(2, (r / max) * usableH);
        const x = padX + i * (barW + gap);
        const y = h - 2 - bh;
        return <rect key={i} x={x} y={y} width={barW} height={bh} fill={color} rx={0.8} />;
      })}
    </svg>
  );
}

// Mini icon — unit-circle for phase (cos, sin) pair
function MiniPhase({ size = 26, color = '#222' }) {
  const r = size / 2 - 2;
  const cx = size / 2, cy = size / 2;
  // a sample phase angle
  const ang = -0.7;
  const px = cx + Math.cos(ang) * r;
  const py = cy + Math.sin(ang) * r;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#bbb" strokeWidth={0.7} />
      <line x1={cx - r - 0.5} y1={cy} x2={cx + r + 0.5} y2={cy} stroke="#ddd" strokeWidth={0.5} />
      <line x1={cx} y1={cy - r - 0.5} x2={cx} y2={cy + r + 0.5} stroke="#ddd" strokeWidth={0.5} />
      <line x1={cx} y1={cy} x2={px} y2={py} stroke={color} strokeWidth={1.3} />
      <circle cx={px} cy={py} r={1.6} fill={color} />
    </svg>
  );
}

// CAV mini-plot — 2D scatter with two classes and a separating hyperplane (the CAV).
function MiniCAVPlot({ w = 260, h = 110, color = '#E08A1F' }) {
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <rect x={0} y={0} width={w} height={h} fill="#FAFAF7" stroke="#E5E0D6" strokeWidth={0.7} rx={6} />
      
      {/* Insert the matplotlib generated plot, moved slightly more to the right */}
      <image href={"data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIKICAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgd2lkdGg9IjEyOC43ODgzODdwdCIgaGVpZ2h0PSI4Ny44NHB0IiB2aWV3Qm94PSIwIDAgMTI4Ljc4ODM4NyA4Ny44NCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2ZXJzaW9uPSIxLjEiPgogPG1ldGFkYXRhPgogIDxyZGY6UkRGIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6Y2M9Imh0dHA6Ly9jcmVhdGl2ZWNvbW1vbnMub3JnL25zIyIgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgPGNjOldvcms+CiAgICA8ZGM6dHlwZSByZGY6cmVzb3VyY2U9Imh0dHA6Ly9wdXJsLm9yZy9kYy9kY21pdHlwZS9TdGlsbEltYWdlIi8+CiAgICA8ZGM6ZGF0ZT4yMDI2LTA0LTI4VDIxOjM3OjMzLjY0ODQ4MjwvZGM6ZGF0ZT4KICAgIDxkYzpmb3JtYXQ+aW1hZ2Uvc3ZnK3htbDwvZGM6Zm9ybWF0PgogICAgPGRjOmNyZWF0b3I+CiAgICAgPGNjOkFnZW50PgogICAgICA8ZGM6dGl0bGU+TWF0cGxvdGxpYiB2My4xMC45LCBodHRwczovL21hdHBsb3RsaWIub3JnLzwvZGM6dGl0bGU+CiAgICAgPC9jYzpBZ2VudD4KICAgIDwvZGM6Y3JlYXRvcj4KICAgPC9jYzpXb3JrPgogIDwvcmRmOlJERj4KIDwvbWV0YWRhdGE+CiA8ZGVmcz4KICA8c3R5bGUgdHlwZT0idGV4dC9jc3MiPip7c3Ryb2tlLWxpbmVqb2luOiByb3VuZDsgc3Ryb2tlLWxpbmVjYXA6IGJ1dHR9PC9zdHlsZT4KIDwvZGVmcz4KIDxnIGlkPSJmaWd1cmVfMSI+CiAgPGcgaWQ9InBhdGNoXzEiPgogICA8cGF0aCBkPSJNIDAgODcuODQgCkwgMTI4Ljc4ODM4NyA4Ny44NCAKTCAxMjguNzg4Mzg3IDAgCkwgMCAwIApMIDAgODcuODQgCnoKIiBzdHlsZT0iZmlsbDogbm9uZSIvPgogIDwvZz4KICA8ZyBpZD0iYXhlc18xIj4KICAgPGcgaWQ9ImxpbmUyZF8xIj4KICAgIDxwYXRoIGQ9Ik0gMjEuOTMwMTI5IDg4Ljg0IApMIDIyLjAwNTQxNSA4OC43NjAxMSAKTCAyMy41NDY4MjUgODcuMTI0NDMxIApMIDI1LjA4ODIzNSA4NS40ODg3NTMgCkwgMjYuNjI5NjQ1IDgzLjg1MzA3NCAKTCAyOC4xNzEwNTUgODIuMjE3Mzk2IApMIDI5LjcxMjQ2NSA4MC41ODE3MTggCkwgMzEuMjUzODc1IDc4Ljk0NjAzOSAKTCAzMi43OTUyODYgNzcuMzEwMzYxIApMIDM0LjMzNjY5NiA3NS42NzQ2ODIgCkwgMzUuODc4MTA2IDc0LjAzOTAwNCAKTCAzNy40MTk1MTYgNzIuNDAzMzI2IApMIDM4Ljk2MDkyNiA3MC43Njc2NDcgCkwgNDAuNTAyMzM2IDY5LjEzMTk2OSAKTCA0Mi4wNDM3NDYgNjcuNDk2MjkgCkwgNDMuNTg1MTU3IDY1Ljg2MDYxMiAKTCA0NS4xMjY1NjcgNjQuMjI0OTMzIApMIDQ2LjY2Nzk3NyA2Mi41ODkyNTUgCkwgNDguMjA5Mzg3IDYwLjk1MzU3NyAKTCA0OS43NTA3OTcgNTkuMzE3ODk4IApMIDUxLjI5MjIwNyA1Ny42ODIyMiAKTCA1Mi44MzM2MTggNTYuMDQ2NTQxIApMIDU0LjM3NTAyOCA1NC40MTA4NjMgCkwgNTUuOTE2NDM4IDUyLjc3NTE4NSAKTCA1Ny40NTc4NDggNTEuMTM5NTA2IApMIDU4Ljk5OTI1OCA0OS41MDM4MjggCkwgNjAuNTQwNjY4IDQ3Ljg2ODE0OSAKTCA2Mi4wODIwNzggNDYuMjMyNDcxIApMIDYzLjYyMzQ4OSA0NC41OTY3OTMgCkwgNjUuMTY0ODk5IDQyLjk2MTExNCAKTCA2Ni43MDYzMDkgNDEuMzI1NDM2IApMIDY4LjI0NzcxOSAzOS42ODk3NTcgCkwgNjkuNzg5MTI5IDM4LjA1NDA3OSAKTCA3MS4zMzA1MzkgMzYuNDE4NCAKTCA3Mi44NzE5NDkgMzQuNzgyNzIyIApMIDc0LjQxMzM2IDMzLjE0NzA0NCAKTCA3NS45NTQ3NyAzMS41MTEzNjUgCkwgNzcuNDk2MTggMjkuODc1Njg3IApMIDc5LjAzNzU5IDI4LjI0MDAwOCAKTCA4MC41NzkgMjYuNjA0MzMgCkwgODIuMTIwNDEgMjQuOTY4NjUyIApMIDgzLjY2MTgyIDIzLjMzMjk3MyAKTCA4NS4yMDMyMzEgMjEuNjk3Mjk1IApMIDg2Ljc0NDY0MSAyMC4wNjE2MTYgCkwgODguMjg2MDUxIDE4LjQyNTkzOCAKTCA4OS44Mjc0NjEgMTYuNzkwMjYgCkwgOTEuMzY4ODcxIDE1LjE1NDU4MSAKTCA5Mi45MTAyODEgMTMuNTE4OTAzIApMIDk0LjQ1MTY5MSAxMS44ODMyMjQgCkwgOTUuOTkzMTAyIDEwLjI0NzU0NiAKTCA5Ny41MzQ1MTIgOC42MTE4NjcgCkwgOTkuMDc1OTIyIDYuOTc2MTg5IApMIDEwMC42MTczMzIgNS4zNDA1MTEgCkwgMTAyLjE1ODc0MiAzLjcwNDgzMiAKTCAxMDMuNzAwMTUyIDIuMDY5MTU0IApMIDEwNS4yNDE1NjIgMC40MzM0NzUgCkwgMTA2LjU5MjQyMyAtMSAKIiBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIiBzdHlsZT0iZmlsbDogbm9uZTsgc3Ryb2tlLWRhc2hhcnJheTogNS45MiwyLjU2OyBzdHJva2UtZGFzaG9mZnNldDogMDsgc3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utd2lkdGg6IDEuNiIvPgogICA8L2c+CiAgIDxnIGlkPSJQYXRoQ29sbGVjdGlvbl8xIj4KICAgIDxkZWZzPgogICAgIDxwYXRoIGlkPSJDMF8wXzE3ZWUxM2MwYjEiIGQ9Ik0gMCAzLjUzNTUzNCAKQyAwLjkzNzYzNSAzLjUzNTUzNCAxLjgzNjk5MiAzLjE2MzAwOCAyLjUgMi41IApDIDMuMTYzMDA4IDEuODM2OTkyIDMuNTM1NTM0IDAuOTM3NjM1IDMuNTM1NTM0IC0wIApDIDMuNTM1NTM0IC0wLjkzNzYzNSAzLjE2MzAwOCAtMS44MzY5OTIgMi41IC0yLjUgCkMgMS44MzY5OTIgLTMuMTYzMDA4IDAuOTM3NjM1IC0zLjUzNTUzNCAwIC0zLjUzNTUzNCAKQyAtMC45Mzc2MzUgLTMuNTM1NTM0IC0xLjgzNjk5MiAtMy4xNjMwMDggLTIuNSAtMi41IApDIC0zLjE2MzAwOCAtMS44MzY5OTIgLTMuNTM1NTM0IC0wLjkzNzYzNSAtMy41MzU1MzQgMCAKQyAtMy41MzU1MzQgMC45Mzc2MzUgLTMuMTYzMDA4IDEuODM2OTkyIC0yLjUgMi41IApDIC0xLjgzNjk5MiAzLjE2MzAwOCAtMC45Mzc2MzUgMy41MzU1MzQgMCAzLjUzNTUzNCAKegoiLz4KICAgIDwvZGVmcz4KICAgIDxnIGNsaXAtcGF0aD0idXJsKCNwOGIzYzUzNjEwYykiPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI0MwXzBfMTdlZTEzYzBiMSIgeD0iMzEuMDQwNDcyIiB5PSIzMS42MDY4NSIgc3R5bGU9ImZpbGw6ICNlMDhhMWY7IGZpbGwtb3BhY2l0eTogMC44NSIvPgogICAgPC9nPgogICAgPGcgY2xpcC1wYXRoPSJ1cmwoI3A4YjNjNTM2MTBjKSI+CiAgICAgPHVzZSB4bGluazpocmVmPSIjQzBfMF8xN2VlMTNjMGIxIiB4PSIxNi4xODY0NTgiIHk9IjE2LjQ1NTA5NSIgc3R5bGU9ImZpbGw6ICNlMDhhMWY7IGZpbGwtb3BhY2l0eTogMC44NSIvPgogICAgPC9nPgogICAgPGcgY2xpcC1wYXRoPSJ1cmwoI3A4YjNjNTM2MTBjKSI+CiAgICAgPHVzZSB4bGluazpocmVmPSIjQzBfMF8xN2VlMTNjMGIxIiB4PSI0Mi45MjIwNzIiIHk9IjI0LjI3ODc3NSIgc3R5bGU9ImZpbGw6ICNlMDhhMWY7IGZpbGwtb3BhY2l0eTogMC44NSIvPgogICAgPC9nPgogICAgPGcgY2xpcC1wYXRoPSJ1cmwoI3A4YjNjNTM2MTBjKSI+CiAgICAgPHVzZSB4bGluazpocmVmPSIjQzBfMF8xN2VlMTNjMGIxIiB4PSI3LjY3MzU0NCIgeT0iMzQuNzA2OTY4IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjQwLjY0NzM4MSIgeT0iMTMuNzIwMzcyIiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjQ4LjE3MTQzOCIgeT0iMjQuMDI2IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjQ4LjMzNzc5MSIgeT0iNDYuNzM0NDQ1IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjY4LjE1ODgyNCIgeT0iMTAuNjc3Nzc4IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjUwLjY2Njg4OCIgeT0iOS44NjY3MjgiIHN0eWxlPSJmaWxsOiAjZTA4YTFmOyBmaWxsLW9wYWNpdHk6IDAuODUiLz4KICAgIDwvZz4KICAgIDxnIGNsaXAtcGF0aD0idXJsKCNwOGIzYzUzNjEwYykiPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI0MwXzBfMTdlZTEzYzBiMSIgeD0iNjIuMTA5MSIgeT0iMjguNTg2MjY1IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjE2LjkwOTgyMSIgeT0iNDMuNTAwODE0IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjQ3LjMxMDM3MiIgeT0iMzkuNzkzMjU5IiBzdHlsZT0iZmlsbDogI2UwOGExZjsgZmlsbC1vcGFjaXR5OiAwLjg1Ii8+CiAgICA8L2c+CiAgICA8ZyBjbGlwLXBhdGg9InVybCgjcDhiM2M1MzYxMGMpIj4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNDMF8wXzE3ZWUxM2MwYjEiIHg9IjQ1LjA1MTQ3MyIgeT0iMTcuMjUxODEiIHN0eWxlPSJmaWxsOiAjZTA4YTFmOyBmaWxsLW9wYWNpdHk6IDAuODUiLz4KICAgIDwvZz4KICAgIDxnIGNsaXAtcGF0aD0idXJsKCNwOGIzYzUzNjEwYykiPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI0MwXzBfMTdlZTEzYzBiMSIgeD0iNTIuMzExODc3IiB5PSI3LjY3MzU0NCIgc3R5bGU9ImZpbGw6ICNlMDhhMWY7IGZpbGwtb3BhY2l0eTogMC44NSIvPgogICAgPC9nPgogICAgPGcgY2xpcC1wYXRoPSJ1cmwoI3A4YjNjNTM2MTBjKSI+CiAgICAgPHVzZSB4bGluazpocmVmPSIjQzBfMF8xN2VlMTNjMGIxIiB4PSI0OC45NTE1OTciIHk9IjIwLjcwMDQ2NyIgc3R5bGU9ImZpbGw6ICNlMDhhMWY7IGZpbGwtb3BhY2l0eTogMC44NSIvPgogICAgPC9nPgogICA8L2c+CiAgIDxnIGlkPSJQYXRoQ29sbGVjdGlvbl8yIj4KICAgIDxkZWZzPgogICAgIDxwYXRoIGlkPSJtM2YzZDhhZjM0MCIgZD0iTSAwIDMuNTM1NTM0IApDIDAuOTM3NjM1IDMuNTM1NTM0IDEuODM2OTkyIDMuMTYzMDA4IDIuNSAyLjUgCkMgMy4xNjMwMDggMS44MzY5OTIgMy41MzU1MzQgMC45Mzc2MzUgMy41MzU1MzQgMCAKQyAzLjUzNTUzNCAtMC45Mzc2MzUgMy4xNjMwMDggLTEuODM2OTkyIDIuNSAtMi41IApDIDEuODM2OTkyIC0zLjE2MzAwOCAwLjkzNzYzNSAtMy41MzU1MzQgMCAtMy41MzU1MzQgCkMgLTAuOTM3NjM1IC0zLjUzNTUzNCAtMS44MzY5OTIgLTMuMTYzMDA4IC0yLjUgLTIuNSAKQyAtMy4xNjMwMDggLTEuODM2OTkyIC0zLjUzNTUzNCAtMC45Mzc2MzUgLTMuNTM1NTM0IDAgCkMgLTMuNTM1NTM0IDAuOTM3NjM1IC0zLjE2MzAwOCAxLjgzNjk5MiAtMi41IDIuNSAKQyAtMS44MzY5OTIgMy4xNjMwMDggLTAuOTM3NjM1IDMuNTM1NTM0IDAgMy41MzU1MzQgCnoKIiBzdHlsZT0ic3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utb3BhY2l0eTogMC45OyBzdHJva2Utd2lkdGg6IDEuNSIvPgogICAgPC9kZWZzPgogICAgPGcgY2xpcC1wYXRoPSJ1cmwoI3A4YjNjNTM2MTBjKSI+CiAgICAgPHVzZSB4bGluazpocmVmPSIjbTNmM2Q4YWYzNDAiIHg9IjEyMS4xMTQ4NDQiIHk9IjMwLjQxNDkwOCIgc3R5bGU9ImZpbGw6ICNmZmZmZmY7IGZpbGwtb3BhY2l0eTogMC45OyBzdHJva2U6ICNlMDhhMWY7IHN0cm9rZS1vcGFjaXR5OiAwLjk7IHN0cm9rZS13aWR0aDogMS41Ii8+CiAgICAgPHVzZSB4bGluazpocmVmPSIjbTNmM2Q4YWYzNDAiIHg9IjgzLjQyNzEwMyIgeT0iNjguNzIwMjU3IiBzdHlsZT0iZmlsbDogI2ZmZmZmZjsgZmlsbC1vcGFjaXR5OiAwLjk7IHN0cm9rZTogI2UwOGExZjsgc3Ryb2tlLW9wYWNpdHk6IDAuOTsgc3Ryb2tlLXdpZHRoOiAxLjUiLz4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNtM2YzZDhhZjM0MCIgeD0iNjcuNDY5ODY4IiB5PSI2Mi4zMTEyODEiIHN0eWxlPSJmaWxsOiAjZmZmZmZmOyBmaWxsLW9wYWNpdHk6IDAuOTsgc3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utb3BhY2l0eTogMC45OyBzdHJva2Utd2lkdGg6IDEuNSIvPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI20zZjNkOGFmMzQwIiB4PSI3MS4wNTUzNSIgeT0iODAuMTY2NDU2IiBzdHlsZT0iZmlsbDogI2ZmZmZmZjsgZmlsbC1vcGFjaXR5OiAwLjk7IHN0cm9rZTogI2UwOGExZjsgc3Ryb2tlLW9wYWNpdHk6IDAuOTsgc3Ryb2tlLXdpZHRoOiAxLjUiLz4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNtM2YzZDhhZjM0MCIgeD0iMTE3Ljk2ODAzNyIgeT0iNjMuNTExMzc0IiBzdHlsZT0iZmlsbDogI2ZmZmZmZjsgZmlsbC1vcGFjaXR5OiAwLjk7IHN0cm9rZTogI2UwOGExZjsgc3Ryb2tlLW9wYWNpdHk6IDAuOTsgc3Ryb2tlLXdpZHRoOiAxLjUiLz4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNtM2YzZDhhZjM0MCIgeD0iODIuMDk4NjYxIiB5PSI0MS44ODI1MDYiIHN0eWxlPSJmaWxsOiAjZmZmZmZmOyBmaWxsLW9wYWNpdHk6IDAuOTsgc3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utb3BhY2l0eTogMC45OyBzdHJva2Utd2lkdGg6IDEuNSIvPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI20zZjNkOGFmMzQwIiB4PSI5Mi4zNTA1MzEiIHk9IjU4LjIxOTYzNiIgc3R5bGU9ImZpbGw6ICNmZmZmZmY7IGZpbGwtb3BhY2l0eTogMC45OyBzdHJva2U6ICNlMDhhMWY7IHN0cm9rZS1vcGFjaXR5OiAwLjk7IHN0cm9rZS13aWR0aDogMS41Ii8+CiAgICAgPHVzZSB4bGluazpocmVmPSIjbTNmM2Q4YWYzNDAiIHg9IjExMS44ODE3OTMiIHk9Ijc5LjE1NDM3NiIgc3R5bGU9ImZpbGw6ICNmZmZmZmY7IGZpbGwtb3BhY2l0eTogMC45OyBzdHJva2U6ICNlMDhhMWY7IHN0cm9rZS1vcGFjaXR5OiAwLjk7IHN0cm9rZS13aWR0aDogMS41Ii8+CiAgICAgPHVzZSB4bGluazpocmVmPSIjbTNmM2Q4YWYzNDAiIHg9IjExMS4xNTU1OSIgeT0iNDEuMDkxMzQiIHN0eWxlPSJmaWxsOiAjZmZmZmZmOyBmaWxsLW9wYWNpdHk6IDAuOTsgc3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utb3BhY2l0eTogMC45OyBzdHJva2Utd2lkdGg6IDEuNSIvPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI20zZjNkOGFmMzQwIiB4PSI3MC41OTM2MzciIHk9Ijc1LjcxODkwNyIgc3R5bGU9ImZpbGw6ICNmZmZmZmY7IGZpbGwtb3BhY2l0eTogMC45OyBzdHJva2U6ICNlMDhhMWY7IHN0cm9rZS1vcGFjaXR5OiAwLjk7IHN0cm9rZS13aWR0aDogMS41Ii8+CiAgICAgPHVzZSB4bGluazpocmVmPSIjbTNmM2Q4YWYzNDAiIHg9IjgzLjk3NDc4OCIgeT0iNTQuODIwNzcyIiBzdHlsZT0iZmlsbDogI2ZmZmZmZjsgZmlsbC1vcGFjaXR5OiAwLjk7IHN0cm9rZTogI2UwOGExZjsgc3Ryb2tlLW9wYWNpdHk6IDAuOTsgc3Ryb2tlLXdpZHRoOiAxLjUiLz4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNtM2YzZDhhZjM0MCIgeD0iMTEwLjY2NzQwNSIgeT0iNTAuMjE4NjczIiBzdHlsZT0iZmlsbDogI2ZmZmZmZjsgZmlsbC1vcGFjaXR5OiAwLjk7IHN0cm9rZTogI2UwOGExZjsgc3Ryb2tlLW9wYWNpdHk6IDAuOTsgc3Ryb2tlLXdpZHRoOiAxLjUiLz4KICAgICA8dXNlIHhsaW5rOmhyZWY9IiNtM2YzZDhhZjM0MCIgeD0iODQuMjQ5MjE4IiB5PSIyNy4xMjYwNzEiIHN0eWxlPSJmaWxsOiAjZmZmZmZmOyBmaWxsLW9wYWNpdHk6IDAuOTsgc3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utb3BhY2l0eTogMC45OyBzdHJva2Utd2lkdGg6IDEuNSIvPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI20zZjNkOGFmMzQwIiB4PSIxMDQuNzQ3NzM5IiB5PSI2Ni4wNTMxNzMiIHN0eWxlPSJmaWxsOiAjZmZmZmZmOyBmaWxsLW9wYWNpdHk6IDAuOTsgc3Ryb2tlOiAjZTA4YTFmOyBzdHJva2Utb3BhY2l0eTogMC45OyBzdHJva2Utd2lkdGg6IDEuNSIvPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI20zZjNkOGFmMzQwIiB4PSI5Ni43MzEzNDkiIHk9IjMzLjkwMzQ2NSIgc3R5bGU9ImZpbGw6ICNmZmZmZmY7IGZpbGwtb3BhY2l0eTogMC45OyBzdHJva2U6ICNlMDhhMWY7IHN0cm9rZS1vcGFjaXR5OiAwLjk7IHN0cm9rZS13aWR0aDogMS41Ii8+CiAgICA8L2c+CiAgIDwvZz4KICAgPGcgaWQ9InBhdGNoXzIiPgogICAgPHBhdGggZD0iTSA1NC43MDcwNDEgMjcuNjk3MjUgCkwgNTguOTM1NjA1IDI5LjI5MzQ1NSAKTCA1Ny43NDgxNDcgMzAuNTUzNTM1IApMIDY3Ljg2OTI3IDQwLjA5MTM1MiAKTCA2Ny44NTk3MzIgNDAuMTAxNDczIApMIDU3LjczODYwOSAzMC41NjM2NTYgCkwgNTYuNTUxMTUxIDMxLjgyMzczNiAKegoiIGNsaXAtcGF0aD0idXJsKCNwOGIzYzUzNjEwYykiIHN0eWxlPSJzdHJva2U6ICMwMDAwMDA7IHN0cm9rZS13aWR0aDogMS41OyBzdHJva2UtbGluZWpvaW46IG1pdGVyIi8+CiAgIDwvZz4KICA8L2c+CiA8L2c+CiA8ZGVmcz4KICA8Y2xpcFBhdGggaWQ9InA4YjNjNTM2MTBjIj4KICAgPHJlY3QgeD0iMC43MiIgeT0iMC43MiIgd2lkdGg9IjEyNy4zNDgzODciIGhlaWdodD0iODYuNCIvPgogIDwvY2xpcFBhdGg+CiA8L2RlZnM+Cjwvc3ZnPgo="} x={45} y={6} width={w - 50} height={h - 12} />
      
      {/* legend built cleanly into top left corner */}
      <rect x={12} y={10} width={88} height={58} fill="#ffffff" stroke="#e5e0d6" strokeWidth={0.5} rx={4} opacity={0.9} />
      <circle cx={26} cy={23} r={4} fill={color} />
      <text x={35} y={27} fontFamily={FIG_FONTS.sans} fontSize={13} fill="#444">concept <tspan fontFamily={FIG_FONTS.math} fontStyle="italic">C</tspan></text>

      <circle cx={26} cy={38} r={4} fill="#fff" stroke={color} strokeWidth={1.4} />
      <text x={35} y={42} fontFamily={FIG_FONTS.sans} fontSize={13} fill="#444">not <tspan fontFamily={FIG_FONTS.math} fontStyle="italic">C</tspan></text>

      <line x1={21} y1={53} x2={31} y2={53} stroke="#000" strokeWidth={1.5} />
      <polygon points="31,53 29,51 34,53 29,55" fill="#000" />
      <text x={36} y={57} fontFamily={FIG_FONTS.math} fontStyle="italic" fontSize={14} fill="#000">v<tspan fontSize={10} dy={2}>C</tspan></text>
    </svg>
  );
}

// Token cells (a₁ a₂ … aₙ) — used in stage III/IV inputs.
function TokenStrip({ tint: bg = '#FBE9C8', border = '#D9A04A', label = 'a', lastLab = 'd', bracket = false, superscript = null }) {
  const Cell = ({ idx, lab }) => (
    <div style={{
      width: 28, height: 22, background: bg, border: `1px solid ${border}`, borderRadius: 3,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: FIG_FONTS.math, fontSize: 13, color: '#333', fontStyle: 'italic',
    }}>
      {label}<span style={{ fontSize: 9, fontStyle: 'normal', marginLeft: 1, marginTop: 4 }}>{lab}</span>
    </div>
  );
  return (
    <div style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
      {bracket && (
        <span style={{ fontFamily: FIG_FONTS.math, fontSize: 30, color: '#666', lineHeight: 1, marginRight: 2, alignSelf: 'center' }}>[</span>
      )}
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <Cell lab="1" />
        <Cell lab="2" />
        <span style={{ fontFamily: FIG_FONTS.sans, fontSize: 16, color: '#888', padding: '0 2px' }}>…</span>
        <Cell lab={lastLab} />
      </div>
      {bracket && (
        <span style={{ fontFamily: FIG_FONTS.math, fontSize: 30, color: '#666', lineHeight: 1, marginLeft: 2, alignSelf: 'center' }}>]</span>
      )}
      {superscript && <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 14, fontWeight: 500, color: '#444', marginLeft: 2, marginTop: -30, position: 'absolute', right: -10 }}>{superscript}</div>}
    </div>
  );
}

// Down arrow drawn as an inline svg so it's always centered.
function DownArrow({ length = 22, color = '#666' }) {
  return (
    <svg width={14} height={length} style={{ overflow: 'visible' }}>
      <line x1={7} y1={1} x2={7} y2={length - 6} stroke={color} strokeWidth={1.1} />
      <polygon points={`7,${length} 3,${length - 6} 11,${length - 6}`} fill={color} />
    </svg>
  );
}

// Right arrow with adjustable length.
function RightArrow({ length = 28, color = '#666' }) {
  return (
    <svg width={length} height={12} style={{ overflow: 'visible' }}>
      <line x1={0} y1={6} x2={length - 6} y2={6} stroke={color} strokeWidth={1.1} />
      <polygon points={`${length},6 ${length - 6},2 ${length - 6},10`} fill={color} />
    </svg>
  );
}

// ── Strip header — sharp colored top, rounded corners ────────────────────────
function StripHeader({ romanNumeral, title, color }) {
  return (
    <div style={{
      background: color, color: '#fff',
      padding: '10px 16px 12px',
      fontFamily: FIG_FONTS.sans,
      display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 4,
      borderRadius: '12px 12px 0 0',
      minHeight: 42,
    }}>
      <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: 0.8, opacity: 0.9, textTransform: 'uppercase', lineHeight: 1 }}>
        Stage {romanNumeral}
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: 0.4, lineHeight: 1.2, textTransform: 'uppercase' }}>
        {title}
      </div>
    </div>
  );
}


// Caption-style intro paragraph at top of strip
function StripIntro({ children }) {
  return (
    <div style={{
      fontFamily: FIG_FONTS.sans, fontSize: 14, lineHeight: 1.5, color: '#222',
      padding: '12px 14px 0',
    }}>
      {children}
    </div>
  );
}

// Section divider line within a strip

// Frozen-encoder architecture row: Tokenizer → [layers] → Classifier.
// The layer at highlightIdx is filled with the strip color; all others are neutral.
function TransformerRow({ color, highlightIdx = 3, layerCount = 5, highlightTokenizer = false, highlightClassifier = false }) {
  const vW = 252, rowH = 82;
  const bH = 48, bY = 22;
  const tokW = 60, clsW = 62, layW = 14, layGap = 3, arrW = 10;
  const allLayW = layerCount * layW + (layerCount - 1) * layGap;
  const totalW = tokW + 16 + allLayW + 16 + clsW;
  const lp = Math.max(0, (vW - totalW) / 2);
  const tokX = lp;
  const arr1X = tokX + tokW + 4;
  const layX = arr1X + arrW + 2;
  const arr2X = layX + allLayW + 4;
  const clsX = arr2X + arrW + 2;
  const midY = bY + bH / 2;

  const tokFill = highlightTokenizer ? color : "#F5F3EE";
  const tokStroke = highlightTokenizer ? color : "#C0BBB2";
  const tokStrokeW = highlightTokenizer ? 1.5 : 1;
  const tokTextC = highlightTokenizer ? "#fff" : "#555";

  const clsFill = highlightClassifier ? color : "#F5F3EE";
  const clsStroke = highlightClassifier ? color : "#C0BBB2";
  const clsStrokeW = highlightClassifier ? 1.5 : 1;
  const clsTextC = highlightClassifier ? "#fff" : "#555";

  return (
    <svg width="100%" viewBox={`0 0 ${vW} ${rowH}`} style={{ display: 'block', flexShrink: 0 }}>
      {/* Box around transformer layers */}
      <rect x={layX - 4} y={bY - 4} width={allLayW + 8} height={bH + 8} rx={4} fill="none" stroke="#A09988" strokeWidth={1} strokeDasharray="3 2" />
      <text x={layX + allLayW / 2} y={bY - 8} textAnchor="middle" fontFamily={FIG_FONTS.sans} fontSize={13} fill="#666" fontWeight="500">Transformer</text>

      <rect x={tokX} y={bY} width={tokW} height={bH} rx={4} fill={tokFill} stroke={tokStroke} strokeWidth={tokStrokeW}/>
      <text x={tokX+tokW/2} y={midY+1} textAnchor="middle" dominantBaseline="middle"
        fontFamily={FIG_FONTS.sans} fontSize={13} fill={tokTextC} fontWeight={highlightTokenizer ? "600" : "500"}>Tokenizer</text>
      <line x1={arr1X} y1={midY} x2={arr1X+arrW-3} y2={midY} stroke="#AAA" strokeWidth={1}/>
      <polygon points={`${arr1X+arrW},${midY} ${arr1X+arrW-4},${midY-2.5} ${arr1X+arrW-4},${midY+2.5}`} fill="#AAA"/>
      
      {Array.from({length: layerCount}, (_, i) => {
        const x = layX + i*(layW+layGap);
        if (i === Math.floor(layerCount / 2)) {
          return <text key={i} x={x + layW/2} y={midY-2} textAnchor="middle" dominantBaseline="middle" fontFamily={FIG_FONTS.sans} fontSize={16} fill="#888" letterSpacing={1}>…</text>;
        }
        const isHi = i === highlightIdx;
        return (
          <g key={i}>
            <rect x={x} y={bY} width={layW} height={bH} rx={2.5}
              fill={isHi ? color : '#EDEAE4'} stroke={isHi ? color : '#C8C4BA'}
              strokeWidth={isHi ? 1.5 : 0.8}/>
            {isHi && (
              <text x={x+layW/2} y={midY+1} textAnchor="middle" dominantBaseline="middle" fontFamily={FIG_FONTS.math} fontSize={14} fill="#fff" fontStyle="italic">ℓ</text>
            )}
          </g>
        );
      })}
      
      <line x1={arr2X} y1={midY} x2={arr2X+arrW-3} y2={midY} stroke="#AAA" strokeWidth={1}/>
      <polygon points={`${arr2X+arrW},${midY} ${arr2X+arrW-4},${midY-2.5} ${arr2X+arrW-4},${midY+2.5}`} fill="#AAA"/>
      <rect x={clsX} y={bY} width={clsW} height={bH} rx={4} fill={clsFill} stroke={clsStroke} strokeWidth={clsStrokeW}/>
      <text x={clsX+clsW/2} y={midY+1} textAnchor="middle" dominantBaseline="middle"
        fontFamily={FIG_FONTS.sans} fontSize={13} fill={clsTextC} fontWeight={highlightClassifier ? "600" : "500"}>Classifier</text>
    </svg>
  );
}

// ── Strip I — Spectral Decoder ────────────────────────────────────────────────
function StripI({ color }) {
  const lightTint = tint(color, 0.12);
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${tint(color, 0.35)}`, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <StripHeader romanNumeral="I" title="Spectral Decoder" color={color} />

      {/* ── Spectral Decoder ───────────────────────────────────────── */}
      <StripIntro>
        A small network learns to read the frozen encoder's internal representations as EEG spectra (amplitude and phase at each frequency) giving us a physiological vocabulary for everything the encoder has learned.
      </StripIntro>

      {/* Diagram — vertical flow: model overview ↓ decoder ↓ (amp, phase) */}
      <div style={{ padding: '14px 14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        {/* Frozen encoder architecture — tokenizer highlighted (embedding tap) */}
        <TransformerRow color={color} highlightIdx={-1} highlightTokenizer={true} />

        {/* token vector + arrow right-down from Tokenizer */}
        <div style={{ position: 'relative', width: 252, height: 26, margin: '2px 0 2px' }}>
          <svg width={252} height={26} style={{ position: 'absolute', inset: 0 }}>
            <line x1={38} y1={0} x2={38} y2={10} stroke={color} strokeWidth={1.3} />
            <line x1={38} y1={10} x2={126} y2={10} stroke={color} strokeWidth={1.3} />
            <line x1={126} y1={10} x2={126} y2={20} stroke={color} strokeWidth={1.3} />
            <polygon points="126,26 122,20 130,20" fill={color} />
          </svg>
          <div style={{ position: 'absolute', left: 132, top: 7, fontFamily: FIG_FONTS.math, fontSize: 12, color: '#444' }}>
            <Mi>t</Mi> ∈ ℝ<sup>d</sup>
          </div>
        </div>

        {/* Spectral Decoder: vertical trapezoid, wide top → narrow bottom */}
        <Trapezoid w={245} h={115} orient="shrinkV" fill={lightTint} stroke={color} color={color} label="Spectral Decoder" />

        {/* Split arrow ┬ */}
        <div style={{ position: 'relative', width: 200, height: 22, marginTop: 0 }}>
          <svg width={200} height={22} style={{ position: 'absolute', inset: 0 }}>
            {/* trunk down from decoder */}
            <line x1={100} y1={0} x2={100} y2={8} stroke="#666" strokeWidth={1} />
            {/* horizontal split */}
            <line x1={50} y1={8} x2={150} y2={8} stroke="#666" strokeWidth={1} />
            {/* down to amplitude */}
            <line x1={50} y1={8} x2={50} y2={18} stroke="#666" strokeWidth={1} />
            <polygon points="50,22 46,16 54,16" fill="#666" />
            {/* down to phase */}
            <line x1={150} y1={8} x2={150} y2={18} stroke="#666" strokeWidth={1} />
            <polygon points="150,22 146,16 154,16" fill="#666" />
          </svg>
        </div>

        {/* outputs row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', width: 240, marginTop: 2 }}>
          <div style={{ width: 106, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
            <div style={{ background: '#fff', border: `1px solid ${tint(color, 0.5)}`, borderRadius: 6, padding: '4px 6px' }}>
              <img src={"data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIKICAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgd2lkdGg9IjEwNy40NHB0IiBoZWlnaHQ9IjQyLjY0cHQiIHZpZXdCb3g9IjAgMCAxMDcuNDQgNDIuNjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmVyc2lvbj0iMS4xIj4KIDxtZXRhZGF0YT4KICA8cmRmOlJERiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgIDxjYzpXb3JrPgogICAgPGRjOnR5cGUgcmRmOnJlc291cmNlPSJodHRwOi8vcHVybC5vcmcvZGMvZGNtaXR5cGUvU3RpbGxJbWFnZSIvPgogICAgPGRjOmRhdGU+MjAyNi0wNC0yOFQxNjowOTo1Ny4wMjc5ODE8L2RjOmRhdGU+CiAgICA8ZGM6Zm9ybWF0PmltYWdlL3N2Zyt4bWw8L2RjOmZvcm1hdD4KICAgIDxkYzpjcmVhdG9yPgogICAgIDxjYzpBZ2VudD4KICAgICAgPGRjOnRpdGxlPk1hdHBsb3RsaWIgdjMuMTAuOSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy88L2RjOnRpdGxlPgogICAgIDwvY2M6QWdlbnQ+CiAgICA8L2RjOmNyZWF0b3I+CiAgIDwvY2M6V29yaz4KICA8L3JkZjpSREY+CiA8L21ldGFkYXRhPgogPGRlZnM+CiAgPHN0eWxlIHR5cGU9InRleHQvY3NzIj4qe3N0cm9rZS1saW5lam9pbjogcm91bmQ7IHN0cm9rZS1saW5lY2FwOiBidXR0fTwvc3R5bGU+CiA8L2RlZnM+CiA8ZyBpZD0iZmlndXJlXzEiPgogIDxnIGlkPSJwYXRjaF8xIj4KICAgPHBhdGggZD0iTSAwIDQyLjY0IApMIDEwNy40NCA0Mi42NCAKTCAxMDcuNDQgMCAKTCAwIDAgCkwgMCA0Mi42NCAKegoiIHN0eWxlPSJmaWxsOiBub25lIi8+CiAgPC9nPgogIDxnIGlkPSJheGVzXzEiPgogICA8ZyBpZD0icGF0Y2hfMiI+CiAgICA8cGF0aCBkPSJNIDAuNzIgNDEuOTIgCkwgMTA2LjcyIDQxLjkyIApMIDEwNi43MiAwLjcyIApMIDAuNzIgMC43MiAKTCAwLjcyIDQxLjkyIAp6CiIgc3R5bGU9ImZpbGw6IG5vbmUiLz4KICAgPC9nPgogICA8ZyBpZD0icGF0Y2hfMyI+CiAgICA8cGF0aCBkPSJNIDUuNTM4MTgyIDQxLjkyIApMIDIxLjU5ODc4OCA0MS45MiAKTCAyMS41OTg3ODggNC40NjU0NTUgCkwgNS41MzgxODIgNC40NjU0NTUgCnoKIiBjbGlwLXBhdGg9InVybCgjcDYxZTIxOTQyNmIpIiBzdHlsZT0iZmlsbDogIzJkNmZiNSIvPgogICA8L2c+CiAgIDxnIGlkPSJwYXRjaF80Ij4KICAgIDxwYXRoIGQ9Ik0gMjUuNjEzOTM5IDQxLjkyIApMIDQxLjY3NDU0NSA0MS45MiAKTCA0MS42NzQ1NDUgMjMuMTkyNzI3IApMIDI1LjYxMzkzOSAyMy4xOTI3MjcgCnoKIiBjbGlwLXBhdGg9InVybCgjcDYxZTIxOTQyNmIpIiBzdHlsZT0iZmlsbDogIzJkNmZiNSIvPgogICA8L2c+CiAgIDxnIGlkPSJwYXRjaF81Ij4KICAgIDxwYXRoIGQ9Ik0gNDUuNjg5Njk3IDQxLjkyIApMIDYxLjc1MDMwMyA0MS45MiAKTCA2MS43NTAzMDMgMjkuNDM1MTUyIApMIDQ1LjY4OTY5NyAyOS40MzUxNTIgCnoKIiBjbGlwLXBhdGg9InVybCgjcDYxZTIxOTQyNmIpIiBzdHlsZT0iZmlsbDogIzJkNmZiNSIvPgogICA8L2c+CiAgIDxnIGlkPSJwYXRjaF82Ij4KICAgIDxwYXRoIGQ9Ik0gNjUuNzY1NDU1IDQxLjkyIApMIDgxLjgyNjA2MSA0MS45MiAKTCA4MS44MjYwNjEgMzIuNTU2MzY0IApMIDY1Ljc2NTQ1NSAzMi41NTYzNjQgCnoKIiBjbGlwLXBhdGg9InVybCgjcDYxZTIxOTQyNmIpIiBzdHlsZT0iZmlsbDogIzJkNmZiNSIvPgogICA8L2c+CiAgIDxnIGlkPSJwYXRjaF83Ij4KICAgIDxwYXRoIGQ9Ik0gODUuODQxMjEyIDQxLjkyIApMIDEwMS45MDE4MTggNDEuOTIgCkwgMTAxLjkwMTgxOCAzNC40MjkwOTEgCkwgODUuODQxMjEyIDM0LjQyOTA5MSAKegoiIGNsaXAtcGF0aD0idXJsKCNwNjFlMjE5NDI2YikiIHN0eWxlPSJmaWxsOiAjMmQ2ZmI1Ii8+CiAgIDwvZz4KICAgPGcgaWQ9Im1hdHBsb3RsaWIuYXhpc18xIi8+CiAgIDxnIGlkPSJtYXRwbG90bGliLmF4aXNfMiIvPgogICA8ZyBpZD0icGF0Y2hfOCI+CiAgICA8cGF0aCBkPSJNIDAuNzIgNDEuOTIgCkwgMC43MiAwLjcyIAoiIHN0eWxlPSJmaWxsOiBub25lOyBzdHJva2U6ICM4ODg4ODg7IHN0cm9rZS1saW5lam9pbjogbWl0ZXI7IHN0cm9rZS1saW5lY2FwOiBzcXVhcmUiLz4KICAgPC9nPgogICA8ZyBpZD0icGF0Y2hfOSI+CiAgICA8cGF0aCBkPSJNIDAuNzIgNDEuOTIgCkwgMTA2LjcyIDQxLjkyIAoiIHN0eWxlPSJmaWxsOiBub25lOyBzdHJva2U6ICM4ODg4ODg7IHN0cm9rZS1saW5lam9pbjogbWl0ZXI7IHN0cm9rZS1saW5lY2FwOiBzcXVhcmUiLz4KICAgPC9nPgogIDwvZz4KIDwvZz4KIDxkZWZzPgogIDxjbGlwUGF0aCBpZD0icDYxZTIxOTQyNmIiPgogICA8cmVjdCB4PSIwLjcyIiB5PSIwLjcyIiB3aWR0aD0iMTA2IiBoZWlnaHQ9IjQxLjIiLz4KICA8L2NsaXBQYXRoPgogPC9kZWZzPgo8L3N2Zz4K"} style={{ width: 96, height: 38, display: 'block' }} alt="amplitude" />
            </div>
            <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: '#666' }}>amplitude</div>
            <div style={{ fontFamily: FIG_FONTS.math, fontSize: 15, color: '#222' }}>
              Â<sub style={{ fontSize: 10 }}>ν</sub>
            </div>
          </div>
          <div style={{ width: 118, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
            <div style={{ background: '#fff', border: `1px solid ${tint(color, 0.5)}`, borderRadius: 6, padding: '4px 6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img src={"data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIKICAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgd2lkdGg9IjU3LjZwdCIgaGVpZ2h0PSI1Ny42cHQiIHZpZXdCb3g9IjAgMCA1Ny42IDU3LjYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmVyc2lvbj0iMS4xIj4KIDxtZXRhZGF0YT4KICA8cmRmOlJERiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgIDxjYzpXb3JrPgogICAgPGRjOnR5cGUgcmRmOnJlc291cmNlPSJodHRwOi8vcHVybC5vcmcvZGMvZGNtaXR5cGUvU3RpbGxJbWFnZSIvPgogICAgPGRjOmRhdGU+MjAyNi0wNC0yOFQxNjowOTo1Ny4wNDA0NjE8L2RjOmRhdGU+CiAgICA8ZGM6Zm9ybWF0PmltYWdlL3N2Zyt4bWw8L2RjOmZvcm1hdD4KICAgIDxkYzpjcmVhdG9yPgogICAgIDxjYzpBZ2VudD4KICAgICAgPGRjOnRpdGxlPk1hdHBsb3RsaWIgdjMuMTAuOSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy88L2RjOnRpdGxlPgogICAgIDwvY2M6QWdlbnQ+CiAgICA8L2RjOmNyZWF0b3I+CiAgIDwvY2M6V29yaz4KICA8L3JkZjpSREY+CiA8L21ldGFkYXRhPgogPGRlZnM+CiAgPHN0eWxlIHR5cGU9InRleHQvY3NzIj4qe3N0cm9rZS1saW5lam9pbjogcm91bmQ7IHN0cm9rZS1saW5lY2FwOiBidXR0fTwvc3R5bGU+CiA8L2RlZnM+CiA8ZyBpZD0iZmlndXJlXzEiPgogIDxnIGlkPSJwYXRjaF8xIj4KICAgPHBhdGggZD0iTSAwIDU3LjYgCkwgNTcuNiA1Ny42IApMIDU3LjYgMCAKTCAwIDAgCkwgMCA1Ny42IAp6CiIgc3R5bGU9ImZpbGw6IG5vbmUiLz4KICA8L2c+CiAgPGcgaWQ9ImF4ZXNfMSI+CiAgIDxnIGlkPSJwYXRjaF8yIj4KICAgIDxwYXRoIGQ9Ik0gNDQuNzg4Nzk1IDEyLjgxMTIwNSAKTCA0My41MjY1MjEgMTYuNTk4MDI1IApMIDQyLjI3MjY2MyAxNS4zNDQxNjcgCkwgMjguODA4NDE1IDI4LjgwODQxNSAKTCAyOC43OTE1ODUgMjguNzkxNTg1IApMIDQyLjI1NTgzMyAxNS4zMjczMzcgCkwgNDEuMDAxOTc1IDE0LjA3MzQ3OSAKegoiIGNsaXAtcGF0aD0idXJsKCNwMTk0MTVjODc4NikiIHN0eWxlPSJmaWxsOiAjMmQ2ZmI1OyBzdHJva2U6ICMyZDZmYjU7IHN0cm9rZS13aWR0aDogMS41OyBzdHJva2UtbGluZWpvaW46IG1pdGVyIi8+CiAgIDwvZz4KICAgPGcgaWQ9ImxpbmUyZF8xIj4KICAgIDxwYXRoIGQ9Ik0gNTIuNjAxNjUzIDI4LjggCkwgNTIuNTUzNzMyIDI3LjI5MDQwNiAKTCA1Mi40MTAxNjQgMjUuNzg2ODkgCkwgNTIuMTcxNTI2IDI0LjI5NTUwOCAKTCA1MS44Mzg3NzkgMjIuODIyMjYzIApMIDUxLjQxMzI2MyAyMS4zNzMwODggCkwgNTAuODk2NjkxIDE5Ljk1MzgxOSAKTCA1MC4yOTExNDQgMTguNTcwMTcxIApMIDQ5LjU5OTA2IDE3LjIyNzcxNCAKTCA0OC44MjMyMjUgMTUuOTMxODU1IApMIDQ3Ljk2Njc2MyAxNC42ODc4MTEgCkwgNDcuMDMzMTI0IDEzLjUwMDU5MiAKTCA0Ni4wMjYwNjYgMTIuMzc0OTc5IApMIDQ0Ljk0OTY0NSAxMS4zMTU1MDMgCkwgNDMuODA4MTk2IDEwLjMyNjQzMSAKTCA0Mi42MDYzMTMgOS40MTE3NDYgCkwgNDEuMzQ4ODM4IDguNTc1MTMgCkwgNDAuMDQwODMyIDcuODE5OTUzIApMIDM4LjY4NzU2NCA3LjE0OTI1NSAKTCAzNy4yOTQ0ODIgNi41NjU3MzcgCkwgMzUuODY3MTk2IDYuMDcxNzQ4IApMIDM0LjQxMTQ1MiA1LjY2OTI3OCAKTCAzMi45MzMxMTQgNS4zNTk5NDggCkwgMzEuNDM4MTMyIDUuMTQ1MDAyIApMIDI5LjkzMjUyOCA1LjAyNTMwNiAKTCAyOC40MjIzNjQgNS4wMDEzNDMgCkwgMjYuOTEzNzIgNS4wNzMyMDkgCkwgMjUuNDEyNjcyIDUuMjQwNjE0IApMIDIzLjkyNTI2MyA1LjUwMjg4NCAKTCAyMi40NTc0ODMgNS44NTg5NjMgCkwgMjEuMDE1MjQyIDYuMzA3NDE5IApMIDE5LjYwNDM0NyA2Ljg0NjQ0MyAKTCAxOC4yMzA0ODEgNy40NzM4NjcgCkwgMTYuODk5MTc0IDguMTg3MTY0IApMIDE1LjYxNTc4NyA4Ljk4MzQ2MSAKTCAxNC4zODU0ODggOS44NTk1NTMgCkwgMTMuMjEzMjMyIDEwLjgxMTkxMSAKTCAxMi4xMDM3MzggMTEuODM2NzAxIApMIDExLjA2MTQ3NCAxMi45Mjk3OTYgCkwgMTAuMDkwNjM3IDE0LjA4Njc5NCAKTCA5LjE5NTEzNiAxNS4zMDMwMzggCkwgOC4zNzg1NzcgMTYuNTczNjI5IApMIDcuNjQ0MjQ3IDE3Ljg5MzQ1MSAKTCA2Ljk5NTEwNCAxOS4yNTcxOTEgCkwgNi40MzM3NjIgMjAuNjU5MzU1IApMIDUuOTYyNDgxIDIyLjA5NDI5OSAKTCA1LjU4MzE1OSAyMy41NTYyNDUgCkwgNS4yOTczMjIgMjUuMDM5MzA2IApMIDUuMTA2MTIzIDI2LjUzNzUwOSAKTCA1LjAxMDMzIDI4LjA0NDgyMyAKTCA1LjAxMDMzIDI5LjU1NTE3NyAKTCA1LjEwNjEyMyAzMS4wNjI0OTEgCkwgNS4yOTczMjIgMzIuNTYwNjk0IApMIDUuNTgzMTU5IDM0LjA0Mzc1NSAKTCA1Ljk2MjQ4MSAzNS41MDU3MDEgCkwgNi40MzM3NjIgMzYuOTQwNjQ1IApMIDYuOTk1MTA0IDM4LjM0MjgwOSAKTCA3LjY0NDI0NyAzOS43MDY1NDkgCkwgOC4zNzg1NzcgNDEuMDI2MzcxIApMIDkuMTk1MTM2IDQyLjI5Njk2MiAKTCAxMC4wOTA2MzcgNDMuNTEzMjA2IApMIDExLjA2MTQ3NCA0NC42NzAyMDQgCkwgMTIuMTAzNzM4IDQ1Ljc2MzI5OSAKTCAxMy4yMTMyMzIgNDYuNzg4MDg5IApMIDE0LjM4NTQ4OCA0Ny43NDA0NDcgCkwgMTUuNjE1Nzg3IDQ4LjYxNjUzOSAKTCAxNi44OTkxNzQgNDkuNDEyODM2IApMIDE4LjIzMDQ4MSA1MC4xMjYxMzMgCkwgMTkuNjA0MzQ3IDUwLjc1MzU1NyAKTCAyMS4wMTUyNDIgNTEuMjkyNTgxIApMIDIyLjQ1NzQ4MyA1MS43NDEwMzcgCkwgMjMuOTI1MjYzIDUyLjA5NzExNiAKTCAyNS40MTI2NzIgNTIuMzU5Mzg2IApMIDI2LjkxMzcyIDUyLjUyNjc5MSAKTCAyOC40MjIzNjQgNTIuNTk4NjU3IApMIDI5LjkzMjUyOCA1Mi41NzQ2OTQgCkwgMzEuNDM4MTMyIDUyLjQ1NDk5OCAKTCAzMi45MzMxMTQgNTIuMjQwMDUyIApMIDM0LjQxMTQ1MiA1MS45MzA3MjIgCkwgMzUuODY3MTk2IDUxLjUyODI1MiAKTCAzNy4yOTQ0ODIgNTEuMDM0MjYzIApMIDM4LjY4NzU2NCA1MC40NTA3NDUgCkwgNDAuMDQwODMyIDQ5Ljc4MDA0NyAKTCA0MS4zNDg4MzggNDkuMDI0ODcgCkwgNDIuNjA2MzEzIDQ4LjE4ODI1NCAKTCA0My44MDgxOTYgNDcuMjczNTY5IApMIDQ0Ljk0OTY0NSA0Ni4yODQ0OTcgCkwgNDYuMDI2MDY2IDQ1LjIyNTAyMSAKTCA0Ny4wMzMxMjQgNDQuMDk5NDA4IApMIDQ3Ljk2Njc2MyA0Mi45MTIxODkgCkwgNDguODIzMjI1IDQxLjY2ODE0NSAKTCA0OS41OTkwNiA0MC4zNzIyODYgCkwgNTAuMjkxMTQ0IDM5LjAyOTgyOSAKTCA1MC44OTY2OTEgMzcuNjQ2MTgxIApMIDUxLjQxMzI2MyAzNi4yMjY5MTIgCkwgNTEuODM4Nzc5IDM0Ljc3NzczNyAKTCA1Mi4xNzE1MjYgMzMuMzA0NDkyIApMIDUyLjQxMDE2NCAzMS44MTMxMSAKTCA1Mi41NTM3MzIgMzAuMzA5NTk0IApMIDUyLjYwMTY1MyAyOC44IAoiIGNsaXAtcGF0aD0idXJsKCNwMTk0MTVjODc4NikiIHN0eWxlPSJmaWxsOiBub25lOyBzdHJva2U6ICNiYmJiYmI7IHN0cm9rZS1saW5lY2FwOiBzcXVhcmUiLz4KICAgPC9nPgogICA8ZyBpZD0ibGluZTJkXzIiPgogICAgPHBhdGggZD0iTSAyLjYxODE4MiAyOC44IApMIDU0Ljk4MTgxOCAyOC44IAoiIGNsaXAtcGF0aD0idXJsKCNwMTk0MTVjODc4NikiIHN0eWxlPSJmaWxsOiBub25lOyBzdHJva2U6ICNkZGRkZGQ7IHN0cm9rZS1saW5lY2FwOiBzcXVhcmUiLz4KICAgPC9nPgogICA8ZyBpZD0ibGluZTJkXzMiPgogICAgPHBhdGggZD0iTSAyOC44IDU0Ljk4MTgxOCAKTCAyOC44IDIuNjE4MTgyIAoiIGNsaXAtcGF0aD0idXJsKCNwMTk0MTVjODc4NikiIHN0eWxlPSJmaWxsOiBub25lOyBzdHJva2U6ICNkZGRkZGQ7IHN0cm9rZS1saW5lY2FwOiBzcXVhcmUiLz4KICAgPC9nPgogICA8ZyBpZD0iUGF0aENvbGxlY3Rpb25fMSI+CiAgICA8ZGVmcz4KICAgICA8cGF0aCBpZD0ibThhNjQ1NGYxYjgiIGQ9Ik0gMCAxLjkzNjQ5MiAKQyAwLjUxMzU2NCAxLjkzNjQ5MiAxLjAwNjE2MiAxLjczMjQ1MSAxLjM2OTMwNiAxLjM2OTMwNiAKQyAxLjczMjQ1MSAxLjAwNjE2MiAxLjkzNjQ5MiAwLjUxMzU2NCAxLjkzNjQ5MiAwIApDIDEuOTM2NDkyIC0wLjUxMzU2NCAxLjczMjQ1MSAtMS4wMDYxNjIgMS4zNjkzMDYgLTEuMzY5MzA2IApDIDEuMDA2MTYyIC0xLjczMjQ1MSAwLjUxMzU2NCAtMS45MzY0OTIgMCAtMS45MzY0OTIgCkMgLTAuNTEzNTY0IC0xLjkzNjQ5MiAtMS4wMDYxNjIgLTEuNzMyNDUxIC0xLjM2OTMwNiAtMS4zNjkzMDYgCkMgLTEuNzMyNDUxIC0xLjAwNjE2MiAtMS45MzY0OTIgLTAuNTEzNTY0IC0xLjkzNjQ5MiAwIApDIC0xLjkzNjQ5MiAwLjUxMzU2NCAtMS43MzI0NTEgMS4wMDYxNjIgLTEuMzY5MzA2IDEuMzY5MzA2IApDIC0xLjAwNjE2MiAxLjczMjQ1MSAtMC41MTM1NjQgMS45MzY0OTIgMCAxLjkzNjQ5MiAKegoiIHN0eWxlPSJzdHJva2U6ICMyZDZmYjUiLz4KICAgIDwvZGVmcz4KICAgIDxnIGNsaXAtcGF0aD0idXJsKCNwMTk0MTVjODc4NikiPgogICAgIDx1c2UgeGxpbms6aHJlZj0iI204YTY0NTRmMWI4IiB4PSI0NC43ODg3OTUiIHk9IjEyLjgxMTIwNSIgc3R5bGU9ImZpbGw6ICMyZDZmYjU7IHN0cm9rZTogIzJkNmZiNSIvPgogICAgPC9nPgogICA8L2c+CiAgPC9nPgogPC9nPgogPGRlZnM+CiAgPGNsaXBQYXRoIGlkPSJwMTk0MTVjODc4NiI+CiAgIDxyZWN0IHg9IjAiIHk9Ii0wIiB3aWR0aD0iNTcuNiIgaGVpZ2h0PSI1Ny42Ii8+CiAgPC9jbGlwUGF0aD4KIDwvZGVmcz4KPC9zdmc+Cg=="} style={{ width: 48, height: 48, display: 'block' }} alt="phase" />
            </div>
            <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: '#666' }}>phase</div>
            <div style={{ fontFamily: FIG_FONTS.math, fontSize: 14, color: '#222', lineHeight: 1.1 }}>
              (cos&nbsp;φ̂<sub style={{ fontSize: 9 }}>ν</sub>,&nbsp;sin&nbsp;φ̂<sub style={{ fontSize: 9 }}>ν</sub>)
            </div>
          </div>
        </div>
      </div>

      {/* Bands box */}
      <div style={{ margin: '8px 14px 16px', padding: '12px 12px', background: lightTint, border: `1px solid ${color}`, borderRadius: 4, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {[
          { roman: 'i', title: 'Physiological vocabulary', body: 'Interpret any encoder direction as an EEG spectrum via the spectral decoder.' },
          { roman: 'ii', title: 'Steering decoder', body: 'Decode concept-steered SAE activations back to readable spectra in Stage IV.' },
        ].map(({ roman, title, body }) => (
          <div key={roman} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
            <div style={{
              width: 22, height: 22, borderRadius: '50%', border: `1.5px solid ${color}`,
              color, fontFamily: FIG_FONTS.sans, fontSize: 11, fontStyle: 'italic', fontWeight: 600,
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 1,
            }}>{roman}</div>
            <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#222', lineHeight: 1.5 }}>
              <span style={{ fontWeight: 700 }}>{title}.</span> {body}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Strip II — SAE Training ───────────────────────────────────────────────────
function StripII({ color }) {
  const lightTint = tint(color, 0.12);
  const midTint = tint(color, 0.24);
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${tint(color, 0.35)}`, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <StripHeader romanNumeral="II" title="Layer-wise SAE Training" color={color} />
      <StripIntro>
        For each transformer layer, we learn a large set of sparse, interpretable features that reconstruct the layer's activations. These features are the building blocks we interpret and manipulate in later stages.
      </StripIntro>

      <div style={{ padding: '14px 14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, flex: 1 }}>
        {/* Same frozen encoder architecture — SAE taps an intermediate layer, not the embedding */}
        <TransformerRow color={color} highlightIdx={4} />

        {/* arrow right-down from layer ℓ to centered TokenStrip */}
        <div style={{ position: 'relative', width: 252, height: 20, margin: '0 0' }}>
          <svg width={252} height={20} style={{ position: 'absolute', inset: 0 }}>
            {/* The layer ℓ is at index 4 (from 0..4).
                In TransformerRow:
                layX = tokX + tokW + 4 + arrW + 2
                tokX = Math.max(0, (252 - totalW) / 2) = (252 - (60+16+(5*14+4*3)+16+62))/2 = (252 - 236)/2 = 8
                layX = 8 + 60 + 4 + 10 + 2 = 84
                x of layer 4 center = 84 + 4*(14+3) + 14/2 = 84 + 68 + 7 = 159
                Target center = 126
            */}
            <line x1={159} y1={0} x2={159} y2={8} stroke={color} strokeWidth={1.3} />
            <line x1={159} y1={8} x2={126} y2={8} stroke={color} strokeWidth={1.3} />
            <line x1={126} y1={8} x2={126} y2={18} stroke={color} strokeWidth={1.3} />
            <polygon points="126,20 122,14 130,14" fill={color} />
          </svg>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <TokenStrip tint={lightTint} border={color} label="a" lastLab="d" bracket={true} superscript="T" />
          <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#666', textAlign: 'center', marginTop: 4 }}>
            layer <Mi>ℓ</Mi> activations (normalised)
          </div>
        </div>

        <DownArrow length={18} color="#666" />

        {/* TopK SAE container */}
        <div style={{
          width: '100%', border: `1.5px solid ${color}`, borderRadius: 6,
          background: lightTint, padding: '12px 8px 14px',
          display: 'flex', flexDirection: 'column', alignItems: 'center',
        }}>
          <div style={{ fontFamily: FIG_FONTS.sans, fontWeight: 700, fontSize: 14, color, textAlign: 'center', marginBottom: 10 }}>
            TopK Sparse Autoencoder
          </div>
          <Trapezoid w={234} h={74} orient="expandV" fill="#fff" stroke={color} color={color} label="Encoder" sub={<M>W<sub>enc</sub></M>} />
          
          <DownArrow length={16} color={color} />
          
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontFamily: FIG_FONTS.math, fontSize: 15, fontStyle: 'italic', fontWeight: 600, color }}>z</span>
              <div style={{ display: 'flex', gap: 2 }}>
                {Array.from({length: 15}).map((_, i) => (
                  <div key={i} style={{
                    width: 12, height: 24,
                    background: (i === 3 || i === 10 || i === 14) ? color : '#fff',
                    border: `1px solid ${color}`, borderRadius: 1
                  }} />
                ))}
              </div>
            </div>
          </div>
          
          <DownArrow length={16} color={color} />
          
          <Trapezoid w={234} h={74} orient="shrinkV" fill="#fff" stroke={color} color={color} label="Decoder" sub={<M>W<sub>dec</sub></M>} />
        </div>

        {/* outputs row from SAE */}
        <DownArrow length={18} color="#666" />
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <TokenStrip tint={lightTint} border={color} label="â" lastLab="d" bracket={true} superscript="T" />
          <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#222', textAlign: 'center', marginTop: 4, lineHeight: 1.25 }}>
            Reconstruction <Mi>â</Mi>
          </div>
        </div>
      </div>

    </div>
  );
}

// Concept Box icon component
function ConceptBox({ icon, label, color, isTarget }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      flex: 1, gap: 5, background: '#fff', border: `1px solid ${tint(color, 0.4)}`,
      padding: '6px 2px 5px', borderRadius: 6,
    }}>
      {icon}
      <div style={{
        fontSize: 11, fontFamily: FIG_FONTS.sans, color: isTarget ? '#111' : '#444',
        textAlign: 'center', lineHeight: 1.1, fontWeight: isTarget ? 700 : 400
      }}>
        {label}
      </div>
    </div>
  );
}

// ── Strip III — TCAV ──────────────────────────────────────────────────────────
function StripIII({ color }) {
  const lightTint = tint(color, 0.14);
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${tint(color, 0.35)}`, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <StripHeader romanNumeral="III" title="Concept Attribution" color={color} />
      <StripIntro>
        We ask which sparse features from Stage II respond to known clinical concepts (abnormality, age, sex, medication) identifying the subset of features that encode each concept.
      </StripIntro>

      <div style={{ padding: '14px 14px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, flex: 1 }}>
        {/* Connection box from Stages I & II */}
        <div style={{ 
          width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', 
          padding: '6px 4px 8px', background: '#FAFAF7', border: `1px dashed ${tint(color, 0.2)}`, borderRadius: 8 
        }}>
          <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#555', marginBottom: 6 }}>
            Sparse latent features (Train Split)
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontFamily: FIG_FONTS.math, fontSize: 15, fontStyle: 'italic', fontWeight: 600, color }}>z</span>
            <div style={{ display: 'flex', gap: 2 }}>
              {Array.from({length: 15}).map((_, i) => (
                <div key={i} style={{
                  width: 12, height: 24,
                  background: (i === 3 || i === 10 || i === 14) ? color : '#fff',
                  border: `1px solid ${color}`, borderRadius: 1
                }} />
              ))}
            </div>
          </div>
        </div>
        
        <DownArrow length={16} />

        {/* For each concept C */}
        <div style={{
          width: '100%', background: lightTint, border: `1.5px solid ${color}`, borderRadius: 8,
          padding: '10px 8px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8
        }}>
          <div style={{ fontFamily: FIG_FONTS.sans, fontWeight: 600, fontSize: 14, color: '#222' }}>
            For each concept&nbsp;<Mi>C</Mi>
          </div>
          <div style={{ display: 'flex', gap: 5, width: '100%' }}>
            <ConceptBox color={color} label="abnormality" icon={
              <svg width={44} height={44} viewBox="0 0 20 20">
                <path d="M 8.5 4 h 3 v 4.5 h 4.5 v 3 h -4.5 v 4.5 h -3 v -4.5 h -4.5 v -3 h 4.5 z" fill={color} />
              </svg>
            } />
            <ConceptBox color={color} label="age" icon={
              <svg width={44} height={44} viewBox="0 0 20 20">
                <circle cx={10} cy={10} r={6.5} fill="none" stroke={color} strokeWidth={1.5} />
                <polyline points="10,5 10,10 13.5,12.5" fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            } />
            <ConceptBox color={color} label="sex" icon={
              <svg width={44} height={44} viewBox="0 0 20 20">
                {/* Female */}
                <circle cx={6.5} cy={8} r={2.5} fill="none" stroke={color} strokeWidth={1.2} />
                <path d="M 6.5 10.5 L 6.5 15 M 4.5 13 L 8.5 13" fill="none" stroke={color} strokeWidth={1.2} />
                {/* Male */}
                <circle cx={13.5} cy={10} r={2.5} fill="none" stroke={color} strokeWidth={1.2} />
                <path d="M 15.3 8.2 L 17.5 6 M 15 6 L 17.5 6 L 17.5 8.5" fill="none" stroke={color} strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            } />
            <ConceptBox color={color} label="medication" icon={
              <svg width={44} height={44} viewBox="0 0 20 20">
                <g transform="rotate(-40 10 10)">
                  <rect x={4} y={7} width={12} height={6} rx={3} fill="none" stroke={color} strokeWidth={1.5} />
                  <line x1={10} y1={7} x2={10} y2={13} stroke={color} strokeWidth={1.5} />
                </g>
              </svg>
            } />
          </div>
        </div>

        <DownArrow length={16} />

        {/* K=10 text outside any box */}
        <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#333', textAlign: 'center', marginBottom: 4, marginTop: 4 }}>
          <span style={{ fontWeight: 600 }}>
            <span style={{ fontFamily: FIG_FONTS.math, fontStyle: 'italic' }}>K</span> = 10-fold Logistic Regression
          </span>
          <br />
        </div>

        {/* CAV mini-plot combining the LogReg and the attribution plot */}
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <MiniCAVPlot w={260} h={140} color={color} />
          <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#222', marginTop: 4, textAlign: 'center' }}>
            CAV direction&nbsp;<M>v<sub>C</sub></M>
          </div>
        </div>

        <DownArrow length={16} />

        {/* Significant features list */}
        <div style={{
          width: '100%', background: '#fff', border: `1.5px solid ${color}`, borderRadius: 6,
          padding: '10px 8px', display: 'flex', flexDirection: 'column', alignItems: 'center'
        }}>
          <div style={{ fontFamily: FIG_FONTS.sans, fontWeight: 700, fontSize: 13, color: color, marginBottom: 6 }}>
            Concept-Enriched Features
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span style={{ fontFamily: FIG_FONTS.math, fontSize: 14, fontStyle: 'italic', fontWeight: 600, color }}>ℱ<sub style={{ fontSize: 9 }}>C</sub></span>
            <span style={{ fontFamily: FIG_FONTS.math, fontSize: 14, color: '#222' }}>=</span>
            <div style={{ display: 'flex', gap: 2 }}>
              {Array.from({length: 15}).map((_, i) => (
                <div key={i} style={{
                  width: 11, height: 24,
                  background: (i === 3 || i === 10 || i === 14) ? color : '#fafaf7',
                  border: `1px solid ${color}`, borderRadius: 2,
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  {(i === 3 || i === 10 || i === 14) && <span style={{ color: '#fff', fontSize: 15, lineHeight: 1 }}>*</span>}
                </div>
              ))}
            </div>
          </div>
          <div style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: '#555' }}>
             SAE indices (q &lt; 0.05)
          </div>
        </div>

      </div>

    </div>
  );
}

// ── Strip IV — Concept Steering ───────────────────────────────────────────────
function StripIV({ color, saeColor }) {
  const lightTint = tint(color, 0.10);
  const NumDot = ({ n }) => (
    <div style={{
      width: 18, height: 18, borderRadius: '50%', background: color, color: '#fff',
      fontFamily: FIG_FONTS.sans, fontWeight: 700, fontSize: 13,
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
    }}>{n}</div>
  );
  const Step = ({ n, children }) => (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 10 }}>
      <NumDot n={n} />
      <div style={{ flex: 1, fontFamily: FIG_FONTS.sans, fontSize: 13, color: '#222', lineHeight: 1.5, paddingTop: 1, minWidth: 0 }}>
        {children}
      </div>
    </div>
  );
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${tint(color, 0.35)}`, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <StripHeader romanNumeral="IV" title="Concept Steering" color={color} />
      <StripIntro>
        To test whether the identified features truly encode a concept, we swap them for the target group's typical values and decode the result back to EEG spectra, producing a readable, mechanistic explanation of the intervention.
      </StripIntro>

      <div style={{ padding: '14px 8px 8px', flex: 1 }}>
        <Step n={1}>
          Rank enriched features by CAV alignment:
          <div style={{
            fontFamily: FIG_FONTS.math, fontSize: 13,
            background: '#FAFAF7', border: '1px solid #E5E0D6', borderRadius: 4,
            padding: '5px 8px', marginTop: 4, textAlign: 'center', color: '#222',
          }}>
            <M>rank</M><sub style={{ fontSize: 10 }}>C</sub>(<Mi>i</Mi>) = |<span style={{ fontFamily: FIG_FONTS.math, fontStyle: 'italic', fontWeight: 700 }}>v</span><sub style={{ fontSize: 10 }}>C</sub> · <span style={{ fontFamily: FIG_FONTS.math, fontStyle: 'italic', fontWeight: 700 }}>w</span><sub style={{ fontSize: 10 }}>i</sub>|
          </div>
        </Step>

        {/* Source vs target context */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, margin: '0 0 8px' }}>
          <div style={{ textAlign: 'center', fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#222', background: '#FAFAF7', border: '1px dashed #E5E0D6', borderRadius: 4, padding: '5px 4px' }}>
            <b>Source</b>
            <div style={{ fontFamily: FIG_FONTS.math, fontSize: 13, color: '#222', marginTop: 1 }}><Mi>X</Mi><sub style={{ fontSize: 9 }}>source</sub></div>
            <div style={{ color: '#666', fontSize: 11, marginTop: 1 }}>Example: Abnormal EEG</div>
          </div>
          <div style={{ textAlign: 'center', fontFamily: FIG_FONTS.sans, fontSize: 12, color: '#222', background: '#FAFAF7', border: '1px dashed #E5E0D6', borderRadius: 4, padding: '5px 4px' }}>
            <b>Target pool</b>
            <div style={{ fontFamily: FIG_FONTS.math, fontSize: 13, color: '#222', marginTop: 1 }}><Mi>X</Mi><sub style={{ fontSize: 9 }}>target</sub></div>
            <div style={{ color: '#666', fontSize: 11, marginTop: 1 }}>Example: Normal EEG</div>
          </div>
        </div>

        {/* Latent z bars for source and target */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, margin: '0 0 10px' }}>
          {[
            { label: 'source', active: [2, 7, 12], c: color },
            { label: 'target pool', active: [3, 10, 14], c: color },
          ].map(({ label, active, c }) => (
            <div key={label} style={{ padding: '5px 6px 6px', background: '#FAFAF7', border: '1px dashed #E5E0D6', borderRadius: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, boxSizing: 'border-box' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontFamily: FIG_FONTS.math, fontSize: 14, fontStyle: 'italic', fontWeight: 600, color: c }}>z</span>
                <span style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: '#666' }}>({label})</span>
              </div>
              <div style={{ display: 'flex', gap: 2 }}>
                {Array.from({length: 15}).map((_, i) => (
                  <div key={i} style={{ width: 13, height: 24, background: active.includes(i) ? c : '#fff', border: `1px solid ${c}`, borderRadius: 1 }} />
                ))}
              </div>
            </div>
          ))}
        </div>

        <Step n={2}>
          Target-concept centroid:
          <div style={{
            fontFamily: FIG_FONTS.math, fontSize: 13,
            background: '#FAFAF7', border: '1px solid #E5E0D6', borderRadius: 4,
            padding: '6px 8px', marginTop: 4, color: '#222',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3,
          }}>
            <span><span style={{ fontStyle: 'italic', fontWeight: 700 }}>c</span><sub style={{ fontSize: 9 }}>target</sub></span>
            <span style={{ margin: '0 3px' }}>=</span>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontSize: 12, borderBottom: '1px solid #555', paddingBottom: 1, lineHeight: 1.3, alignSelf: 'stretch', textAlign: 'center' }}>1</span>
              <span style={{ fontSize: 12, paddingTop: 1, lineHeight: 1.3 }}>|<Mi>X</Mi><sub style={{ fontSize: 9 }}>target</sub>|</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '0 1px' }}>
              <span style={{ fontSize: 18, lineHeight: 0.9 }}>∑</span>
              <span style={{ fontSize: 10, color: '#555', lineHeight: 1.3, whiteSpace: 'nowrap', fontFamily: FIG_FONTS.math }}>
                <Mi>x</Mi><span style={{ fontStyle: 'normal' }}> ∈ </span><Mi>X</Mi><span><sub style={{ fontSize: 8 }}>target</sub></span>
              </span>
            </div>
            <span style={{ marginLeft: 1 }}><span style={{ fontStyle: 'italic', fontWeight: 700 }}>z</span>(<Mi>x</Mi>)</span>
          </div>
        </Step>

        <Step n={3}>Clamping sweep <Mi>f</Mi> ∈ [0, 1]: set top <Mi>n</Mi> source features to <Mi>c</Mi><sub style={{ fontSize: 9 }}>target</sub>:</Step>
        <div style={{ padding: '5px 6px 6px', background: '#FAFAF7', border: '1px dashed #E5E0D6', borderRadius: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, margin: '0 0 4px', boxSizing: 'border-box' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ fontFamily: FIG_FONTS.math, fontSize: 14, fontStyle: 'italic', fontWeight: 600, color }}>z<sup style={{ fontSize: 10, fontStyle: 'normal' }}>*</sup></span>
            <span style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: '#666' }}>(<Mi>f</Mi>)</span>
          </div>
          <div style={{ display: 'flex', gap: 2 }}>
            {Array.from({length: 15}).map((_, i) => {
              const isClamped = i === 2 || i === 7 || i === 12;
              return (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                  <span style={{ fontSize: 12, lineHeight: 1, color: color, visibility: isClamped ? 'visible' : 'hidden' }}>↑</span>
                  <div style={{ width: 13, height: 24, background: isClamped ? color : '#fff', border: `1px solid ${isClamped ? color : saeColor}`, borderRadius: 1 }} />
                </div>
              );
            })}
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 10 }}>
          <span style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: color, display: 'flex', alignItems: 'center', gap: 3 }}>
            <span style={{ display: 'inline-block', width: 7, height: 7, background: color, borderRadius: 1 }} />↑ set to <span><Mi>c</Mi><sub style={{ fontSize: 9 }}>target</sub></span>
          </span>
        </div>

        <Step n={4}>
          Decode → spectral output:
          <div style={{ margin: '8px 0 2px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
              <TokenStrip tint={tint(saeColor, 0.15)} border={saeColor} label="â" lastLab="d" bracket={false} />
              <RightArrow length={22} color="#aaa" />
              <div style={{ background: '#fff', border: `1px solid ${tint(color, 0.5)}`, borderRadius: 6, padding: '4px 6px' }}>
                <MiniSpectrum w={66} h={26} color={color} />
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'center', gap: 6, marginTop: 3 }}>
              <div style={{ width: 110, textAlign: 'center', fontFamily: FIG_FONTS.math, fontSize: 11, fontStyle: 'italic', color: '#555' }}>â*(<Mi>f</Mi>)</div>
              <div style={{ width: 22, flexShrink: 0 }} />
              <div style={{ width: 78, textAlign: 'center', fontFamily: FIG_FONTS.math, fontSize: 11, fontStyle: 'italic', color: '#555' }}>Â*(<Mi>f</Mi>)</div>
            </div>
          </div>
        </Step>
      </div>
    </div>
  );
}

// SAE feature row used in stage IV — wider cells so labels read.
const ZRow = ({ color, highlights = [], steerColor, steered }) => {
  return (
    <div style={{ display: 'flex', gap: 2 }}>
      <span style={{ fontFamily: FIG_FONTS.math, fontSize: 13, color: '#666', marginRight: 3 }}>z =</span>
      <div style={{
        width: 22, height: 22, background: '#fff', border: `1px solid #bbb`, borderRadius: 3,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: FIG_FONTS.math, fontSize: 10, color: '#444',
      }}>
        1
      </div>
      <div style={{
        width: 22, height: 22, background: '#fff', border: `1px solid #bbb`, borderRadius: 3,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: FIG_FONTS.math, fontSize: 10, color: '#444',
      }}>
        i
      </div>
      <span style={{ fontFamily: FIG_FONTS.sans, fontSize: 11, color: '#888' }}>…</span>
      <div style={{
        width: 22, height: 22, background: '#fff', border: `1px solid #bbb`, borderRadius: 3,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: FIG_FONTS.math, fontSize: 10, color: '#444',
      }}>
        j
      </div>
      <div style={{
        width: 22, height: 22, background: '#fff', border: `1px solid #bbb`, borderRadius: 3,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: FIG_FONTS.math, fontSize: 10, color: '#444',
      }}>
        n
      </div>
    </div>
  );
}

// ── Top-level layout ──────────────────────────────────────────────────────────
function FigureClean({ colors }) {
  const W = COL_W * 4 + 14 * 3 + 32; // 4 cols + 3 gaps + outer padding
  return (
    <div id="figure-columns" style={{ width: W, background: '#fff', fontFamily: FIG_FONTS.sans, color: '#111', borderRadius: 14 }}>
<div style={{
        display: 'grid', gridTemplateColumns: `repeat(4, ${COL_W}px)`,
        gap: 14,
        padding: '18px 16px 22px',
      }}>
        <StripI color={colors.spectral} />
        <StripII color={colors.sae} />
        <StripIII color={colors.tcav} />
        <StripIV color={colors.steer} spectralColor={colors.spectral} saeColor={colors.sae} />
      </div>
    </div>
  );
}

Object.assign(window, { FigureClean });
