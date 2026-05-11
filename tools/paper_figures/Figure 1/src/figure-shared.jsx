// Shared primitives for all 3 figure variations.
// Pure SVG/HTML, no faux data plots. Schematic only.

const FIG_FONTS = {
  serif: '"Source Serif 4", "Source Serif Pro", "Crimson Pro", Georgia, serif',
  sans:  '"Inter Tight", "Inter", system-ui, sans-serif',
  mono:  '"JetBrains Mono", "IBM Plex Mono", ui-monospace, monospace',
  math:  '"Latin Modern Math", "STIX Two Math", "Cambria Math", "Source Serif 4", serif',
};

// Default 4-stage palette. paper-friendly, muted, distinct in greyscale.
const DEFAULT_STAGE_COLORS = {
  spectral: '#B8893A', // ochre
  sae:      '#2F7A78', // teal
  tcav:     '#6B4FB8', // violet
  steer:    '#B23A48', // crimson
};

// Soft tinted backgrounds for stage cards
function tint(hex, alpha = 0.08) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Schematic primitives ─────────────────────────────────────────────

// Stylized EEG trace (multi-channel, vector). NOT a faux data plot —
// just a visual signifier for "EEG input".
function EEGTrace({ width = 120, height = 56, channels = 4, color = '#1a1a1a', strokeWidth = 0.9 }) {
  const lines = [];
  const chH = height / channels;
  // deterministic pseudo-random based on channel index
  const seed = (n) => {
    let x = Math.sin(n * 9301 + 49297) * 233280;
    return x - Math.floor(x);
  };
  for (let c = 0; c < channels; c++) {
    const cy = chH * (c + 0.5);
    const pts = [];
    const N = 60;
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const x = t * width;
      // sum of sines + a little noise
      const a = c * 7 + 1;
      const y = cy
        + Math.sin(t * 18 + a) * (chH * 0.18)
        + Math.sin(t * 41 + a * 1.3) * (chH * 0.10)
        + (seed(c * 100 + i) - 0.5) * (chH * 0.18);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    lines.push(<polyline key={c} points={pts.join(' ')} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" opacity={0.85} />);
  }
  return <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>{lines}</svg>;
}

// Token grid — small squares representing tokens t ∈ R^d. Schematic.
function TokenGrid({ rows = 3, cols = 8, size = 8, gap = 2, color = '#333', highlight = -1, highlightColor = '#B23A48' }) {
  const w = cols * size + (cols - 1) * gap;
  const h = rows * size + (rows - 1) * gap;
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const x = c * (size + gap);
      const y = r * (size + gap);
      const on = (i * 7 + 3) % 5 < 2; // sparse-ish
      const isHi = i === highlight;
      cells.push(
        <rect key={i} x={x} y={y} width={size} height={size}
          fill={isHi ? highlightColor : (on ? color : 'transparent')}
          stroke={isHi ? highlightColor : color}
          strokeWidth={0.7}
          opacity={isHi ? 1 : (on ? 0.85 : 0.35)} />
      );
    }
  }
  return <svg width={w} height={h}>{cells}</svg>;
}

// SAE feature column — tall thin column of activations, very sparse (TopK).
function SAEColumn({ height = 110, width = 14, k = 3, n = 22, color = '#2F7A78' }) {
  const cellH = height / n;
  // deterministic "active" indices
  const active = new Set();
  const seed = [3, 9, 14, 7, 18, 1, 11];
  for (let i = 0; i < k; i++) active.add(seed[i] % n);
  const cells = [];
  for (let i = 0; i < n; i++) {
    const y = i * cellH;
    const isOn = active.has(i);
    cells.push(
      <rect key={i} x={0} y={y + 0.5} width={width} height={cellH - 1}
        fill={isOn ? color : 'transparent'}
        stroke={color}
        strokeWidth={0.6}
        opacity={isOn ? 1 : 0.28} />
    );
  }
  return <svg width={width} height={height}>{cells}</svg>;
}

// Spectrum curve — schematic, NOT real data. Just a visual cue for "amplitude vs freq".
function SpectrumCurve({ width = 120, height = 50, color = '#B8893A', strokeWidth = 1.4 }) {
  const N = 80;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    // 1/f-ish shape with a small alpha bump
    const f = t * 60 + 0.5;
    const a = (1 / Math.pow(f, 0.9)) * 30 + Math.exp(-Math.pow((f - 10) / 3, 2)) * 0.45;
    const y = height - a * height * 0.9 - 4;
    const x = t * width;
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" />
      <line x1={0} y1={height - 1} x2={width} y2={height - 1} stroke="#999" strokeWidth={0.5} />
    </svg>
  );
}

// Arrow — single right-arrow with optional label above.
function Arrow({ length = 60, color = '#888', strokeWidth = 1.2, label, dashed = false, vertical = false }) {
  const sw = strokeWidth;
  if (vertical) {
    return (
      <svg width={20} height={length} viewBox={`0 0 20 ${length}`} style={{ overflow: 'visible' }}>
        <line x1={10} y1={0} x2={10} y2={length - 7} stroke={color} strokeWidth={sw} strokeDasharray={dashed ? '3 3' : 'none'} />
        <polygon points={`10,${length} 5,${length - 7} 15,${length - 7}`} fill={color} />
        {label && <text x={14} y={length / 2} fontSize={10} fill="#555" fontFamily={FIG_FONTS.mono} dominantBaseline="middle">{label}</text>}
      </svg>
    );
  }
  return (
    <svg width={length} height={20} viewBox={`0 0 ${length} 20`} style={{ overflow: 'visible' }}>
      <line x1={0} y1={10} x2={length - 7} y2={10} stroke={color} strokeWidth={sw} strokeDasharray={dashed ? '3 3' : 'none'} />
      <polygon points={`${length},10 ${length - 7},5 ${length - 7},15`} fill={color} />
      {label && <text x={length / 2} y={4} fontSize={10} fill="#555" fontFamily={FIG_FONTS.mono} textAnchor="middle">{label}</text>}
    </svg>
  );
}

// Stage label — tiny eyebrow + title
function StageHeader({ stage, title, color, math }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{
        fontFamily: FIG_FONTS.mono, fontSize: 9.5, color, letterSpacing: 0.6,
        textTransform: 'uppercase', fontWeight: 600,
      }}>
        {stage}
      </div>
      <div style={{
        fontFamily: FIG_FONTS.serif, fontSize: 15, color: '#111', fontWeight: 600, letterSpacing: -0.1,
      }}>
        {title}
      </div>
      {math && (
        <div style={{ fontFamily: FIG_FONTS.math, fontStyle: 'italic', fontSize: 12, color: '#444', marginTop: 4 }}>
          {math}
        </div>
      )}
    </div>
  );
}

// Caption / body text under a stage
function StageBody({ children, width }) {
  return (
    <div style={{
      fontFamily: FIG_FONTS.serif, fontSize: 11.5, lineHeight: 1.4, color: '#333',
      width, textWrap: 'pretty',
    }}>
      {children}
    </div>
  );
}

// Stage card — colored top accent bar + content
function StageCard({ color, children, width, height, accentSide = 'top', style = {} }) {
  const accentStyle = accentSide === 'top'
    ? { borderTop: `2px solid ${color}` }
    : accentSide === 'left'
    ? { borderLeft: `2px solid ${color}` }
    : {};
  return (
    <div style={{
      width, minHeight: height, padding: '14px 16px 16px',
      background: tint(color, 0.05),
      ...accentStyle,
      display: 'flex', flexDirection: 'column', gap: 10,
      ...style,
    }}>
      {children}
    </div>
  );
}

// Figure caption (bottom of every variation)
function FigCaption({ label, children, width }) {
  return (
    <div style={{ width, marginTop: 18, fontFamily: FIG_FONTS.serif, fontSize: 11, lineHeight: 1.5, color: '#333' }}>
      <span style={{ fontWeight: 700 }}>{label}</span>{' '}
      {children}
    </div>
  );
}

Object.assign(window, {
  FIG_FONTS, DEFAULT_STAGE_COLORS, tint,
  EEGTrace, TokenGrid, SAEColumn, SpectrumCurve, Arrow,
  StageHeader, StageBody, StageCard, FigCaption,
});
