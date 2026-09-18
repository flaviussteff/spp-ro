# UI — Style Reference & Design System Specification
> Clinical blueprint on frosted paper: Academic rigor meets developer-infrastructure precision

**Theme:** Light  
**Project:** SPP-Ro (Constitutional Pre-training for Romanian Generative LLMs from Token Zero)  
**Aesthetic:** Engineered clinical blueprint, frosted translucent paper surfaces, hairline borders, geometric Geist typography, and surgical Ember accents.

---

## 1. Design Concept & Philosophy

SPP-Ro is a research project exploring token-zero constitutional pre-training for Romanian language models. The visual identity avoids noisy neon gradients, dark gamer tropes, and generic corporate templates. Instead, it embodies a **clinical blueprint on frosted paper**:

1. **Achromatic Blueprint Core**: Pure paper-white surfaces (`#ffffff`) floating over an architectural canvas (`#f5f5f5`) textured with a faint 24px micro-grid watermark.
2. **Frosted Paper Elevation**: Cards and sticky navigational bars utilize subtle glassmorphism (`backdrop-filter: blur(16px)`) with crisp 1px hairline borders (`#e5e5e5`) and whisper-quiet elevation shadows.
3. **Pill & Container Geometry**: Interactive elements (buttons, filter chips, search fields, status badges) adhere strictly to an **18px pill radius**, while primary content surfaces and Bento containers use a **24px container radius**.
4. **Geist Geometric Typography**: Heavy reliance on Geist / Inter with aggressive negative letter-spacing (`-0.050em` on display headings) and tabular figures (`tabular-nums`) for benchmark metrics.
5. **Surgical Ember Accent (`#e7000b`)**: A single warm scarlet accent reserved for critical delta disparities ($\Delta SPM$), live probe alerts, active indicator pips, and attention-blocking tags.

---

## 2. Design Tokens — Color Palette

| Token Name | Hex Value | CSS Variable | Semantic Role |
|------------|-----------|--------------|---------------|
| **Canvas** | `#f5f5f5` | `--color-canvas` | Main viewport background, micro-grid backdrop, secondary button fills |
| **Paper** | `#ffffff` | `--color-paper` | Primary card surfaces, modal dialogs, elevated containers |
| **Surface Alt** | `#fafafa` | `--color-surface-alt` | Sidebar panels, subtle inner card wells, input resting state |
| **Frosted Paper** | `rgba(255, 255, 255, 0.88)` | `--color-frosted` | Sticky navigation bar, floating action overlays |
| **Ink** | `#0a0a0a` | `--color-ink` | Primary text, display headings, high-emphasis button fills |
| **Ink Soft** | `#171717` | `--color-ink-soft` | Secondary headings, active tab backgrounds, dark badge fills |
| **Mid Gray** | `#737373` | `--color-mid-gray` | Muted body text, helper labels, probe context descriptions |
| **Hairline** | `#e5e5e5` | `--color-hairline` | 1px border lines, card outlines, table row dividers, grid guides |
| **Hairline Subtle** | `#f0f0f0` | `--color-hairline-subtle` | Faint inner separators, secondary division lines |
| **Ember** | `#e7000b` | `--color-ember` | Disparity delta indicator ($\Delta SPM$), live probe alerts, active badges |
| **Ember Soft** | `#fef2f2` | `--color-ember-soft` | Soft tinted badge backgrounds for bias metrics and warnings |
| **Terminal Slate** | `#0f172a` | `--color-terminal` | High-contrast code snippet containers, BibTeX citation card |

---

## 3. Design Tokens — Typography & Scale

The system standardizes on **Geist** (with **Inter** as first fallback) for interface and display type, paired with **Geist Mono** / **JetBrains Mono** for token embeddings, attention matrices, and code.

### Font Families
```css
--font-geist: 'Geist', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
--font-mono: 'Geist Mono', 'JetBrains Mono', monospace;
```

### Type Scale
| Role | Font Size | Line Height | Letter Spacing | Font Weight | CSS Token |
|------|-----------|-------------|----------------|-------------|-----------|
| **caption** | 12px (0.75rem) | 1.33 | `+0.050em` | 500 / 600 | `--text-caption` |
| **body-sm** | 13px (0.8125rem) | 1.40 | `0` | 400 | `--text-body-sm` |
| **body** | 14px (0.875rem) | 1.45 | `0` | 400 | `--text-body` |
| **body-lg** | 16px (1.0rem) | 1.50 | `-0.010em` | 400 / 500 | `--text-body-lg` |
| **subheading** | 18px (1.125rem) | 1.40 | `-0.015em` | 600 | `--text-subheading` |
| **heading-sm** | 24px (1.5rem) | 1.30 | `-0.025em` | 600 | `--text-heading-sm` |
| **heading** | 30px (1.875rem) | 1.20 | `-0.035em` | 600 | `--text-heading` |
| **heading-lg** | 36px (2.25rem) | 1.15 | `-0.040em` | 600 | `--text-heading-lg` |
| **display** | 48px–56px | 1.08 | `-0.050em` | 600 / 700 | `--text-display` |

---

## 4. Spacing, Shapes & Elevation

### Base Scale
- **Base Grid Unit:** 4px (spacing variables: 4px, 8px, 12px, 16px, 20px, 24px, 32px, 48px, 64px).
- **Max Viewport Width:** 1280px (centered with responsive padding `20px` to `32px`).

### Border Radii
| Component Group | Radius Value | Semantic Class |
|-----------------|--------------|----------------|
| **Cards & Bento Panels** | `24px` | `--radius-card` |
| **Buttons & Search Inputs** | `18px` (pill) | `--radius-pill` |
| **Badges & Filter Chips** | `18px` (pill) | `--radius-pill` |
| **Nested Matrix Cells & Tags** | `8px`–`10px` | `--radius-nested` |

### Blueprint Elevation & Shadows
- **Card Shadow**:
  ```css
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.04), 
              0 1px 3px rgba(0, 0, 0, 0.05), 
              0 12px 24px -12px rgba(0, 0, 0, 0.03);
  ```
- **Blueprint Watermark**:
  ```css
  background-image: 
    radial-gradient(circle at 50% 0%, rgba(231, 0, 11, 0.025) 0%, transparent 50%),
    linear-gradient(to right, rgba(0, 0, 0, 0.035) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(0, 0, 0, 0.035) 1px, transparent 1px);
  background-size: 100% 100%, 24px 24px, 24px 24px;
  ```

---

## 5. Component System Guidelines

### 5.1 Navigation Bar
- **Surface**: Frosted white (`rgba(255, 255, 255, 0.85)`), `backdrop-filter: blur(16px)`, `border-bottom: 1px solid #e5e5e5`.
- **Brand Title**: Deep Ink `#0a0a0a` with a subtle Ember period/accent.
- **Nav Links**: 14px Geist weight 500, color `#737373`, transitioning to `#0a0a0a` on hover with a smooth slide pill indicator.

### 5.2 Primary & Secondary Buttons
- **Primary Action (Filled)**: Background `#0a0a0a`, text `#ffffff`, radius `18px`, height `38px`, font 14px Geist weight 500, subtle hover translateY(-1px) with soft shadow.
- **Secondary Action (Ghost/Outline)**: Background `#ffffff`, text `#0a0a0a`, border `1px solid #e5e5e5`, radius `18px`, hover background `#f5f5f5`.
- **Ember Alert Action**: Background `#fef2f2`, text `#b91c1c`, border `1px solid #fecaca`, radius `18px`.

### 5.3 Bento Cards & Stat Blocks
- **Card Container**: Background `#ffffff`, border `1px solid #e5e5e5`, radius `24px`, padding `24px`.
- **Stat Metric**: Big numeric display (`36px`–`48px`), weight 600, letter spacing `-0.04em`, font-variant `tabular-nums`.
- **Stat Label**: 12px uppercase Geist, weight 600, letter spacing `0.05em`, color `#737373`.

### 5.4 Interactive Causal Attention Matrix Visualizer
- **Matrix Cells**: Crisp squares (`32px`–`36px`) with `8px` radius.
- **Attended Tokens (Standard Pre-training)**: Soft slate fill `#f5f5f5` with text `#0a0a0a` and border `1px solid #e5e5e5`.
- **Causally Masked SPP Constitutional Reflection**: Ember-tinted cell `#fef2f2` with border `1px dashed #f87171` and Ember block icon `⊘` (demonstrating loss computation without self-token contamination).
- **Hover Inspector**: High-contrast tooltip showing attention weight scalar $\alpha_{i,j}$.

### 5.5 Cross-Lingual Bias Benchmark Explorer
- **Tabs**: 18px pill segment control in `#fafafa` container with 1px border. Active tab highlighted in `#ffffff` with ink text and faint shadow.
- **Parallel Probe Card**: Clean horizontal comparison layout:
  - Romanian stimulus (stereotypical vs. anti-stereotypical sentence pairs).
  - English counterpart with calculated alignment disparity ($\Delta SPM$).
  - Clear, accessible badges: Green `#15803d` for balanced/anti-biased, Ember `#b91c1c` for stereotypical drift.

### 5.6 Live Unmasking Adversarial Playground
- **Input Container**: Soft white surface with hairline outline `#e5e5e5` and 18px pill focus ring.
- **Mode Toggle Switch**: Visual dual-state switch between "Standard Unaligned Base" and "SPP-Ro Constitutional".
- **Real-Time Output Streams**: Side-by-side comparative generation panels rendering token-by-token completion traces.

---

## 6. Do's & Don'ts

### Do:
- Maintain **18px pill radius** on all buttons, badges, and inputs for consistent tactile rhythm.
- Use **24px container radius** on cards, preview panels, and charts.
- Keep the background **#f5f5f5 with a faint 24px grid** to preserve the architectural blueprint feel.
- Reserve **#e7000b** for meaningful disparities, blocked attention reflections, and live alerts.
- Use tabular figures (`tabular-nums`) on all numeric metrics, percentages, and benchmark values.

### Don't:
- Do not add saturated multi-color gradients (purple/neon blue/cyan wash) across cards.
- Do not use harsh sharp corners (0px) or inconsistent random radii (4px, 12px, 30px).
- Do not omit 1px hairline borders (`#e5e5e5`) on white cards.
- Do not use dark mode as the default — this design system is fundamentally built for high-legibility clinical light mode.
