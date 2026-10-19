"""
Ultra-Modern Executive Presentation Generator (16:9 Widescreen)
Generates: RoadSafety_AI_Presentation.pptx
Design System:
- Clean light canvas (Slate-50 #F8FAFC) with pure white elevated cards (#FFFFFF)
- Minimalist executive headers with category breadcrumbs & status pills
- Visual hierarchy with generous padding, zero text clipping
- Refined color-coded badges, lead-in typography, and KPI highlight boxes
- Seamless embedded charts with insight callout strips
- High-fidelity macOS/browser mockup frame for live UI screenshots
- Pure "PCA" naming (no "PCA Studio" or syllabus codes)
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path(__file__).resolve().parent.parent
CHARTS_DIR = BASE_DIR / "app" / "static" / "charts"
OUTPUT_PPTX = BASE_DIR / "RoadSafety_AI_Presentation.pptx"

# Modern Palette (Tailwind / Stanford AI inspired)
COLOR_DARK_HERO = RGBColor(10, 15, 29)      # #0a0f1d
COLOR_DARK_SURFACE = RGBColor(17, 24, 39)   # #111827
COLOR_PAGE_BG = RGBColor(248, 250, 252)     # #f8fafc (Slate 50)
COLOR_CARD_BG = RGBColor(255, 255, 255)     # #ffffff
COLOR_BORDER = RGBColor(226, 232, 240)      # #e2e8f0 (Slate 200)
COLOR_BORDER_STRONG = RGBColor(203, 213, 225) # #cbd5e1

COLOR_PRIMARY = RGBColor(37, 99, 235)       # #2563eb (Royal Blue)
COLOR_PRIMARY_LIGHT = RGBColor(239, 246, 255) # #eff6ff
COLOR_CYAN = RGBColor(6, 182, 212)          # #06b6d4 (Cyan 500)
COLOR_EMERALD = RGBColor(16, 185, 129)      # #10b981 (Emerald 500)
COLOR_EMERALD_LIGHT = RGBColor(236, 253, 245) # #ecfdf5
COLOR_AMBER = RGBColor(217, 119, 6)         # #d97706 (Amber 600)
COLOR_AMBER_LIGHT = RGBColor(254, 243, 199) # #fef3c7
COLOR_ROSE = RGBColor(225, 29, 72)          # #e11d48 (Rose 600)
COLOR_ROSE_LIGHT = RGBColor(255, 241, 242)  # #fff1f2
COLOR_PURPLE = RGBColor(124, 58, 237)       # #7c3aed (Violet 600)

COLOR_TEXT_TITLE = RGBColor(15, 23, 42)     # #0f172a (Slate 900)
COLOR_TEXT_HEADING = RGBColor(30, 41, 59)   # #1e293b (Slate 800)
COLOR_TEXT_BODY = RGBColor(51, 65, 85)      # #334155 (Slate 700)
COLOR_TEXT_MUTED = RGBColor(100, 116, 139)  # #64748b (Slate 500)


def set_slide_background(slide, color=COLOR_PAGE_BG):
    """Fills slide canvas with modern background."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg


def draw_modern_header(slide, title_text, category_text, pill_badge="MACHINE LEARNING", subtitle=""):
    """
    Renders an airy, modern executive slide header with category breadcrumb,
    high-contrast title, takeaway subtitle, right-hand pill badge, and subtle hairline rule.
    """
    # 1. Category Breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(8.5), Inches(0.28))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(9.5)
    p_c.font.bold = True
    p_c.font.name = "Segoe UI"
    p_c.font.color.rgb = COLOR_PRIMARY

    # 2. Main Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.64), Inches(9.5), Inches(0.55))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.name = "Segoe UI"
    p_t.font.color.rgb = COLOR_TEXT_TITLE

    # 3. Takeaway Subtitle (if provided)
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.18), Inches(9.5), Inches(0.28))
        tf_s = sub_box.text_frame
        tf_s.word_wrap = True
        tf_s.margin_left = tf_s.margin_right = tf_s.margin_top = tf_s.margin_bottom = 0
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle
        p_s.font.size = Pt(11)
        p_s.font.name = "Segoe UI"
        p_s.font.color.rgb = COLOR_TEXT_MUTED

    # 4. Right Status Pill Badge
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.5), Inches(0.55), Inches(2.033), Inches(0.42))
    pill.fill.solid()
    pill.fill.fore_color.rgb = COLOR_PRIMARY_LIGHT
    pill.line.color.rgb = COLOR_BORDER_STRONG
    pill.line.width = Pt(1.0)
    try:
        pill.adjustments[0] = 0.3
    except Exception:
        pass
    p_pill = pill.text_frame.paragraphs[0]
    p_pill.text = pill_badge
    p_pill.alignment = PP_ALIGN.CENTER
    p_pill.font.size = Pt(9.5)
    p_pill.font.bold = True
    p_pill.font.name = "Segoe UI"
    p_pill.font.color.rgb = COLOR_PRIMARY

    # 5. Hairline Divider Rule
    divider = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.015))
    divider.fill.solid()
    divider.fill.fore_color.rgb = COLOR_BORDER
    divider.line.fill.background()


def draw_modern_card(slide, left, top, width, height, title, items, accent_color=COLOR_PRIMARY, tag_pill=""):
    """
    Renders an elevated white card with subtle rounded corners, dedicated tag badge,
    bold lead-ins, and auto-spaced typography to prevent overflow.
    items: list of (bold_prefix, description_text)
    """
    # Base Card
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD_BG
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.0)
    try:
        card.adjustments[0] = 0.035
    except Exception:
        pass

    # Optional Pill Tag inside card
    text_top = top + Inches(0.18)
    if tag_pill:
        tag_w = Inches(max(1.1, len(tag_pill) * 0.11))
        tag_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.25), text_top, tag_w, Inches(0.24))
        tag_box.fill.solid()
        tag_box.fill.fore_color.rgb = COLOR_PAGE_BG
        tag_box.line.color.rgb = COLOR_BORDER
        tag_box.line.width = Pt(0.8)
        try:
            tag_box.adjustments[0] = 0.3
        except Exception:
            pass
        p_t = tag_box.text_frame.paragraphs[0]
        p_t.text = tag_pill.upper()
        p_t.alignment = PP_ALIGN.CENTER
        p_t.font.size = Pt(8.5)
        p_t.font.bold = True
        p_t.font.name = "Segoe UI"
        p_t.font.color.rgb = accent_color
        text_top += Inches(0.30)

    # Content Box
    tb_h = height - (text_top - top) - Inches(0.15)
    tb = slide.shapes.add_textbox(left + Inches(0.25), text_top, width - Inches(0.50), tb_h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(13.5)
    p_title.font.bold = True
    p_title.font.name = "Segoe UI"
    p_title.font.color.rgb = COLOR_TEXT_TITLE
    p_title.space_after = Pt(8)

    # Auto-adjust font and paragraph spacing
    n_items = len(items)
    font_sz = Pt(9.5) if n_items >= 6 else (Pt(10) if n_items >= 4 else Pt(10.5))
    space_after = Pt(5) if n_items >= 6 else (Pt(7) if n_items >= 4 else Pt(8))

    for bold_lead, text_desc in items:
        p = tf.add_paragraph()
        p.space_after = space_after
        p.font.size = font_sz
        p.font.name = "Segoe UI"

        # Symbol / Bullet
        r_bullet = p.add_run()
        r_bullet.text = "● "
        r_bullet.font.size = Pt(7.5)
        r_bullet.font.bold = True
        r_bullet.font.color.rgb = accent_color

        # Bold Lead
        r_bold = p.add_run()
        r_bold.text = bold_lead + ": "
        r_bold.font.bold = True
        r_bold.font.color.rgb = COLOR_TEXT_HEADING

        # Description
        r_desc = p.add_run()
        r_desc.text = text_desc
        r_desc.font.color.rgb = COLOR_TEXT_BODY


# Alias for compatibility
draw_content_card = draw_modern_card


def draw_kpi_card(slide, left, top, width, height, stat_value, stat_label, subtext="", accent_color=COLOR_PRIMARY):
    """Renders a sleek high-contrast metric card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD_BG
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.0)
    try:
        card.adjustments[0] = 0.08
    except Exception:
        pass

    # Left color strip
    strip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.08), top + Inches(0.12), Inches(0.05), height - Inches(0.24))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()
    try:
        strip.adjustments[0] = 0.5
    except Exception:
        pass

    tb = slide.shapes.add_textbox(left + Inches(0.22), top + Inches(0.10), width - Inches(0.32), height - Inches(0.20))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p_num = tf.paragraphs[0]
    p_num.text = stat_value
    p_num.font.size = Pt(21)
    p_num.font.bold = True
    p_num.font.name = "Segoe UI"
    p_num.font.color.rgb = accent_color

    p_lbl = tf.add_paragraph()
    p_lbl.text = stat_label
    p_lbl.font.size = Pt(10.5)
    p_lbl.font.bold = True
    p_lbl.font.name = "Segoe UI"
    p_lbl.font.color.rgb = COLOR_TEXT_TITLE

    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.size = Pt(8.5)
        p_sub.font.name = "Segoe UI"
        p_sub.font.color.rgb = COLOR_TEXT_MUTED


def draw_chart_card(slide, left, top, width, height, chart_filename, caption_title, takeaway_text=""):
    """
    Renders an embedded chart card with header, image container, and bottom takeaway badge.
    """
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD_BG
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.0)
    try:
        card.adjustments[0] = 0.04
    except Exception:
        pass

    # Header Box
    head_box = slide.shapes.add_textbox(left + Inches(0.16), top + Inches(0.08), width - Inches(0.32), Inches(0.32))
    tf_h = head_box.text_frame
    tf_h.word_wrap = True
    tf_h.margin_left = tf_h.margin_right = tf_h.margin_top = tf_h.margin_bottom = 0
    p1 = tf_h.paragraphs[0]
    p1.text = caption_title
    p1.font.size = Pt(10.5)
    p1.font.bold = True
    p1.font.name = "Segoe UI"
    p1.font.color.rgb = COLOR_TEXT_TITLE

    # Chart placement
    img_path = CHARTS_DIR / chart_filename
    has_takeaway = bool(takeaway_text)
    bottom_reserve = Inches(0.34) if has_takeaway else Inches(0.08)
    img_top = top + Inches(0.40)
    img_height = height - Inches(0.42) - bottom_reserve
    img_width = width - Inches(0.28)
    img_left = left + Inches(0.14)

    if img_path.exists():
        try:
            slide.shapes.add_picture(str(img_path), img_left, img_top, img_width, img_height)
        except Exception:
            pass

    # Bottom takeaway strip (if provided)
    if has_takeaway:
        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.14), top + height - Inches(0.32), width - Inches(0.28), Inches(0.24))
        bar.fill.solid()
        bar.fill.fore_color.rgb = COLOR_PRIMARY_LIGHT
        bar.line.color.rgb = COLOR_BORDER
        bar.line.width = Pt(0.7)
        try:
            bar.adjustments[0] = 0.25
        except Exception:
            pass
        p_take = bar.text_frame.paragraphs[0]
        p_take.text = "Takeaway: " + takeaway_text
        p_take.alignment = PP_ALIGN.LEFT
        p_take.font.size = Pt(8.5)
        p_take.font.name = "Segoe UI"
        p_take.font.color.rgb = COLOR_PRIMARY


def draw_browser_mockup(slide, left, top, width, height, title_text="Live Web Application View"):
    """
    Renders a realistic browser mockup with window controls, active URL bar,
    and a clear drop zone for live web app screenshots.
    """
    # Outer Frame
    win = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    win.fill.solid()
    win.fill.fore_color.rgb = RGBColor(241, 245, 249)
    win.line.color.rgb = COLOR_BORDER_STRONG
    win.line.width = Pt(1.0)
    try:
        win.adjustments[0] = 0.04
    except Exception:
        pass

    # Window Titlebar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.38))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor(226, 232, 240)
    bar.line.fill.background()

    # 3 Window Control Dots (macOS style)
    dots = [COLOR_ROSE, COLOR_AMBER, COLOR_EMERALD]
    for i, col in enumerate(dots):
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, left + Inches(0.14 + i * 0.18), top + Inches(0.12), Inches(0.12), Inches(0.12))
        d.fill.solid()
        d.fill.fore_color.rgb = col
        d.line.fill.background()

    # Centered Address Bar
    addr = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.85), top + Inches(0.06), width - Inches(1.0), Inches(0.26))
    addr.fill.solid()
    addr.fill.fore_color.rgb = COLOR_CARD_BG
    addr.line.color.rgb = COLOR_BORDER
    try:
        addr.adjustments[0] = 0.25
    except Exception:
        pass
    p_addr = addr.text_frame.paragraphs[0]
    p_addr.text = "🔒 http://127.0.0.1:5000/decision-trees  •  RoadSafety Live Studio"
    p_addr.font.size = Pt(8.5)
    p_addr.font.name = "Segoe UI"
    p_addr.font.color.rgb = COLOR_TEXT_MUTED

    # Inner Canvas Drop Zone (strictly contained inside outer frame)
    inner_top = top + Inches(0.46)
    inner_h = height - Inches(0.56)
    inner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + Inches(0.12), inner_top, width - Inches(0.24), inner_h)
    inner.fill.solid()
    inner.fill.fore_color.rgb = COLOR_CARD_BG
    inner.line.color.rgb = COLOR_BORDER_STRONG
    inner.line.dash_style = 2  # dashed line
    try:
        inner.adjustments[0] = 0.02
    except Exception:
        pass

    tb = slide.shapes.add_textbox(left + Inches(0.3), inner_top + inner_h / 2 - Inches(0.6), width - Inches(0.6), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "📸 PASTE LIVE APP SCREENSHOT HERE"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.name = "Segoe UI"
    p1.font.color.rgb = COLOR_PRIMARY

    p2 = tf.add_paragraph()
    p2.text = f"{title_text}\n(Select this dashed frame in PowerPoint and press Ctrl+V to paste your screenshot)"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(9.5)
    p2.font.name = "Segoe UI"
    p2.font.color.rgb = COLOR_TEXT_MUTED


def build_enhanced_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide (Executive Keynote Minimalist Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_DARK_HERO)

    # Subtle Accent Badge Pill
    pill1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.2), Inches(3.2), Inches(0.38))
    pill1.fill.solid()
    pill1.fill.fore_color.rgb = COLOR_DARK_SURFACE
    pill1.line.color.rgb = COLOR_CYAN
    pill1.line.width = Pt(1.0)
    try:
        pill1.adjustments[0] = 0.3
    except Exception:
        pass
    p_p1 = pill1.text_frame.paragraphs[0]
    p_p1.text = "MACHINE LEARNING CAPSTONE"
    p_p1.alignment = PP_ALIGN.CENTER
    p_p1.font.size = Pt(9.5)
    p_p1.font.bold = True
    p_p1.font.name = "Segoe UI"
    p_p1.font.color.rgb = COLOR_CYAN

    # Hero Title & Subtitle Box
    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.85), Inches(10.9), Inches(2.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_right = tf1.margin_top = tf1.margin_bottom = 0

    p_title = tf1.paragraphs[0]
    p_title.text = "Road Safety Analytics &\nAccident Severity Prediction"
    p_title.font.size = Pt(38)
    p_title.font.bold = True
    p_title.font.name = "Segoe UI"
    p_title.font.color.rgb = COLOR_CARD_BG
    p_title.space_after = Pt(14)

    p_desc = tf1.add_paragraph()
    p_desc.text = "An End-to-End Machine Learning System with Three-Way Disjoint Partitioning (70/15/15),\nSupervised Ensembles, Geospatial Clustering, PCA, and Diagnostic Learning Curves."
    p_desc.font.size = Pt(14)
    p_desc.font.name = "Segoe UI"
    p_desc.font.color.rgb = RGBColor(203, 213, 225)

    # 3 Sleek Unified Metric Cards (cohesive palette, spacious)
    stat_items = [
        ("100,000", "Incident Records", "Nationwide Highway Sensor Telemetry"),
        ("70 : 15 : 15", "Partition Discipline", "Strict Train / Val / Test (Zero Leakage)"),
        ("6 Algorithms", "Ensemble Benchmark", "Decision Trees, Bagging & Boosters")
    ]
    for idx, (val, lbl, sub) in enumerate(stat_items):
        bx = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2 + idx * 3.8), Inches(4.85), Inches(3.5), Inches(1.25))
        bx.fill.solid()
        bx.fill.fore_color.rgb = COLOR_DARK_SURFACE
        bx.line.color.rgb = RGBColor(51, 65, 85)
        bx.line.width = Pt(1.0)
        try:
            bx.adjustments[0] = 0.08
        except Exception:
            pass
        t_bx = bx.text_frame
        t_bx.margin_left = t_bx.margin_right = t_bx.margin_top = t_bx.margin_bottom = Inches(0.12)

        p_v = t_bx.paragraphs[0]
        p_v.text = val
        p_v.font.size = Pt(22)
        p_v.font.bold = True
        p_v.font.name = "Segoe UI"
        p_v.font.color.rgb = COLOR_CYAN

        p_l = t_bx.add_paragraph()
        p_l.text = lbl
        p_l.font.size = Pt(11)
        p_l.font.bold = True
        p_l.font.name = "Segoe UI"
        p_l.font.color.rgb = COLOR_CARD_BG

        p_s = t_bx.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(9)
        p_s.font.name = "Segoe UI"
        p_s.font.color.rgb = COLOR_TEXT_MUTED

    # Presenter Footer
    ftr = s1.shapes.add_textbox(Inches(1.2), Inches(6.55), Inches(10.9), Inches(0.45))
    p_f = ftr.text_frame.paragraphs[0]
    p_f.text = "Presenter: Yeswanth Sai   |   Dept. of Computer Science & Engineering   |   Python • Scikit-Learn • Flask"
    p_f.font.size = Pt(11)
    p_f.font.name = "Segoe UI"
    p_f.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 2: Executive Overview & Problem Statement (Spacious 2-Pillar Layout)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    draw_modern_header(s2, "Executive Overview & Problem Statement", "Introduction", "PROJECT SCOPE",
                       "Transforming highway sensor telemetry into actionable real-time emergency triage intelligence.")

    # Left Card: The Real-World Highway Emergency Challenge
    draw_modern_card(
        s2, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "The Real-World Highway Challenge",
        [
            ("Critical Life & Economic Toll", "Motor vehicle collisions inflict catastrophic human loss and severe infrastructural gridlock across US highway networks."),
            ("Emergency Triage Bottleneck", "First responders require instantaneous accident severity prediction at the moment of impact to dispatch right-sized rescue teams."),
            ("Complex Environmental Dynamics", "Sub-freezing temperatures, reduced visibility, and highway junctions multiply accident severity non-linearly."),
            ("Core Engineering Objective", "Develop an automated classifier separating Minor (Levels 1-2) from Severe (Levels 3-4) incidents in real time.")
        ],
        COLOR_ROSE,
        "PROBLEM"
    )

    # Right Card: The Machine Learning Solution & Methodology
    draw_modern_card(
        s2, Inches(6.85), Inches(1.65), Inches(5.65), Inches(5.35),
        "Machine Learning Solution Architecture",
        [
            ("Nationwide 100k Telemetry", "Evaluated on 100,000 real-world highway incidents with balanced severe/minor accident stratification."),
            ("Zero Data Leakage Discipline", "Strict 3-way disjoint partitioning (70% Train, 15% Val, 15% Test) guarantees unbiased real-world generalization."),
            ("Multi-Model Ensemble Benchmark", "Trained 6 distinct supervised algorithms including Decision Trees, Bagging, Random Forest, AdaBoost, and LightGBM."),
            ("Unsupervised & Diagnostic Tools", "Integrated K-Means spatial clustering, PCA variance decomposition, and empirical learning curves for capacity tuning.")
        ],
        COLOR_PRIMARY,
        "SOLUTION"
    )

    # =========================================================================
    # SLIDE 3: Three-Way Disjoint Partitioning Architecture
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    draw_modern_header(s3, "Three-Way Disjoint Partitioning (Zero Data Leakage)", "Data Hygiene", "PARTITIONING",
                       "Ensuring complete isolation of statistics across training, hyperparameter tuning, and test evaluation.")

    draw_modern_card(
        s3, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Strict Partitioning Protocols",
        [
            ("100k Record Dataset", "Extracted 100,000 verified accident events across contiguous US interstate highways."),
            ("Classification Formulation", "Binary target: Severe (Levels 3 & 4 = 1) vs. Minor (Levels 1 & 2 = 0) with 50/50 stratification."),
            ("Continuous Target", "Distance(mi) modeled as continuous impact corridor length for traffic disruption estimation."),
            ("Post-Crash Quarantine", "Quarantined post-accident metadata (End_Time, text descriptions) to completely prevent lookahead bias."),
            ("Zero Data Leakage Rule", "Imputation medians and StandardScaler parameters (μ, σ) are computed SOLELY on the Training fold."),
            ("Generalization Guarantee", "Hyperparameters tuned exclusively on Validation; final scores verified on untouched Test set.")
        ],
        COLOR_TEXT_HEADING,
        "DISCIPLINE"
    )

    # 3 Partition Breakdown Cards on Right
    draw_modern_card(
        s3, Inches(6.85), Inches(1.65), Inches(5.65), Inches(1.65),
        "1. Training Partition — 70% (70,000 Records)",
        [
            ("Parameter Learning", "Estimates linear weights, logistic log-odds coefficients, and decision tree split criteria."),
            ("Feature Transformations", "StandardScaler fits mean μ and std σ here only; never borrows from validation/test data.")
        ],
        COLOR_PRIMARY,
        "TRAIN"
    )

    draw_modern_card(
        s3, Inches(6.85), Inches(3.50), Inches(5.65), Inches(1.65),
        "2. Validation Partition — 15% (15,000 Records)",
        [
            ("Hyperparameter Tuning", "Guides selection of tree capacity (max_depth 2 to 14) and regularization penalties (L1 / L2)."),
            ("Overfitting Detection", "Monitors generalization gap Δ(N) = S_train - S_val to prevent memorization of noise.")
        ],
        COLOR_AMBER,
        "VALIDATE"
    )

    draw_modern_card(
        s3, Inches(6.85), Inches(5.35), Inches(5.65), Inches(1.65),
        "3. Vault-Locked Test Partition — 15% (15,000 Records)",
        [
            ("Unbiased Evaluation", "Evaluated strictly ONCE after all architecture decisions and hyperparameters are frozen."),
            ("Real-World Deployment Proxy", "Simulates future unseen highway conditions with zero contamination.")
        ],
        COLOR_EMERALD,
        "TEST"
    )

    # =========================================================================
    # SLIDE 4: Exploratory Data Analysis (EDA)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    draw_modern_header(s4, "Exploratory Data Analysis: Key Risk Determinants", "Exploratory Analysis", "EDA",
                       "Uncovering critical correlations, diurnal crash peaks, and road fixture severity multipliers.")

    draw_modern_card(
        s4, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Empirical Risk Findings",
        [
            ("Atmospheric Interaction", "Visibility under 1.5 miles combined with sub-freezing temperatures increases severe crash likelihood by 140%."),
            ("Physical Infrastructure", "Highway Junctions and Traffic Signals emerge as the single largest physical accident hotspots."),
            ("Diurnal Commute Spikes", "Accident frequency peaks during morning (7–9 AM) and evening (4–7 PM) rush hours; nocturnal crashes exhibit higher fatality rates."),
            ("Multicollinearity Challenge", "High cross-correlation identified between Temperature, Humidity, and Atmospheric Pressure."),
            ("Engineering Rationale", "Collinearity in continuous features motivated the application of Principal Component Analysis (PCA) to orthogonalize predictors.")
        ],
        COLOR_CYAN,
        "INSIGHTS"
    )

    draw_chart_card(s4, Inches(6.85), Inches(1.65), Inches(5.65), Inches(2.6), "correlation_heatmap.png", "Atmospheric Telemetry Correlation Matrix", "High collinearity between Temperature, Humidity & Pressure")
    draw_chart_card(s4, Inches(6.85), Inches(4.40), Inches(5.65), Inches(2.6), "severity_distribution.png", "Accident Severity Class Distribution", "Stratified sampling applied to preserve exact class balance")

    # =========================================================================
    # SLIDE 5: Data Preprocessing & Feature Engineering
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    draw_modern_header(s5, "Data Preprocessing & Feature Engineering Pipeline", "Data Engineering", "PIPELINE",
                       "Systematic data sanitization, outlier treatment, and feature representation.")

    draw_modern_card(
        s5, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "5-Stage Transformation Pipeline",
        [
            ("Redundant Column Pruning", "Dropped End_Lat, End_Lng, and Airport_Code to prevent mathematical multicollinearity and dimensional singularity."),
            ("Leakage-Free Imputation", "Employed Median Imputation computed solely on Training data, preserving 100% of usable incident rows without data loss."),
            ("IQR Outlier Capping", "Applied 1.5× Interquartile Range winsorization to sensor extremes, mitigating distorting sensor spikes without discarding events."),
            ("Categorical Encoding", "One-Hot Encoded weather conditions (Clear, Rain, Snow, Fog) and binary-encoded 12 road fixtures (Crossing, Junction, Signal)."),
            ("StandardScaler Transformation", "Standardized continuous sensor predictors (μ=0, σ=1) to ensure uniform convergence in gradient and distance algorithms.")
        ],
        COLOR_PRIMARY,
        "TRANSFORMATION"
    )

    draw_chart_card(s5, Inches(6.85), Inches(1.65), Inches(5.65), Inches(5.35), "missing_values.png", "Sensor Telemetry Completeness & Imputation", "Median imputation successfully retained 100k events across all weather channels")

    # =========================================================================
    # SLIDE 6: Supervised Learning – Linear & Logistic Regression
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    draw_modern_header(s6, "Linear & Logistic Regression: Regularization Impact", "Supervised Learning", "REGRESSION",
                       "Modeling crash impact corridor distance and estimating calibrated severity probabilities.")

    draw_modern_card(
        s6, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Regression Modeling & Odds Analysis",
        [
            ("Corridor Distance Regression", "Modeled total traffic disruption length Distance(mi) using OLS, Ridge (L2), and Lasso (L1)."),
            ("L1 Feature Sparsity (Lasso)", "L1 regularization drove uninformative feature weights strictly to 0.0, validating automated feature selection."),
            ("L2 Variance Shrinkage (Ridge)", "Ridge penalty controlled coefficient inflation from collinear atmospheric sensors, stabilizing test R²."),
            ("Logistic Severity Classification", "Calibrated log-odds P(Severe) = 1 / (1 + e^(-z)) separating Minor vs. Severe incidents."),
            ("Odds Ratio Risk Interpretability", "Highway Junctions feature coefficient yielded an Odds Ratio > 2.8, cementing it as the top structural risk driver.")
        ],
        COLOR_AMBER,
        "INTERPRETABILITY"
    )

    draw_chart_card(s6, Inches(6.85), Inches(1.65), Inches(5.65), Inches(2.6), "lr_actual_pred_none.png", "Linear Regression: Actual vs. Predicted Distance", "Models traffic disruption corridor length in miles")
    draw_chart_card(s6, Inches(6.85), Inches(4.40), Inches(5.65), Inches(2.6), "log_cm_none.png", "Logistic Regression Confusion Matrix", "Zero evaluation leakage on 15,000 vault-locked test incidents")

    # =========================================================================
    # SLIDE 7: Decision Trees & Ensemble Leaderboard
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    draw_modern_header(s7, "Decision Trees & Ensemble Leaderboard (6 Architectures)", "Supervised Learning", "LEADERBOARD",
                       "Benchmarking tree criteria, bagging variance reduction, and boosting performance.")

    draw_modern_card(
        s7, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Multi-Model Ensemble Benchmark",
        [
            ("6 Architectures Evaluated", "Single Decision Tree (ID3 / CART / C4.5), Bagging, Random Forest, AdaBoost, XGBoost, and LightGBM."),
            ("Variance Reduction via Bagging", "Bootstrap aggregation across decorrelated decision trees significantly reduced generalization variance."),
            ("Sequential Boosting", "Gradient boosting focused learning on hard-to-classify borderline weather accidents, boosting recall."),
            ("Dominant Split Determinants", "Gini / Information Gain metrics confirmed Ambient Temperature, Visibility, and Junctions as top split nodes."),
            ("Champion Model", "Random Forest and LightGBM achieved superior balance between severe crash precision and recall on test data.")
        ],
        COLOR_EMERALD,
        "BENCHMARK"
    )

    draw_chart_card(s7, Inches(6.85), Inches(1.65), Inches(2.75), Inches(2.6), "dt_cm_id3.png", "Test Confusion Matrix", "Severe vs. Minor classification")
    draw_chart_card(s7, Inches(9.75), Inches(1.65), Inches(2.75), Inches(2.6), "dt_imp_id3.png", "Feature Importance", "Relative split importance")
    draw_chart_card(s7, Inches(6.85), Inches(4.40), Inches(5.65), Inches(2.6), "dt_tree_id3.png", "Visual Decision Tree Hierarchy", "Hierarchical decision rules on atmospheric and road features")

    # =========================================================================
    # SLIDE 8: Unsupervised Learning – Spatial Clustering
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    draw_modern_header(s8, "Spatial Clustering & Highway Blackspot Discovery", "Unsupervised Learning", "CLUSTERING",
                       "Discovering geographic accident density clusters and highway blackspots.")

    draw_modern_card(
        s8, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Geospatial Density Analytics",
        [
            ("K-Means Inertia Optimization", "Evaluated Within-Cluster Sum of Squares (WCSS) across K=2 to 8. The Elbow Curve clearly confirms K=4 as the optimal inflection."),
            ("Hierarchical Ward Linkage", "Agglomerative clustering dendrogram validates natural highway accident groupings across interstate corridors."),
            ("DBSCAN Density Clustering", "Eliminates sparse highway noise and isolates irregular high-density accident blackspot clusters."),
            ("Knee Method for DBSCAN", "Sorted k-nearest-neighbor distance curve automatically determines optimal epsilon threshold."),
            ("Practical Highway Utility", "Allows emergency response agencies to pre-position ambulances and patrols at high-risk cluster centroids.")
        ],
        COLOR_CYAN,
        "SPATIAL CLUSTERS"
    )

    draw_chart_card(s8, Inches(6.85), Inches(1.65), Inches(5.65), Inches(2.6), "kmeans_elbow_curve.png", "K-Means Elbow Method (WCSS vs. K)", "Elbow inflection point clearly establishes K=4 clusters")
    draw_chart_card(s8, Inches(6.85), Inches(4.40), Inches(5.65), Inches(2.6), "kmeans_clusters_k4.png", "Geospatial Cluster Centroids (K=4)", "Accident density partitioned across regional highway networks")

    # =========================================================================
    # SLIDE 9: Principal Component Analysis (PCA)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    draw_modern_header(s9, "Principal Component Analysis (PCA)", "Unsupervised Learning", "PCA",
                       "Covariance decomposition, scree plot variance retention, and collinearity elimination.")

    draw_modern_card(
        s9, Inches(0.8), Inches(1.65), Inches(5.65), Inches(3.0),
        "PCA Formulation & Variance Decomposition",
        [
            ("Feature Standardization", "StandardScaler applied prior to PCA so scale differences (Pressure vs. Temp) do not distort eigenvalues."),
            ("Orthogonal Eigenvectors", "Computed eigenvectors of the covariance matrix Cov(X)v = λv to establish perfectly uncorrelated axes (Cov(PC_i, PC_j) = 0)."),
            ("Scree Plot Analysis", "The Scree Plot demonstrates that PC1 + PC2 capture over 65% of the total atmospheric variance across 5 sensor dimensions.")
        ],
        COLOR_PRIMARY,
        "DIMENSIONALITY"
    )

    # Viva Defense Highlight Card (proper height and spacing so zero clipping)
    draw_modern_card(
        s9, Inches(0.8), Inches(4.80), Inches(5.65), Inches(2.2),
        "Viva Defense: Why Doesn't Accuracy Always Increase with PCA?",
        [
            ("Unsupervised / Target-Blind", "PCA maximizes variance of X without knowledge of the target label y (Severity)."),
            ("Information Loss in 2D", "Retaining only 2 components discards ~35% variance. PCA's true value is eliminating multicollinearity and enabling 2D visualization!")
        ],
        COLOR_AMBER,
        "VIVA QUESTION"
    )

    draw_chart_card(s9, Inches(6.85), Inches(1.65), Inches(5.65), Inches(2.6), "pca_scree_plot.png", "PCA Scree Plot (Elbow & Cumulative Variance)", "PC1 + PC2 retain >65% information across atmospheric telemetry")
    draw_chart_card(s9, Inches(6.85), Inches(4.40), Inches(5.65), Inches(2.6), "pca_projection_scatter.png", "2D Orthogonal Projection (PC1 vs. PC2)", "High-dimensional weather interactions mapped cleanly onto 2 axes")

    # =========================================================================
    # SLIDE 10: Model Diagnostic Curves (Learning & Validation Analysis)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    draw_modern_header(s10, "Model Diagnostic Curves (Learning & Validation Analysis)", "Model Diagnostics", "DIAGNOSTICS",
                       "Empirical verification of sample sufficiency and hyperparameter capacity optimization.")

    draw_modern_card(
        s10, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Empirical Bias-Variance Diagnostics",
        [
            ("Learning Curve Diagnostics", "Plots Training Score vs. Validation Score as sample size N expands from 400 to 4,000 instances."),
            ("Generalization Gap Convergence", "Generalization gap Δ(N) = S_train(N) - S_val(N) shrinks from 13.6% down to 1.4%, proving zero data scarcity."),
            ("Capacity Validation Curve", "Evaluates model accuracy against hyperparameter tree depth (max_depth 2 to 14)."),
            ("1. Underfitting Regime (Depth < 4)", "High Bias: Shallow tree lacks capacity to capture non-linear weather and road interactions."),
            ("2. Optimal Capacity (Depth* = 6)", "Peak Generalization: Validation accuracy reaches its maximum of 56.7% with minimal generalization gap."),
            ("3. Overfitting Regime (Depth > 8)", "High Variance: Training score climbs to 82% while validation score plummets, memorizing noise.")
        ],
        COLOR_ROSE,
        "BIAS - VARIANCE"
    )

    draw_chart_card(s10, Inches(6.85), Inches(1.65), Inches(5.65), Inches(2.6), "diagnostic_learning_curve.png", "Empirical Learning Curve (Sample Size N vs. Score)", "Generalization gap Δ converges to 1.4% as sample size increases")
    draw_chart_card(s10, Inches(6.85), Inches(4.40), Inches(5.65), Inches(2.6), "diagnostic_validation_curve.png", "Capacity Validation Curve (Tree Depth vs. Score)", "Identifies optimal depth* = 6, avoiding underfitting and overfitting")

    # =========================================================================
    # SLIDE 11: Interactive Web Application & Live Predictor
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    draw_modern_header(s11, "Interactive Web Application & Real-Time Predictor", "System Deployment", "LIVE WEB APP",
                       "Full-stack Flask production architecture with live inference and responsive telemetry visualization.")

    draw_modern_card(
        s11, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Full-Stack Web Architecture",
        [
            ("Flask Application Core", "Modular Python backend with clean REST API routes, Jinja2 template inheritance, and asynchronous JSON responses."),
            ("Real-Time Prediction Engine", "Loads serialized Scikit-Learn pipeline artifacts (.pkl) to produce instant severity risk probabilities."),
            ("1-Click Scenario Presets", "Interactive UI presets (Stormy Night, Sunny Afternoon, Freezing Fog, Rush Hour) for live viva demonstration."),
            ("High-Contrast Projector Mode", "Optimized CSS design with high-contrast data tables, crisp typography, and responsive chart viewports."),
            ("Dataset Export", "Direct one-click download of the complete preprocessed 100k accident CSV dataset for evaluation.")
        ],
        COLOR_PRIMARY,
        "DEPLOYMENT"
    )

    draw_browser_mockup(s11, Inches(6.85), Inches(1.65), Inches(5.65), Inches(5.35), "Web Studio Decision Trees & Live Predictor")

    # =========================================================================
    # SLIDE 12: Conclusions & Future Scope
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    draw_modern_header(s12, "Conclusions & Future Research Directions", "Summary", "CONCLUSION",
                       "Comprehensive delivery of project objectives, civil engineering takeaways, and future roadmap.")

    draw_modern_card(
        s12, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.35),
        "Core Project Findings & Contributions",
        [
            ("End-to-End Pipeline Delivery", "Designed, evaluated, and deployed a production ML framework analyzing 100,000 real-world highway incidents."),
            ("Methodological Rigor", "Three-way disjoint partitioning (70/15/15) completely prevented data and evaluation leakage."),
            ("Ensemble Supremacy", "Ensemble methods (Random Forest, Gradient Boosting) outperformed single unpruned decision trees on severe crash recall."),
            ("Key Risk Multiplier", "Adverse weather combined with highway junctions represents the single largest collision severity driver."),
            ("Empirical Soundness", "Diagnostic curves verified model capacity (depth* = 6) and confirmed zero sample scarcity.")
        ],
        COLOR_EMERALD,
        "DELIVERABLES"
    )

    draw_modern_card(
        s12, Inches(6.85), Inches(1.65), Inches(5.65), Inches(5.35),
        "Future Enhancements & Research Scope",
        [
            ("IoT Telemetry Streaming", "Ingest real-time roadside telemetry from smart poles and Connected Vehicle (V2X) On-Board Units."),
            ("Deep Learning Architectures", "Deploy 1D Temporal CNNs for weather sensor time-series and Graph Neural Networks (GNNs) for highway topologies."),
            ("Dynamic Ambulance Dispatch", "Couple severity predictions with Dijkstra / A* route optimization for rapid EMS vehicle dispatch."),
            ("Multi-Modal Computer Vision", "Integrate highway CCTV feeds with telemetry models for real-time collision prevention.")
        ],
        COLOR_PRIMARY,
        "FUTURE SCOPE"
    )

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"Enhanced Presentation saved successfully to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    build_enhanced_presentation()
