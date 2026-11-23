"""Build the 2026-11-23 meeting deck (Step 8 recipe and Step 9 consecutive goals) in the style of the 11-16 deck.

Needs python-pptx (not in the training environment):
    python build_slides.py <template.pptx> <media folder> <output.pptx>

The template is the 11-16 deck: its theme and layouts are kept and its slides removed. Every number below is copied
from REPORT.md, results/grid8/summary_8*.md, results/chains/summary.md and results/front_crashes/summary.md.
"""
import re
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
INK, MUTED, HEAD_FILL, BODY_FILL, MARK_FILL, WHITE = "0F172A", "595959", "E6F0F1", "FFFFFF", "0E7C7B", "FFFFFF"
X0, W0 = 0.34, 9.32  # left edge and full content width (inches), as in the 11-16 deck


# --- primitives --------------------------------------------------------------------------------------------------
def new_slide(deck, title, notes=""):
    slide = deck.slides.add_slide(deck.slide_layouts.get_by_name("TITLE_AND_BODY"))
    for shape in list(slide.placeholders):
        if shape.placeholder_format.idx != 0:  # keep only the title, as the 11-16 deck does
            shape._element.getparent().remove(shape._element)
    head = slide.shapes.title
    head.left, head.top, head.width, head.height = Emu(311700), Emu(291250), Emu(8520600), Emu(550200)
    head.text_frame.text = title
    head.text_frame.paragraphs[0].runs[0].font.size = Pt(19.2)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def nb(words):
    """Keep numbers with their units and comparison signs with their values on one line."""
    words = re.sub(r"(?<=\d) (?=(m/s|s|m|M|of|seeds|goals|decisions)\b)", " ", words)
    words = re.sub(r"\bp (?=[<≥=\d])", "p ", words)
    return re.sub(r"(?<=[<≥≤]) ", " ", words)


def _run(paragraph, text, size, bold=False, italic=False, color=INK, font=None):
    run = paragraph.add_run()
    run.text = nb(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    if font:
        run.font.name = font
    return run


def _bullet(paragraph, space_after=5):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", "285750")
    pPr.set("indent", "-285750")
    spc = etree.SubElement(pPr, qn("a:spcAft"))
    etree.SubElement(spc, qn("a:spcPts")).set("val", str(int(space_after * 100)))
    etree.SubElement(pPr, qn("a:buChar")).set("char", "-")


def text(slide, x, y, w, h, paragraphs, anchor=None, name=None):
    """paragraphs: list of dicts: runs=[(text, {bold, italic, color, size, font})] or text=..., bullet, size, ..."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    if anchor:
        frame.vertical_anchor = anchor
    for i, spec in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        size = spec.get("size", 13)
        runs = spec.get("runs") or [(spec["text"], {})]
        for chunk, style in runs:
            _run(p, chunk, style.get("size", size), bold=style.get("bold", spec.get("bold", False)),
                 italic=style.get("italic", spec.get("italic", False)), color=style.get("color", spec.get("color", INK)),
                 font=style.get("font", spec.get("font")))
        if spec.get("bullet"):
            _bullet(p, spec.get("space_after", 5))
        elif spec.get("space_after") is not None:
            p.space_after = Pt(spec["space_after"])
    return box


def bullets(slide, x, y, w, h, items, size=13, heading=None, space_after=5):
    paragraphs = [{"text": heading, "bold": True, "size": size, "space_after": 4}] if heading else []
    for item in items:
        runs = item if isinstance(item, list) else [(item, {})]
        paragraphs.append({"runs": runs, "bullet": True, "size": size, "space_after": space_after})
    return text(slide, x, y, w, h, paragraphs)


def subtitle(slide, words, y=0.91, h=0.33):
    return text(slide, X0, y, W0, h, [{"text": words, "italic": True, "color": MUTED, "size": 12}])


def caption(slide, words, y, h=0.33, size=11, x=X0, w=W0):
    return text(slide, x, y, w, h, [{"text": words, "italic": True, "color": MUTED, "size": size}])


def _cell(cell, spec, size, header, column):
    """spec: a string, or a dict with t (text, '\n' for lines), b, color, fill, align, mark (shaded cell)."""
    spec = {"t": spec} if isinstance(spec, str) else dict(spec)
    marked = spec.get("mark", False)
    bold = spec.get("b", header or column == 0 or marked)
    color = spec.get("color", WHITE if marked else INK)
    fill = spec.get("fill", MARK_FILL if marked else HEAD_FILL if header else BODY_FILL)
    align = spec.get("align", "l" if column == 0 else "ctr")
    tc = cell._tc
    body = tc.get_or_add_txBody()
    for p in body.findall(qn("a:p")):
        body.remove(p)
    for line in str(spec["t"]).split("\n"):
        p = etree.SubElement(body, qn("a:p"))
        etree.SubElement(p, qn("a:pPr")).set("algn", align)
        r = etree.SubElement(p, qn("a:r"))
        rPr = etree.SubElement(r, qn("a:rPr"), lang="en-US", sz=str(int(round(spec.get("size", size) * 100))),
                               b="1" if bold else "0")
        if spec.get("i"):
            rPr.set("i", "1")
        etree.SubElement(etree.SubElement(rPr, qn("a:solidFill")), qn("a:srgbClr")).set("val", color)
        etree.SubElement(rPr, qn("a:latin")).set("typeface", spec.get("font", "Arial"))
        etree.SubElement(r, qn("a:t")).text = nb(line)
    tcPr = tc.get_or_add_tcPr()
    for child in list(tcPr):
        tcPr.remove(child)
    for key, value in (("marL", "63500"), ("marR", "63500"), ("marT", "25400"), ("marB", "25400"), ("anchor", "ctr")):
        tcPr.set(key, value)
    for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        line = etree.SubElement(tcPr, qn(side), w="12700")
        etree.SubElement(etree.SubElement(line, qn("a:solidFill")), qn("a:srgbClr")).set("val", WHITE)
    etree.SubElement(etree.SubElement(tcPr, qn("a:solidFill")), qn("a:srgbClr")).set("val", fill)


def table(slide, x, y, widths, rows, row_h=0.36, size=12, header_h=None, name=None):
    frame = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)),
                                   Inches(row_h * len(rows)))
    if name:
        frame.name = name
    tbl = frame.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for i, width in enumerate(widths):
        tbl.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(header_h if (r == 0 and header_h) else row_h)
        for c, spec in enumerate(row):
            _cell(tbl.cell(r, c), spec, size, header=(r == 0), column=c)
    return frame


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def demo_row(slide, gif, y, heading, words, h=1.31):
    picture(slide, gif, X0, y, 4.37, h)
    text(slide, 4.93, y, 4.73, h, [{"text": heading, "bold": True, "size": 13},
                                   {"text": words, "color": MUTED, "size": 12}], anchor=MSO_ANCHOR.MIDDLE)


def mark(value):
    return {"t": value, "mark": True}


# --- slides ----------------------------------------------------------------------------------------------------
def build(deck, media):
    # 1. Recap
    s = new_slide(deck, "Recap: where we were last time (P6)",
                  "Last time P6 learned to pass, but crashed a lot and varied by seed. The open questions were whether "
                  "the easy starts are needed, whether longer or steadier training helps, and how to cut crashes. Step 8 "
                  "answers them with a controlled grid; Step 9 is the first test of consecutive goals.")
    bullets(s, X0, 1.09, 5.47, 3.9, [
        "Task: reach a FRONT, BACK, LEFT or RIGHT slot of a moving anchor car in highway traffic, within 25 s; PPO",
        "P6 (practice with easy starts, then the crash cost, then the full task) learned to pass, and beat the scripted "
        "driver on BACK/LEFT/RIGHT on all 3 seeds",
        "But it crashed far more (18–57 of 200, against 5), mostly on FRONT, and results varied by seed",
        "Open questions: are the easy starts needed? Does steadier or longer training help? How to cut crashes?",
        "This time: settle the reward and training stages with a controlled grid (Step 8), then test consecutive goals "
        "(Step 9)",
    ])
    table(s, 6.07, 1.15, [1.3, 1.17, 1.12], [
        ["Policy", "Reached\n(of 200)", "Crashes\n(of 200)"],
        ["P6 seed 7", "182", "18"], ["P6 seed 8", "166", "34"], ["P6 seed 9", "143", "57"], ["Scripted", "153", "5"],
    ], row_h=0.36, size=11, header_h=0.5)
    caption(s, "200 test episodes (seeds 10,000–10,199), as shown last time.", 3.25, size=10, x=6.07, w=3.59)

    # 2. Method
    s = new_slide(deck, "Step 8: how recipes are compared",
                  "Everything in Step 8 is decided by bars written before the runs, on per-seed scores. Screening on 3 "
                  "seeds can only see big effects, so finalists are retrained on 5 new seeds, and the final choice is "
                  "tested on episodes nobody had used.")
    bullets(s, X0, 1.09, W0, 3.9, [
        [("Score", {"bold": True}), (" per episode = reached − crash: +1 reached, −1 crash, 0 timeout; averaged over all "
                                     "slots, FRONT included (from −1 to +1)", {})],
        [("One change at a time:", {"bold": True}), (" every compared pair is checked automatically to differ only in "
                                                     "the factor under test", {})],
        [("Seeds:", {"bold": True}), (" screen on 3 (7, 10, 13), then retrain the finalists on 5 new seeds (16–28): 8 "
                                      "seeds each", {})],
        [("Tests:", {"bold": True}), (" paired t-test over seeds (same seeds, same episodes); each stage's bar is written "
                                      "before its runs", {})],
        [("Selection rule:", {"bold": True}), (" highest score wins, but a more complex option must beat a simpler one "
                                               "with p < 0.05; otherwise the simpler one is kept", {})],
        [("Episodes:", {"bold": True}), (" 400 selection episodes to choose; fresh test sets (50,000–50,499 and "
                                         "60,000–60,499) used once each", {})],
        [("Scale:", {"bold": True}), (" 363 training runs of 2–3M transitions, up to 32 at once on one PC "
                                      "(i9-14900K, CPU only)", {})],
    ])

    # 3. Factors
    s = new_slide(deck, "Step 8: what we varied",
                  "Four training orders, twelve rewards, two network widths and two training lengths. Orders and "
                  "rewards were screened first; width and length were tested on the finalists.")
    table(s, X0, 1.09, [2.3, 7.02], [
        ["Factor", {"t": "Levels tested", "align": "l"}],
        ["Training order", {"t": "N: full task in every round · H: practice (natural start) in round 1 · E: practice "
                                 "with easy starts in round 1 · O: practice with easy starts in rounds 1 and 2 "
                                 "(the P4 → P5 → P6 order)", "align": "l"}],
        ["Crash cost (× goal)", {"t": "0, 0.5, 1, 2", "align": "l"}],
        ["Progress shaping", {"t": "off, or potential-based 0.002 per m toward the slot", "align": "l"}],
        ["Time cost per decision", {"t": "off, 0.001 or 0.004 (× goal, per 0.1 s); only with crash 0.5 or 1",
                                    "align": "l"}],
        ["Network", {"t": "64-64 or 256-256 (separate policy and value networks)", "align": "l"}],
        ["Training length", {"t": "3 rounds × 2M = 6M, or a 4th round of 3M (9M)", "align": "l"}],
    ], row_h=0.42, size=12, header_h=0.4)
    caption(s, "Fixed for every run: PPO (Stable-Baselines3), lr 3e-4, 3 envs × 256 steps per update, minibatch 256, "
               "5 epochs, γ 0.99, GAE λ 0.95, clip 0.2, entropy 0.01, gradient clip 0.5; reward scale 0.02 (goal = +1); "
               "2M transitions per round.", 4.2, h=0.55)

    # 4. Training orders
    s = new_slide(deck, "Training orders: what each round trains on",
                  "Round 1 is always success only, which is how the study got past the give-up collapse. The orders differ "
                  "only in which task rounds 1 and 2 train on, so each neighbouring pair differs in one thing.")
    subtitle(s, "Practice = blocked BACK/LEFT/RIGHT episodes only, with the blocking car held in its lane, so passing is "
                "the only way to succeed. Easy starts put ego past, beside or behind the slow car, or at the normal start.",
             h=0.5)
    table(s, X0, 1.55, [0.75, 2.95, 2.81, 2.81], [
        ["Order", "Round 1 (2M), success only", "Round 2 (2M), reward R", "Round 3 (2M), reward R"],
        ["N", "Full task", "Full task", "Full task"],
        ["H", "Practice, natural start", "Full task", "Full task"],
        ["E", "Practice with easy starts", "Full task", "Full task"],
        [mark("O"), mark("Practice with easy starts"), mark("Practice with easy starts"), mark("Full task")],
    ], row_h=0.42, size=12, header_h=0.42)
    caption(s, "Each round starts from the previous round's policy (same seed); round 1 from fresh weights. With a crash "
               "cost from fresh weights the policy learns to stop before it learns to pass (B1, P1–P3), so round 1 has "
               "none. Shaded: the order adopted in the end.", 3.8, h=0.55)

    # 5. Rewards, exact
    s = new_slide(deck, "Rewards tested (exact terms)",
                  "All rewards are per decision and scaled by 0.02, so reaching the slot is +1. Names read crash, progress, "
                  "time: c1w0t4 is crash 1, no progress, time cost 0.004.")
    table(s, X0, 1.09, [3.3, 3.42, 2.6], [
        [{"t": "Reward term", "align": "l"}, "Levels (goal units, per decision)", "Before the 0.02 scale"],
        ["Reaching the slot (ends the episode)", "+1", "+50"],
        ["Crash (ends the episode)", "0, −0.5, −1, −2", "0, −25, −50, −100"],
        ["Progress shaping (potential-based)", "0 or 0.002 × (0.99·Φ′ − Φ),\nΦ = −distance to the slot (m)",
         "0 or 0.1 × (…)"],
        ["Time cost, every 0.1 s", "0, −0.001, −0.004", "0, −0.05, −0.2"],
        ["Timeout (25 s) or anchor lost", "0", "0"],
        ["Hold bonus, inherited per-step progress", "0 (off in every run)", "0"],
    ], row_h=0.4, size=12, header_h=0.42)
    caption(s, "12 rewards per training order: crash {0, 0.5, 1, 2} × progress {off, on}, plus time {0.001, 0.004} × "
               "crash {0.5, 1} without progress. Why 0.004 at most: a whole timed-out episode then costs 1 (0.37 after "
               "discounting), still cheaper than the smallest crash cost tested, so crashing never pays better than "
               "waiting.", 4.05, h=0.75)

    # 6. Reward grid results
    s = new_slide(deck, "Reward grid (8c, 8c2): nothing beats success only on 3 seeds",
                  "Screening on seeds 7, 10, 13 with 400 selection episodes. The best cell in both orders is P6's reward "
                  "(c1w01), but no reward beats success only beyond noise. The visible effect: any crash cost trades "
                  "crashes for timeouts, because whole runs stop trying on FRONT.")
    grid = [("c0w0  (success only)", "+0.534", "23.3%", "0.1%", "+0.681", "15.9%", "0.1%"),
            ("c0w01", "+0.497", "24.8%", "0.8%", "+0.553", "22.3%", "0.1%"),
            ("c05w0", "+0.535", "12.2%", "22.2%", "+0.578", "13.4%", "15.4%"),
            ("c05w01", "+0.511", "14.2%", "20.6%", "+0.656", "8.6%", "17.1%"),
            ("c05w0t1", "+0.592", "11.9%", "17.1%", "+0.577", "16.6%", "9.0%"),
            ("c05w0t4", "+0.490", "20.5%", "10.1%", "+0.629", "18.4%", "0.3%"),
            ("c1w0", "+0.545", "10.2%", "25.1%", "+0.672", "8.0%", "16.8%"),
            ("c1w01  (P6's reward)", "+0.641", "8.4%", "19.1%", "+0.697", "8.2%", "13.9%"),
            ("c1w0t1", "+0.565", "8.6%", "26.2%", "+0.662", "8.3%", "17.2%"),
            ("c1w0t4", "+0.600", "12.7%", "14.5%", "+0.686", "8.3%", "14.8%"),
            ("c2w0", "+0.490", "5.5%", "40.0%", "+0.451", "13.0%", "28.9%"),
            ("c2w01", "+0.487", "6.6%", "38.2%", "+0.609", "8.9%", "21.2%")]
    rows = [["Reward", "N score", "N crash", "N timeout", "O score", "O crash", "O timeout"]]
    for row in grid:
        best = row[0].startswith("c1w01")
        rows.append([row[0]] + [mark(v) if best and i in (0, 3) else v for i, v in enumerate(row[1:])])
    rows.append([{"t": "Scripted driver", "b": True}, "+0.725", "3.2%", "21.0%", "", "", ""])
    table(s, X0, 1.0, [2.42, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15], rows, row_h=0.255, size=10, header_h=0.3)
    caption(s, "Mean of seeds 7, 10, 13; 400 selection episodes. Shaded: best per order. Rule result: success only (c0w0), "
               "since no reward beats it with p < 0.05 (best: N c1w01 +0.107, p 0.095).", 4.95, h=0.5, size=10)

    # 7. 8d confirmation
    s = new_slide(deck, "Confirmation on 5 new seeds (8d): a crash cost is needed",
                  "The finalists were retrained from scratch on 5 new seeds. Success only now loses beyond noise in order "
                  "N, so a crash cost is needed. The crash-cost rewards are all within 0.07 of each other. The rule picks "
                  "the simplest best one, N_c05w0t1, which then lost to the scripted driver on fresh episodes.")
    table(s, X0, 1.09, [3.9, 1.05, 1.05, 1.05, 2.27], [
        [{"t": "Finalist", "align": "l"}, "Score", "Crash", "Timeout", "Seeds giving up\non FRONT (of 5)"],
        ["N_c05w0t1  (no practice; crash 0.5, time 0.001)", mark("+0.633"), "11.1%", "14.6%", "4"],
        ["O_c1w0t4  (original order; crash 1, time 0.004)", "+0.632", "10.5%", "15.8%", "2"],
        ["O_c1w01  (original order; P6's reward)", "+0.614", "10.1%", "18.5%", "2"],
        ["N_c1w0t4  (no practice; crash 1, time 0.004)", "+0.576", "13.0%", "16.4%", "4"],
        ["N_c1w01  (no practice; P6's reward)", "+0.561", "10.0%", "23.8%", "5"],
        ["O_c0w0  (original order; success only)", "+0.507", "24.6%", "0.1%", "0"],
        ["N_c0w0  (no practice; success only)", "+0.471", "26.3%", "0.3%", "0"],
    ], row_h=0.33, size=11, header_h=0.48)
    bullets(s, X0, 4.1, W0, 1.4, [
        "Success only crashes about 25% and loses to every crash-cost reward in order N (p 0.002–0.049)",
        "Winner by the rule: N_c05w0t1. On the final test (50,000–50,499) it scores +0.628, below the scripted driver's "
        "+0.774 (−0.146, p 0.029); the whole gap is FRONT, where 64-wide runs give up or crash",
    ], size=12)

    # 8. 8e width and length
    s = new_slide(deck, "Width and training length (8e): 256 wide removes the FRONT give-up",
                  "Thin lines are seeds, thick lines the mean of 8; orange is 256 wide. In the original order at 256, the "
                  "practice rounds alone reach the scripted driver's level by 3–4M. By the bars written before, nothing "
                  "was adopted here; the reliability gain was put to a fresh test instead.")
    picture(s, RESULTS / "grid8" / "curves_8e" / "score.png", X0, 1.0, W0, W0 * 598 / 1950)
    bullets(s, X0, 3.95, W0, 1.5, [
        "256 wide: runs that give up on FRONT (≥ 30% timeouts) fall from 13 to 2 of 24; the practice rounds work on "
        "every seed, and the original order then beats no practice (+0.087, p 0.023)",
        "Average gains of +0.06 to +0.13 missed the bar (p ≥ 0.16), because 64-wide seeds vary so much",
        "3M more training (9M) adds +0.02 to +0.04, not beyond noise; the curves are flat by 9M, so 6M is kept",
    ], size=12)

    # 9. 8f final test
    s = new_slide(deck, "Final test on 500 fresh episodes (8f): the 256-wide recipe is adopted",
                  "Seeds 60,000 to 60,499 were checked unused, and the bar was written before the evaluation. The candidate "
                  "passed every condition. Against the scripted driver it is level overall: it reaches far more but "
                  "crashes about four times as often.")
    table(s, X0, 1.0, [2.62, 0.95, 1.25, 0.95, 0.85, 1.05, 0.95, 0.7], [
        [{"t": "Model (8 seeds, 6M)", "align": "l"}, "Score", "Seeds", "Reached", "Crash", "FRONT\nreached",
         "FRONT\ncrash", "Give\nup"],
        ["N_c05w0t1  (8d winner, 64)", "+0.614", "+0.42 to +0.84", "73.6%", "12.2%", "39.6%", "16.6%", "6"],
        ["O_c1w0t4  (64)", "+0.683", "+0.34 to +0.86", "76.1%", "7.8%", "50.7%", "13.0%", "3"],
        ["N_c05w0t1_n256  (256)", "+0.573", "+0.28 to +0.82", "76.8%", "19.6%", "58.4%", "32.5%", "1"],
        ["O_c1w0t4_n256  (256, adopted)", mark("+0.791"), "+0.68 to +0.84", mark("89.5%"), "10.4%", mark("76.1%"),
         "23.4%", "0"],
        [{"t": "Scripted driver", "b": True}, "+0.758", "", "78.2%", "2.4%", "62.7%", "3.9%", ""],
        [{"t": "Random actions", "b": True}, "−0.652", "", "14.0%", "79.2%", "2.6%", "88.2%", ""],
    ], row_h=0.34, size=11, header_h=0.48)
    bullets(s, X0, 3.6, W0, 1.8, [
        [("Bar, written before: ", {"bold": True}), ("gain ≥ +0.10 over the winner with p < 0.05 and ahead on ≥ 6 of 8 "
                                                    "seeds; crashes not higher beyond noise; at most 1 seed giving up", {})],
        [("Result: ", {"bold": True}), ("+0.177 (p 0.006, ahead on 8 of 8); crashes −1.8 points (p 0.21); no seed gives "
                                        "up. Adopted", {})],
        [("Against the scripted driver: ", {"bold": True}), ("+0.033 (p 0.16), level. The remaining loss is crashes, "
                                                            "mostly on FRONT", {})],
    ], size=12)
    caption(s, "Shaded: better than the scripted driver. Give up = seeds that time out on ≥ 30% of FRONT episodes.", 4.95,
            h=0.3, size=10)

    # 10. Decision trail
    s = new_slide(deck, "Decision trail: every bar was written before its runs",
                  "This is the order in which the recipe was settled. The only bar changed before its test is the 8f "
                  "crash condition, from 'not higher' to 'not higher beyond noise'; the stricter version was also met.")
    table(s, X0, 1.0, [0.75, 2.45, 3.62, 2.5], [
        ["Stage", {"t": "Question", "align": "l"}, {"t": "Result", "align": "l"}, {"t": "Decision", "align": "l"}],
        ["8a", {"t": "64-64 or 256-256? (3 seeds)", "align": "l"},
         {"t": "+0.067 (p 0.45); bar +0.10 on all seeds", "align": "l"}, {"t": "64 kept", "align": "l"}],
        ["8b", {"t": "Training order N / H / E / O", "align": "l"},
         {"t": "All within 0.12; no difference beyond noise", "align": "l"},
         {"t": "N by the rule; O best: both go on", "align": "l"}],
        ["8c", {"t": "8 rewards (crash × progress)", "align": "l"},
         {"t": "None beats success only; crash costs cause FRONT give-up", "align": "l"},
         {"t": "Time cost triggered", "align": "l"}],
        ["8c2", {"t": "Time cost 0.001 / 0.004", "align": "l"},
         {"t": "No difference beyond noise", "align": "l"}, {"t": "7 finalists", "align": "l"}],
        ["8d", {"t": "Finalists on 5 new seeds", "align": "l"},
         {"t": "Crash cost needed; crash-cost rewards all tie", "align": "l"},
         {"t": "Winner N_c05w0t1 (64)", "align": "l"}],
        ["8e", {"t": "256 wide? 3M more?", "align": "l"},
         {"t": "Bars missed; but 256 ends the FRONT give-up", "align": "l"},
         {"t": "Fresh test for 256 + order O", "align": "l"}],
        ["8f", {"t": "256 + O against the winner, fresh episodes", "align": "l"},
         {"t": "+0.177 (p 0.006, 8 of 8), no more crashes", "align": "l"},
         mark("Adopted: O_c1w0t4, 256, 6M")],
    ], row_h=0.42, size=11, header_h=0.36)
    caption(s, "What decides the result is the order and the width together: at 256 the two practice rounds end with "
               "every seed passing (86–95% of practice episodes); at 64, 3 of 8 seeds never learned to pass. Among crash "
               "costs, the reward makes no detectable difference (8e: +0.001).", 4.45, h=0.75, size=11)

    # 11. Final recipe: stages and reward
    s = new_slide(deck, "Final recipe: training stages and the exact reward",
                  "Three rounds of 2M transitions, each from the previous round's policy. The reward in rounds 2 and 3 "
                  "has exactly three terms.")
    table(s, X0, 1.0, [0.75, 2.55, 3.5, 1.05, 1.47], [
        ["Round", {"t": "Task", "align": "l"}, {"t": "Reward per decision (goal units)", "align": "l"}, "Transitions",
         "Starts from"],
        ["1", {"t": "Practice with easy starts", "align": "l"},
         {"t": "+1 reached; crash and timeout 0", "align": "l"}, "2M", "fresh weights"],
        ["2", {"t": "Practice with easy starts", "align": "l"},
         {"t": "+1 reached, −1 crash, −0.004 per decision", "align": "l"}, "2M", "round 1"],
        ["3", {"t": "Full task", "align": "l"},
         {"t": "+1 reached, −1 crash, −0.004 per decision", "align": "l"}, "2M", "round 2"],
    ], row_h=0.36, size=12, header_h=0.36)
    formula = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(X0), Inches(2.62), Inches(W0), Inches(0.5))
    formula.name = "Reward formula"
    formula.fill.solid()
    formula.fill.fore_color.rgb = RGBColor.from_string(HEAD_FILL)
    formula.line.fill.background()
    formula.shadow.inherit = False
    frame = formula.text_frame
    frame.margin_left = Inches(0.15)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    _run(p, "Rounds 2–3:   r_t = 0.02 × ( 50·[reached at t] − 50·[crash at t] − 0.2 )", 14, font="Courier New")
    bullets(s, X0, 3.25, W0, 2.2, [
        "[reached at t], [crash at t] = 1 on the decision where the episode ends that way, else 0; a crash wins a tie. "
        "A timeout (25 s) or a lost anchor adds nothing",
        "−0.004 every 0.1 s is −1 over a whole 25 s episode (−0.37 after discounting at γ 0.99)",
        "Switched off: progress shaping, the hold bonus and the inherited per-step progress",
        "Network 256-256, separate policy and value; 8 seeds (7–28); the final model of round 3 is the one tested",
    ], size=12)

    # 12. Final recipe: every setting
    s = new_slide(deck, "Final recipe: every setting",
                  "Read from the saved configuration of O3_c1w0t4_n256. The same PPO settings are used in all three "
                  "rounds; only the task, the reward and the starting weights change.")
    table(s, X0, 1.0, [2.0, 1.7], [
        [{"t": "PPO (SB3 2.9.2)", "align": "l"}, "Value"],
        ["Policy / value net", "256-256 tanh, separate"], ["Learning rate", "3e-4, constant"],
        ["Environments", "3 in parallel"], ["Steps per env per update", "256 (768)"], ["Minibatch", "256"],
        ["Epochs per update", "5"], ["Discount γ", "0.99"], ["GAE λ", "0.95"], ["Clip range", "0.2"],
        ["Entropy coefficient", "0.01"], ["Value-loss coefficient", "0.5"], ["Gradient-norm clip", "0.5"],
    ], row_h=0.3, size=10, header_h=0.3)
    table(s, 4.3, 1.0, [1.55, 3.81], [
        [{"t": "Environment", "align": "l"}, {"t": "Setting", "align": "l"}],
        ["Simulator", {"t": "highway-env 1.12.1, highway-v0, 3 lanes", "align": "l"}],
        ["Traffic", {"t": "8 cars, 12–17 m/s, IDM + MOBIL; anchor keeps its lane", "align": "l"}],
        ["Start", {"t": "ego 20 m/s; a 12 m/s car 35 m ahead in its lane", "align": "l"}],
        ["Anchor, slot", {"t": "car 30–120 m ahead; FRONT +35 m, BACK −15 m, LEFT/RIGHT beside", "align": "l"}],
        ["Decisions", {"t": "10 per s; 25 s limit (250 decisions)", "align": "l"}],
        ["Actions", {"t": "lane left / idle / lane right / faster / slower; 10–25 m/s in 2.5 m/s steps", "align": "l"}],
        ["Observation", {"t": "53 numbers: 7 nearest cars × 5, goal, anchor, slot type, slot error", "align": "l"}],
        ["Success", {"t": "ego's center within ±3 m along, ±1.5 m across, for one decision", "align": "l"}],
        ["Practice task", {"t": "blocked BACK/LEFT/RIGHT only; blocking car held in its lane", "align": "l"}],
        ["Easy starts", {"t": "next lane, 10 m past / level / 15 m behind the slow car, or normal; "
                                  "needs 8 m clear", "align": "l"}],
    ], row_h=0.36, size=10, header_h=0.3)

    # 13. Demos of the adopted model
    s = new_slide(deck, "Demo: the adopted model (seed 7)",
                  "Episode 10045 is the blocked episode shown last time: B4 crashed, the scripted driver timed out and P6 "
                  "reached it in 6.4 s. The FRONT crash is typical of what is left: full speed into a slower car.")
    demo_row(s, media / "adopted_left10045.gif", 1.09, "LEFT, blocked (episode 10045): reached, 6.3 s",
             "Passes the slow car in the slot's lane, then merges in")
    demo_row(s, media / "adopted_front10001.gif", 2.54, "FRONT (episode 10001): reached, 12.2 s",
             "Passes the anchor and merges ahead of it")
    demo_row(s, media / "adopted_front10054_crash.gif", 3.98, "FRONT (episode 10054): crash, 10.8 s",
             "Rear-ends the anchor at 25 m/s against its 15 m/s, without braking")

    # 14. Where it still loses
    s = new_slide(deck, "Where it still loses: crashes, mostly on FRONT (8g)",
                  "Diagnosis without training, on the 133 FRONT selection episodes of all 8 seeds. The failure is the "
                  "same as in B0 to B4, driving into a slower car, now moved to FRONT.")
    bullets(s, X0, 1.09, 5.47, 3.9, [
        "It reaches far more than the scripted driver and almost never gives up, but crashes about four times as often",
        "91% of its FRONT crashes are rear-ends at full speed: 25 m/s, about 12 m/s faster than the car it hits",
        "It chooses SLOWER in the last 2 s before only 2% of these crashes: it passes, but does not brake",
        "A crash cost made earlier policies wait instead; at 256 wide this one keeps trying, and the crashes remain",
    ])
    table(s, 6.07, 1.15, [1.6, 1.0, 1.0], [
        [{"t": "500 fresh episodes", "align": "l"}, "Adopted", "Scripted"],
        ["Reached", mark("89.5%"), "78.2%"], ["Crash", "10.4%", "2.4%"], ["Timeout", "0.1%", "19.4%"],
        ["FRONT reached", mark("76.1%"), "62.7%"], ["FRONT crash", "23.4%", "3.9%"],
    ], row_h=0.36, size=11, header_h=0.36)
    caption(s, "Shaded: better than the scripted driver.", 3.4, size=10, x=6.07, w=3.59)

    # 15. Step 9 setup
    s = new_slide(deck, "Consecutive goals (Step 9): setup",
                  "No training. The question is whether the single-goal model can do goals back to back when only the "
                  "goal in its observation changes. Every next goal is checked to be feasible.")
    bullets(s, X0, 1.09, 5.47, 4.2, [
        "After a slot is reached the episode goes on: only the goal inputs change, and each goal gets its own 25 s",
        "Every next goal is feasible: time needed (gap at the speed limits, plus 2 s per lane change) ≤ 15.72 s, the 95th "
        "percentile of normal goals",
        "From → to pairs the scripted driver reaches in under 20% are flagged and reported separately",
        "500 chains × 3 conditions × 8 seeds, up to 5 goals (12,000 model chains); the scripted driver drives the same "
        "chains",
        "Bar, written before: drop = goal 1 score − goals 2–5 score, paired over seeds; it breaks if ≥ 0.10 with p < 0.05",
    ], size=12)
    table(s, 6.07, 1.15, [1.15, 2.44], [
        [{"t": "Condition", "align": "l"}, {"t": "The next goal", "align": "l"}],
        ["New car", {"t": "a random slot of a different car", "align": "l"}],
        ["Same car", {"t": "another slot of the same anchor", "align": "l"}],
        ["Origin kept", {"t": "as new car, but goal coordinates keep the first goal's origin: only the goal changes",
                         "align": "l"}],
    ], row_h=0.55, size=11, header_h=0.36)

    # 16. Step 9 results figure
    s = new_slide(deck, "Consecutive goals: later goals score about 0.4 lower",
                  "Top row: score of goal k. Purple is the model, dashed black the scripted driver. Bottom row: what "
                  "happened on goal 1 and on goals 2 to 5. Goal 1 is identical to an ordinary episode, so the drop is "
                  "about what comes after a reached slot.")
    picture(s, RESULTS / "chains" / "chains.png", X0, 0.95, 6.75, 6.75 * 888 / 1440)
    bullets(s, 7.2, 1.05, 2.46, 4.2, [
        "Goal 1 is unchanged: the same outcome as an ordinary episode",
        "Later goals break in every condition, on every seed (p < 0.001)",
        "New car: the scripted driver drops too, by timing out; the model drops by crashing",
        "Same car: the scripted driver does better on later goals; the whole drop is the model's",
    ], size=11)

    # 17. Step 9 how much is the model's own
    s = new_slide(deck, "Consecutive goals: how much of the drop is the model's own",
                  "The scripted driver also scores lower after goal 1 with a new car, so later goals are harder for any "
                  "driver. Subtracting its drop leaves the model's own part. This is a rough split, because each driver's "
                  "later goals start where its own previous goal ended.")
    table(s, X0, 1.0, [1.5, 1.2, 1.35, 1.2, 1.37, 2.7], [
        [{"t": "Condition", "align": "l"}, "Model\ngoal 1", "Model\ngoals 2–5", "Model\ndrop", "Scripted\ndrop",
         "Model's own extra drop\n(95% interval)"],
        ["New car", "+0.78", "+0.37", "0.41", "0.25", mark("+0.16  (+0.13 to +0.19)")],
        ["Same car", "+0.78", "+0.28", "0.50", "−0.16", mark("+0.66  (+0.49 to +0.83)")],
        ["Origin kept", "+0.78", "+0.34", "0.43", "0.25", mark("+0.19  (+0.13 to +0.25)")],
    ], row_h=0.38, size=11, header_h=0.5)
    bullets(s, X0, 2.75, W0, 2.6, [
        "Why: it reaches a slot by driving through it, at 25 m/s, 11 m/s faster than the anchor (99.8% of successes). "
        "The scripted driver arrives 2.7 m/s faster",
        "So the next goal starts at full speed close behind slower cars: 63% of later-goal crashes come within 3 s of the "
        "switch, and after a BACK slot half hit the previous anchor (median 1.5 s)",
        "Same car: from FRONT it cannot drop back to a slot behind (9–15% reached); the scripted driver reaches "
        "FRONT → LEFT and FRONT → RIGHT every time",
        "Keeping the goal origin changes nothing (0.43 against 0.41): the inputs are fine, the driving is not",
    ], size=12)

    # 18. Step 9 demos
    s = new_slide(deck, "Demo: consecutive goals (seed 7)",
                  "Chosen after the results, by stated rules, to show the main findings. The green box jumps to the next "
                  "slot each time one is reached.")
    demo_row(s, media / "chain_new_car_70003.gif", 1.09, "New car, chain 70003: all 5 goals, 31.9 s",
             "Works when traffic leaves room after each slot")
    demo_row(s, media / "chain_new_car_70007.gif", 2.54, "New car, chain 70007: crash after goal 3",
             "Arrives at full speed, then rams the previous anchor right after the switch")
    demo_row(s, media / "chain_same_car_70004.gif", 3.98, "Same car, chain 70004: stuck after FRONT",
             "Cannot drop back to the next slot behind the anchor; times out")

    # 19. Conclusions
    s = new_slide(deck, "Conclusions and discussion",
                  "The recipe question is settled as far as this design resolves it. The next direction is for the "
                  "meeting: Step 9 shows that the single-goal skill does not compose as it is.")
    bullets(s, X0, 1.09, 5.47, 4.3, [
        "Settled recipe: practice with easy starts (success only) → practice with crash 1 + time 0.004 → full task; "
        "256-256; 6M",
        "A crash cost is needed; which crash-cost reward makes no detectable difference",
        "256 wide makes the practice rounds reliable: +0.177 over the 8d winner (64 wide) on fresh episodes, and no "
        "seed gives up",
        "Level with the scripted driver: reaches far more, crashes about 4× as often (full-speed rear-ends, mostly FRONT)",
        "Consecutive goals break: the model treats 'reached' as the end and arrives at full speed",
    ], size=12, heading="Conclusions", space_after=4)
    bullets(s, 6.01, 1.09, 3.64, 4.3, [
        "Train with goals that do not end the episode (consecutive goals)?",
        "How the sequential task should feed into DFA-conditioned driving",
        "How to make the policy brake (FRONT crashes)",
        "Requiring the slot to be held: dropped, since it only restricts an intermediate task",
    ], size=12, heading="For discussion (not decided)", space_after=4)
    caption(s, "Full details in baseline_dynamic_reach/: REPORT.md (section 5b has the exact recipe), EXPERIMENT_NOTES.md "
               "(every prediction and decision), attempts/ (one page per attempt with reward, settings, curves and demos).",
            4.85, h=0.5, size=10)


def main():
    template, media, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    deck = Presentation(str(template))
    slide_ids = deck.slides._sldIdLst
    for slide_id in list(slide_ids):
        deck.part.drop_rel(slide_id.rId)
        slide_ids.remove(slide_id)
    build(deck, media)
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(str(output))
    print(output, len(deck.slides), "slides")


if __name__ == "__main__":
    main()
