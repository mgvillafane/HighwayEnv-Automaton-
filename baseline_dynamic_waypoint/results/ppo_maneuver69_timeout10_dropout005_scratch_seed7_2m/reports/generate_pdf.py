"""Build a standalone, source-verified guide to the latest PPO observations/rewards.

Run extract_report_data.py with the training Python first. Run this script with
Python 3.13 and the existing ros-labs-study-pdf-tools ReportLab installation.
"""
from html import escape
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(tempfile.gettempdir()) / "ros-labs-study-pdf-tools"))
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Table,
    TableStyle, KeepTogether, Flowable,
)

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "report_data.json").read_text(encoding="utf-8"))
OUTPUT = HERE / "Rewards_and_Observations_Maneuver69_PPO.pdf"
W, H = 612, 792
MARGIN, WIDTH = 46, 520
INK = colors.HexColor("#172B3A")
TEAL = colors.HexColor("#087E8B")
GRAY = colors.HexColor("#536571")
LIGHT = colors.HexColor("#EEF5F6")
RULE = colors.HexColor("#D8E3E6")
BLUE = colors.HexColor("#2277B8")
AMBER = colors.HexColor("#D28D24")

for font, filename in [("Body", "segoeui.ttf"), ("Bold", "segoeuib.ttf"),
                       ("Italic", "segoeuii.ttf"), ("Code", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(font, "C:/Windows/Fonts/" + filename))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Italic", boldItalic="Bold")
STYLES = {
    "body": ParagraphStyle("BodyText", fontName="Body", fontSize=10.1, leading=14.1,
                           textColor=INK, spaceAfter=8),
    "small": ParagraphStyle("Small", fontName="Body", fontSize=8.7, leading=11.8,
                            textColor=GRAY, spaceAfter=6),
    "title": ParagraphStyle("SectionTitle", fontName="Bold", fontSize=22, leading=27,
                            textColor=INK, spaceAfter=10),
    "sub": ParagraphStyle("Subheading", fontName="Bold", fontSize=12, leading=16,
                          textColor=TEAL, spaceBefore=8, spaceAfter=7),
    "table": ParagraphStyle("TableText", fontName="Body", fontSize=8.6, leading=11.1,
                            textColor=INK),
    "thead": ParagraphStyle("TableHeading", fontName="Bold", fontSize=8.5, leading=11.0,
                            textColor=colors.white),
    "code": ParagraphStyle("Equation", fontName="Code", fontSize=9.1, leading=13.2,
                           textColor=INK, spaceAfter=0),
}
STORY, HEADINGS = [], []


def p(text, style="body"):
    STORY.append(Paragraph(text, STYLES[style]))


def sub(text):
    p(text, "sub")


def eq(*lines):
    paragraphs = [Paragraph(escape(line).replace(" ", "&#160;"), STYLES["code"]) for line in lines]
    t = Table([[paragraph] for paragraph in paragraphs], colWidths=[WIDTH - 20])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), .5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    STORY.extend([t, Spacer(1, 9)])


def table(headers, rows, widths, size=8.6):
    style = ParagraphStyle("Cell", parent=STYLES["table"], fontSize=size, leading=size + 2.7)
    content = [[Paragraph(escape(str(x)), STYLES["thead"]) for x in headers]]
    content += [[Paragraph(escape(str(x)), style) for x in row] for row in rows]
    t = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, -1), (-1, -1), .5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    STORY.extend([t, Spacer(1, 9)])


def page(title, strapline):
    if STORY:
        STORY.append(PageBreak())
    HEADINGS.append(title)
    p(f"{len(HEADINGS):02d} / {escape(strapline.upper())}", "small")
    p(escape(title), "title")


class GoalDiagram(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 170

    def draw(self):
        c = self.canv
        c.setFillColor(LIGHT)
        c.roundRect(0, 12, WIDTH, 147, 8, fill=1, stroke=0)
        c.setFont("Body", 8)
        for lane, y in [(0, 126), (1, 84), (2, 42)]:
            c.setStrokeColor(RULE)
            c.line(15, y - 18, WIDTH - 12, y - 18)
            c.setFillColor(GRAY)
            c.drawString(18, y - 2, f"lane {lane}")
        for label, x, y in [("LEFT", 290, 126), ("RIGHT", 290, 42),
                            ("BACK: -15 m", 200, 84), ("FRONT: +20 m", 410, 84)]:
            c.setStrokeColor(TEAL)
            c.setLineWidth(1.5)
            c.roundRect(x - 29, y - 12, 58, 24, 3, fill=0, stroke=1)
            c.setFillColor(TEAL)
            c.setFont("Bold", 7.6)
            c.drawCentredString(x, y + 16, label)
        for label, x, color in [("EGO", 102, BLUE), ("ANCHOR", 290, AMBER)]:
            c.setFillColor(color)
            c.roundRect(x - 14, 77, 28, 14, 3, fill=1, stroke=0)
            c.setFont("Bold", 8)
            c.drawCentredString(x, 63, label)
        c.setFont("Body", 8)
        c.setFillColor(GRAY)
        c.drawString(14, 0, "Schematic: x increases rightward; y and lane IDs increase downward.")


class BlendChart(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 143

    def draw(self):
        c = self.canv
        x0, y0, pw, ph = 42, 30, 445, 83
        fx = lambda lead: x0 + (lead + 10) / 40 * pw
        fy = lambda b: y0 + ph * b
        c.setStrokeColor(RULE)
        c.line(x0, y0, x0 + pw, y0)
        c.line(x0, y0, x0, y0 + ph)
        c.setFont("Body", 8)
        c.setFillColor(GRAY)
        for lead in [-10, 0, 10, 15, 20, 30]:
            c.line(fx(lead), y0 - 3, fx(lead), y0)
            c.drawCentredString(fx(lead), y0 - 14, str(lead))
        c.drawRightString(x0 - 7, fy(0) - 2, "0")
        c.drawRightString(x0 - 7, fy(1) - 2, "1")
        c.drawString(42, 131, "Merge fraction b")
        c.drawCentredString(x0 + pw / 2, 1, "Ego lead over anchor, in meters")
        c.setStrokeColor(TEAL)
        c.setLineWidth(2.5)
        points = [(-10, 0), (10, 0), (20, 1), (30, 1)]
        for (a, b), (d, e) in zip(points, points[1:]):
            c.line(fx(a), fy(b), fx(d), fy(e))
        c.setFont("Bold", 8)
        c.setFillColor(TEAL)
        c.drawString(fx(0) - 30, 47, "PASS LANE")
        c.drawString(fx(22), 96, "GOAL LANE")


class Report(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "SectionTitle":
            title = flowable.getPlainText()
            key = "section-" + str(self.page)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0, False)


def furniture(c, doc):
    c.saveState()
    c.setStrokeColor(TEAL)
    c.setLineWidth(2)
    c.line(MARGIN, H - 30, W - MARGIN, H - 30)
    c.setFont("Bold", 8)
    c.setFillColor(TEAL)
    c.drawString(MARGIN, H - 22, "HIGHWAYENV / EXPERIMENT REFERENCE")
    c.setFont("Body", 8)
    c.setFillColor(GRAY)
    c.drawRightString(W - MARGIN, H - 22, "15 NOVEMBER 2026")
    c.setStrokeColor(RULE)
    c.setLineWidth(.5)
    c.line(MARGIN, 34, W - MARGIN, 34)
    c.setFont("Body", 7.6)
    c.drawString(MARGIN, 21, "Maneuver69 PPO | rewards and observations | final 2M-step model")
    c.drawRightString(W - MARGIN, 21, str(doc.page))
    c.restoreState()


def f(values, digits=4):
    if isinstance(values, list):
        return "[" + ", ".join(f(v, digits) for v in values) + "]"
    return f"{values:.{digits}f}".rstrip("0").rstrip(".") if isinstance(values, float) else str(values)


def build():
    run, evaluation = DATA["run"], DATA["evaluation"]
    stats = evaluation["ppo"]
    page("Rewards and observations", "Full explanation of the completed navigation-guidance experiment")
    p("<b>What the agent sees, what it is paid for, and how the two connect.</b> This report documents the final "
      "single-traffic-car PPO experiment, including every observation field and every active reward component.")
    p("Run: <b>ppo_maneuver69_timeout10_dropout005_scratch_seed7_2m</b><br/>"
      "Profile: <b>maneuver10</b> | status: <b>trained</b> | fresh initialization | seed: <b>7</b>")
    table(["Key fact", "Saved setting"], [
        ["Policy inputs", "69 float32 scalars: 53 original features + 16 navigation features"],
        ["Task", "Reach and hold FRONT, BACK, LEFT, or RIGHT of one moving traffic car"],
        ["Timing", "10 decisions and physics steps per second; 30-second episode limit"],
        ["Training", "2,000,000 transitions; four parallel environments"],
        ["Core outcome weights", "+50 success; -50 ego collision; -10 timeout"],
        ["Policy", "Separate 256 x 256 actor/critic MLPs; dropout 0.05"],
    ], [119, 401])
    sub("Observed final evaluation")
    table(["Goal", "Successes", "Success rate", "Failures"], [
        ["FRONT", "14 / 18", "77.8%", "4 timeouts"],
        ["BACK", "27 / 27", "100%", "None"],
        ["LEFT", "33 / 33", "100%", "None"],
        ["RIGHT", "21 / 22", "95.5%", "1 timeout"],
        ["Overall", "95 / 100", "95.0%", "5 timeouts; 0 collisions"],
    ], [80, 94, 105, 241])
    p("These are deterministic evaluations on monitoring seeds 10000-10099, reused during training. "
      "They describe this final checkpoint; they are not a fresh independent test or a multi-seed robustness claim.", "small")
    sub("Reading map")
    p("Pages 2-8 explain the task and the 69 inputs. Pages 9-12 explain the reward with worked examples. "
      "Pages 13-14 describe how PPO uses the signals, interpretation limits, and source provenance.")

    page("The task and success conditions", "Geometry before neural-network inputs")
    p("There are exactly <b>two vehicles</b>: ego, which PPO controls, and an anchor traffic car that holds its "
      "lane and constant speed. Three straight lanes have centers at y = 0, 4, and 8 m, with IDs 0, 1, and 2.")
    STORY.extend([GoalDiagram(), Spacer(1, 10)])
    p("FRONT is 20 m ahead of the anchor in its lane; BACK is 15 m behind it. LEFT and RIGHT are alongside "
      "the anchor in its adjacent lane. These are <b>anchor-relative goal names</b>: a RIGHT goal can still require "
      "a LEFT action if ego starts farther right than the goal lane.")
    sub("A goal is a moving box, not a stationary point or a 5 m disk")
    eq("valid_slot = |e_long| < 5 m AND |e_goal_lat| < 1.8 m",
       "             AND |v_anchor - v_ego| <= 2 m/s")
    p("Collision and anchor loss disallow a valid slot. A successful episode needs <b>five valid 0.1-second "
      "ticks</b>, totaling 0.5 s. Up to three consecutive invalid ticks (0.3 s) preserve the count without "
      "advancing it; a fourth invalid tick resets it. Success is only declared on a valid tick. Thus valid "
      "holding time need not be uninterrupted when a short exit uses the grace period.")
    sub("Scenario sampling and episode endings")
    p("Ego starts at 20 m/s in a randomized lane. Traffic speed is sampled uniformly in 12-18 m/s. "
      "The initial longitudinal anchor gap is <b>35 m</b>. The saved distance_range [30, 40] m is an anchor "
      "eligibility interval, not a separately sampled initial gap. All four goals are sampled uniformly; "
      "the anchor lane is chosen so the selected side slot remains on the road.")
    p("Success or an ego collision terminates the episode. Reaching 30 s without either truncates it as "
      "a timeout. Anchor loss also truncates it, but receives no timeout penalty. Ego going off road alone "
      "is not configured as a terminal event.", "small")

    page("The complete input contract", "69 numbers at every policy decision")
    p("The policy receives a dictionary of arrays. Stable-Baselines3's <b>CombinedExtractor</b> flattens each "
      "array and concatenates the fields in the saved space's order. The actual model was loaded to verify "
      "the following layout; indices are zero-based.")
    meanings = {
        "achieved_goal": "Ego position in the episode coordinate frame",
        "anchor": "Traffic position and velocity relative to ego",
        "desired_goal": "Moving slot position in the episode coordinate frame",
        "goal_delta": "Desired position minus achieved position",
        "hold_progress": "Fraction of required valid holding time accumulated",
        "navigation": "16 engineered maneuver and risk features",
        "slot": "One-hot FRONT / BACK / LEFT / RIGHT",
        "slot_error": "Local longitudinal, lateral, and anchor-speed errors",
        "vehicles": "Seven rows of presence, x, y, vx, vy",
    }
    table(["Field", "Shape", "Flat indices", "Meaning"], [
        [r["key"], " x ".join(map(str, r["shape"])), f"{r['start']}-{r['end']}", meanings[r["key"]]]
        for r in DATA["contract"]
    ], [105, 48, 64, 303])
    eq("35 vehicles + 6 position-goal values + 4 anchor values",
       "+ 4 slot bits + 1 hold value + 3 local errors = 53",
       "53 original values + 16 navigation values = 69")
    sub("Three kinds of normalization")
    p("<b>Kinematics and anchor:</b> signed values scaled and clipped to [-1, 1]. <b>Goal positions:</b> "
      "divided by [200 m, 12 m], with no clipping. <b>Navigation and local errors:</b> feature-specific "
      "scales followed by clipping to [-1, 1]. Presence, goal one-hot bits, availability bits, risk, and "
      "hold progress keep their natural ranges inside these bounds.")
    p("Normalization makes numerical magnitudes manageable. It does not turn a quantity into a probability. "
      "The actor sees one current observation; this run has no frame stacking or recurrent memory.", "small")

    page("Vehicle kinematics: 35 inputs", "vehicles[7, 5]")
    p("Every row has the fixed feature order <b>[presence, x, y, vx, vy]</b>. Presence is 1 for a represented "
      "vehicle and 0 for padding; it is not transformed into -1 for an empty row.")
    table(["Row", "Coordinates and velocities"], [
        ["0: ego", "x is episode-relative longitudinal progress; y and velocity components are ego's world-frame values."],
        ["1: traffic", "When within perception range, x, y, vx, vy are traffic minus ego. Traffic behind ego is visible."],
        ["2-6: unused", "All-zero padding in this two-car task when the anchor is represented."],
    ], [105, 415])
    table(["Feature", "Physical normalization range", "Actual mapping"], [
        ["presence", "0 or 1", "Unchanged"],
        ["x", "-200 to +200 m", "clip(x / 200, -1, 1)"],
        ["y", "-12 to +12 m", "clip(y / 12, -1, 1)"],
        ["vx", "-30 to +30 m/s", "clip(vx / 30, -1, 1)"],
        ["vy", "-10 to +10 m/s", "clip(vy / 10, -1, 1)"],
    ], [73, 188, 259])
    sub("The ego row is deliberately different")
    p("Native Kinematics would keep ego x as an absolute road coordinate. The wrapper replaces that "
      "one entry with (ego_x - reset_ego_x) / 200 to avoid saturating it just because the road starts far "
      "from zero. Ego y remains absolute: lane centers encode as 0, 1/3, and 2/3. Ego vx is an absolute "
      "velocity component; the traffic row's vx is a relative velocity component.")
    sub("Visibility and ordering")
    p("The encoder selects nearby traffic within a 200 m Euclidean perception radius, ordered by absolute "
      "longitudinal lane distance. It includes cars behind ego and excludes obstacles. Only one traffic "
      "car exists here. If it leaves perception range, its vehicle row is absent and there are six padded "
      "rows; the wrapper's explicit anchor features can still describe that tracked car, subject to clipping.")
    p("Example: a traffic row x = 0.175 means 35 m ahead; vx = -0.1667 means 5 m/s slower than ego. "
      "A zero relative position is not an absent car when presence = 1.", "small")

    page("Goal, anchor, slot, and hold", "The other 18 original inputs")
    p("Let p_e be ego's position, p_a the anchor's position, and p_g the moving goal center. "
      "The episode origin is o = [ego_x at reset, 0]; the y origin is the road's y = 0, not the ego's reset lane.")
    eq("achieved_goal = (p_e - o) / [200, 12]",
       "desired_goal  = (p_g - o) / [200, 12]",
       "goal_delta    = (p_g - p_e) / [200, 12]")
    p("Each position field has two entries. Desired and achieved positions are <b>not clipped</b>, so they "
      "can exceed 1. achieved_goal names the measured ego position; it does not mean the task is complete. "
      "Although goal_delta is mathematically redundant, it gives the network direction and distance directly.")
    sub("anchor: four relative kinematic values")
    eq("anchor = clip([(x_a-x_e)/200, (y_a-y_e)/12,",
       "               (vx_a-vx_e)/30, (vy_a-vy_e)/10], -1, 1)")
    p("The anchor block provides an explicit tracked reference independent of the seven-row encoder. "
      "It uses velocity components, whereas local speed-error features below use scalar vehicle speeds.")
    sub("slot: four task-identification bits")
    table(["FRONT", "BACK", "LEFT", "RIGHT"], [["[1, 0, 0, 0]", "[0, 1, 0, 0]", "[0, 0, 1, 0]", "[0, 0, 0, 1]"]], [130]*4)
    sub("slot_error: three local values with finer scales")
    eq("slot_error = clip([e_long/10, e_goal_lat/4,",
       "                   (v_anchor-v_ego)/10], -1, 1)")
    p("The errors are <b>slot minus ego</b> in the goal lane's coordinates. Positive longitudinal error "
      "means the goal is ahead; positive lateral error means it lies to the right on this straight road. "
      "The small 10 m scale exposes the 5 m success boundary as 0.5, but errors beyond 10 m saturate. "
      "Navigation adds a wider 60 m longitudinal scale to preserve more approach information.")
    eq("hold_progress = min(1, valid_hold_ticks / 5)")
    p("This value advances by 0.2 on each valid tick, is preserved during short exits, and resets after "
      "a long exit or collision/anchor loss. The separate consecutive-exit counter is not an input.", "small")

    page("All 16 navigation features", "navigation[16], in exact saved order")
    p("All entries are float32 and clipped to [-1, 1]. Lane differences are signed counts divided by two "
      "because the road has three lanes. b is the FRONT merge fraction explained on the next page.")
    nav_rows = [
        ["0", "anchor_lane_minus_ego_lane", "(anchor lane - ego lane) / 2"],
        ["1", "goal_lane_minus_ego_lane", "(goal lane - ego lane) / 2"],
        ["2", "requested_lane_minus_ego_lane", "(ego target lane - ego lane) / 2"],
        ["3", "requested_lane_lateral_error", "Signed error to ego's requested lane center / 4 m"],
        ["4", "anchor_gap", "(anchor longitudinal position - ego position) / 60 m"],
        ["5", "anchor_minus_ego_speed", "(anchor scalar speed - ego speed) / 10 m/s"],
        ["6", "slot_longitudinal_error", "e_long / 60 m; wider approach scale"],
        ["7", "navigation_lateral_error", "e_nav_lat / 8 m; passing target for FRONT, goal otherwise"],
        ["8", "preferred_minus_ego_speed", "(preferred speed - ego speed) / 10 m/s"],
        ["9", "preferred_speed", "Preferred speed / 25 m/s (range 0.4 to 1)"],
        ["10", "front_merge_fraction", "b for FRONT; 0 for the other goals"],
        ["11", "front_pass_fraction", "1 - b for FRONT; 0 for the other goals"],
        ["12", "front_merge_clearance", "(ego lead - 10 m) / 60 m for FRONT; 0 otherwise"],
        ["13", "left_lane_available", "1 if ego lane ID > 0; otherwise 0"],
        ["14", "right_lane_available", "1 if ego lane ID < 2; otherwise 0"],
        ["15", "collision_risk", "Closing-risk heuristic in [0, 1]; see page 8"],
    ]
    table(["Index", "Exact feature name", "Value before final clipping"], nav_rows, [37, 192, 291], size=8.2)
    p("Requested lane is the native controller's current target lane, not a new recommendation. "
      "Availability bits describe road boundaries, not whether another car blocks a lane. "
      "front_pass_fraction and front_merge_fraction describe geometric guidance, not completed "
      "maneuver percentages measured from an action history.", "small")

    page("Passing target and preferred speed", "Guidance features, not replacement actions")
    p("For FRONT, define lead = ego longitudinal position - anchor longitudinal position. "
      "A negative lead means ego is behind. The implementation finds the nearest valid lane beside "
      "the anchor, using distance from ego to that lane center. A distance tie chooses the lower lane ID "
      "because adjacent lanes are enumerated left first.")
    eq("b = clip((lead - 10) / (20 - 10), 0, 1)",
       "e_nav_lat = (1-b)*e_pass_lane + b*e_goal_lat")
    STORY.extend([BlendChart(), Spacer(1, 7)])
    p("At a lead of 10 m or less, the target is the passing lane. At 15 m, it lies halfway toward the "
      "goal lane. At 20 m or more, it is the goal lane. The interpolated target may lie between lane "
      "centers; it is a feature and reward target, not a forced steering command. For BACK, LEFT, and "
      "RIGHT, e_nav_lat always equals e_goal_lat.")
    sub("Preferred speed changes from catching up to matching")
    eq("v_pref = clip(v_anchor + clip(0.25*e_long, -5, 5),",
       "              10, 25)")
    table(["Longitudinal error", "Traffic speed", "Preferred speed"], [
        ["-30 m: ego beyond slot", "15 m/s", "10 m/s"],
        ["-10 m", "15 m/s", "12.5 m/s"],
        ["0 m: aligned along road", "15 m/s", "15 m/s"],
        ["+10 m", "15 m/s", "17.5 m/s"],
        ["+30 m: slot ahead", "15 m/s", "20 m/s"],
    ], [220, 130, 170])
    p("PPO chooses among the original five actions. Native speed targets are discrete; the preferred speed "
      "can be continuous and is never directly written into ego's controller by this guidance.", "small")

    page("Collision risk and interpreting scales", "A deterministic heuristic, not a probability")
    p("Both vehicles are 5 m long and 2 m wide in the inspected simulator. Let g be the center-to-center "
      "longitudinal anchor gap, c = v_ego - v_anchor the closing speed, and a the ego lateral offset from "
      "the anchor lane center. Risk is computed only when <b>g &gt; 0, c &gt; 0, and |a| &lt; 2 m</b>.")
    eq("bumper_gap = max(0, g - (5+5)/2) = max(0, g-5)",
       "TTC = bumper_gap / closing_speed",
       "risk = clip(1 - TTC/5 seconds, 0, 1)")
    p("Risk is zero when that gate is false, including an anchor behind ego, no closing motion, or enough "
      "lateral separation. It is also zero for TTC of five seconds or more. The calculation is not a full "
      "collision predictor: it ignores rear-closing situations, other traffic, curved-road interactions, "
      "and future steering.")
    table(["Illustrative case", "Calculation", "Risk"], [
        ["10 m center gap; closing 5 m/s; same path", "Bumper gap 5 m; TTC 1 s", "0.8"],
        ["35 m center gap; closing 5 m/s; same path", "Bumper gap 30 m; TTC 6 s", "0"],
        ["10 m gap; equal speeds", "No closing speed; gate is false", "0"],
        ["10 m gap; closing 5 m/s; adjacent lane", "Lateral separation 4 m; gate is false", "0"],
    ], [224, 240, 56])
    sub("Why one physical value appears on different scales")
    table(["Physical quantity", "Encoding", "What the policy sees"], [
        ["35 m longitudinal slot error", "goal_delta x / 200", "0.175; not clipped"],
        ["Same 35 m slot error", "slot_error[0] / 10", "1.0 after clipping"],
        ["Same 35 m slot error", "navigation[6] / 60", "0.5833"],
        ["4 m lateral slot error", "slot_error[1] / 4", "1.0"],
        ["4 m navigation lateral error", "navigation[7] / 8", "0.5"],
    ], [190, 180, 150])
    p("Clipping limits the observation, not the raw error used for route reward. The navigation scales "
      "make direction and approach distance explicit while retaining the original fine local scale. "
      "Different encodings of the same geometry are intentional redundancy.", "small")

    page("The complete environment reward", "Every active term, evaluated each 0.1-second step")
    eq("r_t = 50*S_t - 50*C_t - 10*T_t - 0.05 + 0.2*I_t",
       "      + 0.1*(d_prev-d_now)",
       "      + 0.5*(E_prev-E_now) - 0.05*A_now")
    p("S is successful completion of the hold; C is an ego collision; T is a timeout; I is valid slot "
      "occupancy on the current tick. d is Euclidean distance to the moving goal center. E is the route "
      "error score and A the bounded tracking error defined on page 10.")
    table(["Logged reward component", "Contribution", "When / interpretation"], [
        ["goal", "+50 x S", "Once, on successful termination"],
        ["collision", "-50 x C", "Once, on an ego collision"],
        ["timeout", "-10 x T", "Once at the 30 s deadline, if no success/collision; not on anchor loss"],
        ["time", "-0.05", "Every tick, including the terminal tick; -0.5 per simulated second"],
        ["hold", "+0.2 x I", "Every valid slot tick, including ticks before success"],
        ["progress", "+0.1 x (d_prev-d_now)", "Closer to the moving point is positive; farther away is negative"],
        ["route_progress", "+0.5 x (E_prev-E_now)", "Reduction in weighted navigation and speed errors"],
        ["tracking", "-0.05 x A_now", "A small nonpositive cost for remaining errors and risk"],
    ], [133, 150, 237], size=8.4)
    p("Saved weights named <b>lane</b>, <b>speed_match</b>, and <b>overtake</b> are all zero in this run. "
      "Those older reward components contribute nothing; the new route_progress and tracking terms "
      "supply the navigation guidance instead. There is no bonus simply for issuing RIGHT or LEFT.")
    sub("Order of operations")
    p("The environment records the previous route score, executes one native action/physics step, moves "
      "the slot with the anchor, updates collision/hold/termination flags, computes the base components, "
      "then adds the new route and tracking components. The returned reward equals the sum of all "
      "logged components. Native highway-v0 reward is discarded.", "small")

    page("Route progress and tracking in detail", "Raw errors drive reward; normalized features drive the network")
    eq("e_speed = v_pref - v_ego",
       "E = |e_long| + 2*|e_nav_lat| + 1*|e_speed|",
       "route_progress = 0.5*(E_prev - E_now)")
    p("E is a weighted error score, not the literal distance along a planned trajectory. It combines "
      "meters and speed error using hand-selected coefficients (the speed coefficient effectively "
      "converts its contribution into the score's distance scale). It uses <b>unclipped raw values</b>. "
      "A one-point decrease in E earns +0.5; a one-point increase costs -0.5.")
    p("For FRONT, lateral error is initially measured to a passing lane, so a useful detour can earn "
      "progress even though the final goal lane is the anchor's lane. As the merge fraction changes, "
      "the target and preferred speed also change; reward can change even with the same lane or speed.")
    eq("L = min(1, |e_nav_lat| / 8)",
       "D = min(1, |e_long| / 60)",
       "V = min(1, |e_speed| / 10)",
       "A = 0.4*L + 0.25*D + 0.25*V + 0.1*risk",
       "tracking = -0.05*A, with 0 <= A <= 1")
    table(["Part of tracking cost", "Largest magnitude per tick"], [
        ["Navigation lateral error", "0.0200"], ["Longitudinal slot error", "0.0125"],
        ["Preferred-speed error", "0.0125"], ["Collision risk", "0.0050"],
        ["Combined", "0.0500 (always zero or negative)"],
    ], [305, 215])
    p("Across 300 ticks, tracking alone lies between -15 and 0. This bound does not bound the entire "
      "reward, whose progress and terminal terms are separate. Tracking does not repeatedly pay for "
      "sitting near a target; hold reward is a separate positive occupancy signal.")
    p("Without discounting, route differences telescope to 0.5(E_initial - E_final). PPO uses gamma = 0.99, "
      "so the timing of progress matters. This is not the gamma-adjusted potential formula and carries "
      "no claim of policy invariance. Changed shaped returns should not be compared as though they "
      "were the previous experiment's reward.", "small")

    page("Worked FRONT observation", "A constructed state checked in the actual simulator")
    ex = DATA["examples"]["front"]
    obs = ex["initial_observation"]
    p("Ego is at (100, 4) m, traveling at 20 m/s. The anchor is at (135, 4) m, traveling at 15 m/s. "
      "Both occupy lane 1, and ego's requested lane is 1. Set the episode origin to (100, 0). The FRONT "
      "goal center is (155, 4): 55 m ahead of ego. These are explanatory placements, not a selected "
      "training or evaluation episode.")
    table(["Field", "Actual normalized value"], [
        ["achieved_goal", f(obs["achieved_goal"])],
        ["desired_goal", f(obs["desired_goal"])],
        ["goal_delta", f(obs["goal_delta"])],
        ["anchor", f(obs["anchor"])],
        ["slot", f(obs["slot"])],
        ["hold_progress", f(obs["hold_progress"])],
        ["slot_error", f(obs["slot_error"])],
        ["vehicles row 0: ego", f(obs["vehicles"][0])],
        ["vehicles row 1: anchor", f(obs["vehicles"][1])],
        ["vehicles rows 2-6", "Five rows of [0, 0, 0, 0, 0]"],
    ], [174, 346])
    sub("The added navigation vector")
    vec = obs["navigation"]
    eq("indices  0- 3: " + f(vec[:4]),
       "indices  4- 7: " + f(vec[4:8]),
       "indices  8-11: " + f(vec[8:12]),
       "indices 12-15: " + f(vec[12:]))
    p("The goal is in ego's lane, so navigation[1] = 0. But FRONT first needs a pass: the tied adjacent "
      "lanes resolve to lane 0, making navigation lateral error -4 m and navigation[7] = -0.5. "
      "The merge fraction is 0 and pass fraction is 1. Anchor gap is 35/60 = 0.5833; the FRONT slot "
      "error is 55/60 = 0.9167. The local slot error has already saturated at 1.")
    p("Preferred speed is 15 + clip(0.25 x 55, -5, 5) = 20 m/s, so preferred-speed error is zero. "
      "The bumper gap is 30 m and closing speed is 5 m/s: TTC = 6 s, giving risk = 0.", "small")

    page("Worked step and terminal rewards", "Arithmetic reconciled with returned simulator rewards")
    ex = DATA["examples"]["front"]
    p("Apply IDLE to the FRONT example for one 0.1 s tick. On a straight lane with the existing speed "
      "targets, ego advances 2 m and the anchor 1.5 m. The moving goal also advances 1.5 m. Longitudinal "
      "slot error falls from 55 to 54.5 m; the passing-lane error stays -4 m and speed error stays zero.")
    eq("d_prev = 55; d_now = 54.5",
       "E_prev = 55 + 2*4 + 0 = 63",
       "E_now  = 54.5 + 2*4 + 0 = 62.5",
       "A_now  = 0.4*(4/8) + 0.25*(54.5/60)",
       "       = 0.4270833333")
    names = ["goal", "collision", "timeout", "hold", "time", "progress", "route_progress", "tracking"]
    table(["Component", "Returned value"], [[key, f(ex["components"][key], 9)] for key in names]
          + [["TOTAL", f(ex["reward"], 9)]], [270, 250])
    p("The extra straight-line progress does not guarantee a safe overtake. The passing target, risk "
      "feature, later costs, terminal collision penalty, and PPO's trajectory optimization jointly "
      "influence the sequence of actions.", "small")
    sub("Holding and episode endings")
    p("If ego is exactly at the FRONT slot center, both cars move at 15 m/s, and it stays aligned, then "
      "d = E = A = 0. Each of the first four valid holding ticks returns 0.2 - 0.05 = <b>0.15</b>. "
      "The fifth adds +50, returning <b>50.15</b>. Those five ticks sum to <b>50.75</b>.")
    p("An ego collision adds -50; a timeout adds -10. Neither value is necessarily the whole terminal "
      "reward: time, progress, route, and tracking still contribute on that tick. Success/collision "
      "takes precedence over a simultaneous deadline. Anchor loss alone receives no -10 timeout charge.")

    page("How PPO uses these signals", "Model settings and interpretation limits")
    p("The 69 values enter separate actor and critic MLPs. The actor outputs a categorical distribution "
      "over five actions; the critic estimates discounted future return. Both networks have two 256-unit "
      "Tanh hidden layers, with 0.05 dropout after each hidden activation during optimization.")
    table(["Action ID", "Meaning", "Native effect"], [
        ["0", "LEFT", "Request the adjacent lower-ID lane"],
        ["1", "IDLE", "Keep the current requested lane and target speed"],
        ["2", "RIGHT", "Request the adjacent higher-ID lane"],
        ["3 / 4", "FASTER / SLOWER", "Choose one grid step above/below the current speed index"],
    ], [68, 125, 327])
    p("The target-speed grid is [10, 12.5, 15, 17.5, 20, 22.5, 25] m/s. A native low-level controller "
      "handles acceleration and steering. PPO makes a new choice every 0.1 s; lane changes are not "
      "held as multi-tick committed actions in this run.", "small")
    table(["PPO setting", "Value"], [
        ["Learning rate; optimizer", "0.0001 constant; Adam"],
        ["Discount; GAE lambda", "gamma 0.99 per decision; lambda 0.95"],
        ["Rollout", "4 environments x 250 steps = 1,000 transitions"],
        ["Updates", "batch size 250; 5 epochs per rollout"],
        ["Clipping and losses", "clip 0.2; value coefficient 0.5; entropy coefficient 0.01"],
        ["Other", "gradient norm cap 0.5; normalized advantages; no target KL or value clipping"],
    ], [173, 347])
    p("Environment reward is separate from PPO's value loss and entropy regularization. At gamma .99, "
      "an outcome 100 decisions (10 s) ahead carries a factor of about 0.366. Evaluation uses deterministic "
      "actions and disables dropout. Logged episode reward is the undiscounted sum of environment rewards.")
    p("<b>Limits:</b> the feature vector is not a complete simulator state: for example, native target speed "
      "and the consecutive hold-exit counter are not directly encoded. Both observations and rewards "
      "changed in this experiment, so the result does not isolate which addition caused improvement. "
      "One seed and reused monitoring cases do not establish generalization to more traffic or other roads.", "small")

    page("Implementation and provenance", "How this explanation was verified")
    p("The saved final model, run.json, evaluation.json, installed encoder, and project implementation were "
      "read directly. The saved policy confirms 69 inputs and the dictionary ordering on page 3. The "
      "worked FRONT and RIGHT states were constructed in the actual simulator; the step reward was "
      "checked against the sum of returned components.")
    refs = DATA["references"]
    source_specs = [
        ("GoalHighwayEnv._observation", "Episode coordinates, ego x override, goal fields"),
        ("DynamicGoalHighwayEnv._observation", "Anchor, one-hot slot, local errors, hold progress"),
        ("DynamicGoalHighwayEnv.step", "Base reward, holding, terminal flags and timeout rules"),
        ("SingleCarWaypointEnv.in_slot", "Scalar-speed tolerance added to geometric slot check"),
        ("SingleCarWaypointEnv._generate_goal", "Uniform goals and valid randomized anchor lanes"),
        ("ManeuverGuidanceEnv.navigation_state", "Passing target, preferred speed, risk, route and tracking"),
        ("ManeuverGuidanceEnv._observation", "Exact 16-feature ordering and normalization"),
        ("ManeuverGuidanceEnv.step", "Additional reward and component accounting"),
    ]
    table(["Source function and line", "Responsibility"], [
        [name + "\n" + refs[name]["path"] + ":" + str(refs[name]["line"]), explanation]
        for name, explanation in source_specs
    ], [308, 212], size=8.0)
    p("All eight snapshotted dynamic-waypoint source files match the copies saved when the experiment "
      "was launched. Full SHA-256 hashes, numerical examples, policy contract, and inspected package "
      "versions are preserved in <b>report_data.json</b> alongside this PDF.", "small")
    p("<b>Final model SHA-256</b><br/>" + DATA["model_sha256"], "small")
    sub("Companion files")
    p("<b>run.json</b> stores configuration; <b>evaluation.json</b> and <b>evaluation_episodes.csv</b> store "
      "the final 100 monitoring outcomes. <b>training_episodes.csv</b> separates reward_route_progress "
      "and reward_tracking from the original components. The videos folder contains verified replays "
      "of all four final-evaluation FRONT timeouts.", "small")
    p("To reproduce the report, run extract_report_data.py with the training environment, then "
      "generate_pdf.py with Python 3.13 and the existing ReportLab tools. This generates a document; "
      "it does not train or alter the saved policy.", "small")

    doc = Report(str(OUTPUT), pagesize=(W, H), leftMargin=MARGIN, rightMargin=MARGIN,
                 topMargin=47, bottomMargin=46, title="Rewards and Observations: Maneuver69 PPO",
                 author="HighwayEnv-Automaton experiment documentation",
                 subject="Complete observation contract, navigation guidance and reward equations")
    frame = Frame(MARGIN, 46, WIDTH, H-93, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0, id="body")
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=furniture))
    doc.build(STORY)
    (HERE / "report_sections.json").write_text(json.dumps(HEADINGS, indent=2) + "\n", encoding="utf-8")
    print(str(OUTPUT))


if __name__ == "__main__":
    build()
