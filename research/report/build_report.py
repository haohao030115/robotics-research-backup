from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepInFrame, Flowable, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor
from pathlib import Path

OUT = Path('/home/group2/research/report/Group2_Beyond_Motion_Generation_Final.pdf')

# Fonts available on the host; embedding a TrueType family keeps the result portable.
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DejaVu', str(FONT_DIR/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', str(FONT_DIR/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Oblique', '/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf'))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu-Oblique')

NAVY = HexColor('#15364A')
TEAL = HexColor('#137C80')
ORANGE = HexColor('#D86F3C')
INK = HexColor('#1D2730')
MUTED = HexColor('#53636D')
PALE = HexColor('#EEF4F5')
PALE2 = HexColor('#F6F8F8')
RULE = HexColor('#C8D5D9')
GREEN = HexColor('#2F8A66')
RED = HexColor('#B34E4E')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body2', fontName='DejaVu', fontSize=9.05, leading=11.25,
    textColor=INK, spaceAfter=4.2, alignment=TA_LEFT))
styles.add(ParagraphStyle(name='Small', fontName='DejaVu', fontSize=7.8, leading=9.2,
    textColor=INK, spaceAfter=2.4))
styles.add(ParagraphStyle(name='Tiny', fontName='DejaVu', fontSize=7.15, leading=8.5,
    textColor=INK, spaceAfter=1.3))
styles.add(ParagraphStyle(name='PageTitle', fontName='DejaVu-Bold', fontSize=16.2, leading=18.2,
    textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle(name='Section', fontName='DejaVu-Bold', fontSize=11.2, leading=13.2,
    textColor=NAVY, spaceBefore=5, spaceAfter=3))
styles.add(ParagraphStyle(name='Sub', fontName='DejaVu-Bold', fontSize=9.4, leading=11,
    textColor=TEAL, spaceBefore=3, spaceAfter=2))
styles.add(ParagraphStyle(name='Caption', fontName='DejaVu', fontSize=7.35, leading=8.8,
    textColor=MUTED, spaceBefore=2, spaceAfter=4, alignment=TA_LEFT))
styles.add(ParagraphStyle(name='Table', fontName='DejaVu', fontSize=7.15, leading=8.45,
    textColor=INK))
styles.add(ParagraphStyle(name='TableHead', fontName='DejaVu-Bold', fontSize=7.15, leading=8.45,
    textColor=colors.white, alignment=TA_LEFT))
styles.add(ParagraphStyle(name='CoverTitle', fontName='DejaVu-Bold', fontSize=26, leading=29,
    textColor=NAVY, alignment=TA_LEFT, spaceAfter=4))
styles.add(ParagraphStyle(name='Subtitle', fontName='DejaVu', fontSize=14.8, leading=18,
    textColor=TEAL, alignment=TA_LEFT, spaceAfter=16))
styles.add(ParagraphStyle(name='Lead', fontName='DejaVu', fontSize=11.0, leading=14.1,
    textColor=INK, spaceAfter=8))
styles.add(ParagraphStyle(name='Box', fontName='DejaVu', fontSize=7.55, leading=9.1,
    textColor=INK, spaceAfter=0))
styles.add(ParagraphStyle(name='Ref', fontName='DejaVu', fontSize=7.2, leading=8.6,
    textColor=INK, leftIndent=10, firstLineIndent=-10, spaceAfter=4))

def P(text, style='Body2'):
    return Paragraph(text, styles[style])

def section(text):
    return Paragraph(text, styles['Section'])

def sub(text):
    return Paragraph(text, styles['Sub'])

def caption(text):
    return Paragraph(text, styles['Caption'])

def box(text, bg=PALE, border=RULE, pad=6):
    t = Table([[P(text, 'Box')]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),bg), ('BOX',(0,0),(-1,-1),0.6,border),
        ('LEFTPADDING',(0,0),(-1,-1),pad), ('RIGHTPADDING',(0,0),(-1,-1),pad),
        ('TOPPADDING',(0,0),(-1,-1),pad), ('BOTTOMPADDING',(0,0),(-1,-1),pad),
    ]))
    return t

def tbl(data, widths, header=True, tiny=False, repeatRows=1):
    sty = 'Table' if not tiny else 'Tiny'
    rows=[]
    for ri,row in enumerate(data):
        rows.append([Paragraph(str(x), styles['TableHead' if (header and ri==0) else sty]) for x in row])
    t=Table(rows, colWidths=widths, repeatRows=repeatRows if header else 0, hAlign='LEFT')
    commands=[('GRID',(0,0),(-1,-1),0.35,RULE), ('VALIGN',(0,0),(-1,-1),'TOP'),
              ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
              ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
    if header:
        commands += [('BACKGROUND',(0,0),(-1,0),NAVY), ('TEXTCOLOR',(0,0),(-1,0),colors.white)]
        for i in range(1,len(rows)):
            if i%2==0: commands.append(('BACKGROUND',(0,i),(-1,i),PALE2))
    t.setStyle(TableStyle(commands))
    return t

class Diagram(Flowable):
    def __init__(self, kind, width=450, height=118):
        Flowable.__init__(self); self.kind=kind; self.width=width; self.height=height
    def draw(self):
        d=Drawing(self.width,self.height)
        if self.kind=='overview':
            boxes=[('Text +\nspatial plan',8,57,72,43,ORANGE),('Kimodo / ARDY\nreference',106,57,82,43,TEAL),('Adapter +\nG1 retarget',206,57,82,43,NAVY),('SONIC body\n50 Hz / 29 DoF',306,57,88,43,TEAL),('MuJoCo +\nHand PD',422,57,80,43,ORANGE)]
            for label,x,y,w,h,c in boxes:
                d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=c,strokeColor=c))
                for j,line in enumerate(label.split('\n')):
                    d.add(String(x+w/2,y+h/2+7-j*11,line,fontName='DejaVu-Bold',fontSize=7.2,fillColor=colors.white,textAnchor='middle'))
            for x in [80,188,288,404]:
                d.add(Line(x,78,x+20,78,strokeColor=MUTED,strokeWidth=1.2))
                d.add(Polygon([x+20,78,x+14,82,x+14,74],fillColor=MUTED,strokeColor=MUTED))
            d.add(Rect(119,8,292,29,rx=5,ry=5,fillColor=PALE,strokeColor=RULE))
            d.add(String(265,25,'Canonical Data Factory: trace, forensic join, qualification, horizons',fontName='DejaVu-Bold',fontSize=7.2,fillColor=NAVY,textAnchor='middle'))
            d.add(Line(265,57,265,37,strokeColor=TEAL,strokeWidth=1.2)); d.add(Polygon([265,37,261,43,269,43],fillColor=TEAL,strokeColor=TEAL))
            d.add(Line(411,22,454,22,strokeColor=TEAL,strokeWidth=1.0)); d.add(Polygon([454,22,448,26,448,18],fillColor=TEAL,strokeColor=TEAL))
            d.add(String(462,18,'VA bridge',fontName='DejaVu',fontSize=6.6,fillColor=MUTED))
        elif self.kind=='control':
            # two lanes: planned reference and executed feedback
            d.add(String(10,102,'PLANNED REFERENCE',fontName='DejaVu-Bold',fontSize=7.3,fillColor=TEAL))
            d.add(String(10,29,'EXECUTED STATE / FEEDBACK',fontName='DejaVu-Bold',fontSize=7.3,fillColor=ORANGE))
            top=[('body q,dq\nroot pose',10,62,82,30,NAVY),('encode\n1762 -> 64',120,62,82,30,TEAL),('decode\n994 -> 29',230,62,82,30,TEAL),('SONIC\ntracking',340,62,82,30,NAVY),('Hand PD\n14 DoF',450,62,68,30,ORANGE)]
            for label,x,y,w,h,c in top:
                d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=c,strokeColor=c))
                for j,line in enumerate(label.split('\n')): d.add(String(x+w/2,y+h/2+5-j*10,line,fontName='DejaVu-Bold',fontSize=6.7,fillColor=colors.white,textAnchor='middle'))
            for x in [92,202,312,422]:
                d.add(Line(x,77,x+28,77,strokeColor=MUTED,strokeWidth=1)); d.add(Polygon([x+28,77,x+22,81,x+22,73],fillColor=MUTED,strokeColor=MUTED))
            bot=[('MuJoCo\n200 Hz',90,8,100,30,ORANGE),('G1 state +\ncontacts',220,8,100,30,NAVY),('RGB 10 Hz\n+ proprio',350,8,100,30,TEAL)]
            for label,x,y,w,h,c in bot:
                d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=c,strokeColor=c))
                for j,line in enumerate(label.split('\n')): d.add(String(x+w/2,y+h/2+5-j*10,line,fontName='DejaVu-Bold',fontSize=6.7,fillColor=colors.white,textAnchor='middle'))
            d.add(Line(390,62,390,39,strokeColor=ORANGE,strokeWidth=1)); d.add(Polygon([390,39,386,45,394,45],fillColor=ORANGE,strokeColor=ORANGE))
            d.add(Line(270,38,270,62,strokeColor=ORANGE,strokeWidth=1)); d.add(Polygon([270,62,266,56,274,56],fillColor=ORANGE,strokeColor=ORANGE))
        elif self.kind=='factory':
            labels=[('reference\n+ provenance',10,46,82,40,TEAL),('native replay\n50 Hz ticks',116,46,82,40,NAVY),('RGB / state /\ntokens / hands',222,46,94,40,ORANGE),('forensic join\nby IDs',340,46,82,40,TEAL),('episode\n+ horizons',446,46,74,40,NAVY)]
            for label,x,y,w,h,c in labels:
                d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=c,strokeColor=c))
                for j,line in enumerate(label.split('\n')): d.add(String(x+w/2,y+h/2+6-j*11,line,fontName='DejaVu-Bold',fontSize=6.8,fillColor=colors.white,textAnchor='middle'))
            for x in [92,198,316,422]:
                d.add(Line(x,66,x+24,66,strokeColor=MUTED,strokeWidth=1)); d.add(Polygon([x+24,66,x+18,70,x+18,62],fillColor=MUTED,strokeColor=MUTED))
            d.add(String(10,27,'planning time != native control time != post-step observation time',fontName='DejaVu-Bold',fontSize=7.2,fillColor=RED))
            d.add(String(10,13,'IDs: sample_id | observation_id | token_id | hand_ref_id; missing rows fail closed',fontName='DejaVu',fontSize=6.7,fillColor=MUTED))
        elif self.kind=='va':
            d.add(Rect(8,43,95,42,rx=4,ry=4,fillColor=ORANGE,strokeColor=ORANGE)); d.add(String(55,66,'RGB 3x128x128',fontName='DejaVu-Bold',fontSize=7,fillColor=colors.white,textAnchor='middle')); d.add(String(55,54,'4 conv blocks',fontName='DejaVu',fontSize=6.4,fillColor=colors.white,textAnchor='middle'))
            d.add(Rect(8,5,95,25,rx=4,ry=4,fillColor=TEAL,strokeColor=TEAL)); d.add(String(55,20,'body 68 + hand 28',fontName='DejaVu-Bold',fontSize=6.8,fillColor=colors.white,textAnchor='middle'))
            d.add(Rect(143,31,115,54,rx=4,ry=4,fillColor=NAVY,strokeColor=NAVY)); d.add(String(200,69,'64 visual tokens x 256',fontName='DejaVu-Bold',fontSize=7,fillColor=colors.white,textAnchor='middle')); d.add(String(200,56,'4-head cross-attention',fontName='DejaVu',fontSize=6.5,fillColor=colors.white,textAnchor='middle')); d.add(String(200,44,'body / grasp queries',fontName='DejaVu',fontSize=6.5,fillColor=colors.white,textAnchor='middle'))
            d.add(Line(103,65,143,65,strokeColor=MUTED,strokeWidth=1)); d.add(Polygon([143,65,137,69,137,61],fillColor=MUTED,strokeColor=MUTED)); d.add(Line(103,18,143,45,strokeColor=MUTED,strokeWidth=1)); d.add(Polygon([143,45,136,47,140,40],fillColor=MUTED,strokeColor=MUTED))
            outs=[('body tokens\n10 x 64',300,58,82,28,TEAL),('hand refs\n50 x 14',300,17,82,28,ORANGE),('phase 3\n+ progress',414,38,82,28,NAVY)]
            for label,x,y,w,h,c in outs:
                d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=c,strokeColor=c))
                for j,line in enumerate(label.split('\n')): d.add(String(x+w/2,y+h/2+5-j*10,line,fontName='DejaVu-Bold',fontSize=6.8,fillColor=colors.white,textAnchor='middle'))
            for y in [64,45]:
                d.add(Line(258,y,300,y,strokeColor=MUTED,strokeWidth=1)); d.add(Polygon([300,y,294,y+4,294,y-4],fillColor=MUTED,strokeColor=MUTED))
            d.add(Line(258,55,414,52,strokeColor=MUTED,strokeWidth=1)); d.add(Polygon([414,52,408,56,408,48],fillColor=MUTED,strokeColor=MUTED))
        d.scale(self.width/530.0, 1.0)
        d.drawOn(self.canv, 0, 0)

def page_header(n, label):
    return [P(f'{n:02d}  {label}', 'PageTitle'), HRFlowable(width='100%', thickness=1.1, color=TEAL, spaceAfter=7)]

def draw_page(canvas, doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(25*mm, 14*mm, w-25*mm, 14*mm)
    canvas.setFont('DejaVu', 6.7); canvas.setFillColor(MUTED)
    canvas.drawString(25*mm, 9*mm, 'GROUP 2  |  TRACK 4  |  Beyond Motion Generation')
    canvas.drawRightString(w-25*mm, 9*mm, f'{doc.page}')
    canvas.restoreState()

def page1():
    s=[]
    s += [P('GROUP 2  |  TRACK 4', 'Small'), P('Beyond Motion Generation', 'CoverTitle'), P('From Motion Execution to Visuomotor Learning', 'Subtitle')]
    s += [P('<b>Abstract.</b> We study a complete text-conditioned tabletop grasp pipeline for a Unitree G1 in MuJoCo. Kimodo and ARDY generate distinct upper-body motion references; deterministic task-conditioned adapters retarget them to a canonical 29-DoF body and an independently scheduled 14-DoF hand; SONIC then performs feedback tracking while MuJoCo supplies dynamics. A Canonical Data Factory records the qualified execution as synchronized learning data, and a V4 visuomotor policy predicts future SONIC body tokens and hand references from RGB and proprioception. The historical October 8 comparison gives 19/20 validated successes for each method in a fixed-center scene, but adapter and hardware confounds preclude an architecture-level ranking. V4 achieves strong offline metrics and complete closed-loop scheduling, yet its recorded learned rollout does not grasp the cube. The central result is therefore an evidence-aligned interface from generated motion to visuomotor learning, together with explicit boundaries on what has and has not been demonstrated.', 'Lead')]
    s += [Diagram('overview',450,116), caption('Figure 1. End-to-end interface. Learned motion generation ends at a reference boundary; execution, qualification, canonicalization, and VA learning are separate gates.'), section('Research question and contributions')]
    s += [P('The project asks whether text-conditioned whole-body motion can be made physically executable and then converted into aligned expert demonstrations for visual action learning. The contribution is a maintained chain rather than a claim of autonomous manipulation: (i) two genuinely different generators under one G1/SONIC/MuJoCo stack; (ii) explicit wrist, IK, clearance, hand, and timing adaptation; (iii) a provenance-preserving Canonical Data Factory; and (iv) a V4 model whose output contract matches SONIC and Hand PD. We also preserve failed trials, separate physical success from evidence integrity, and report the limits of small, adapter-dependent experiments [1-4].')]
    s += [box('<b>Scope.</b> The evidence concerns a right-hand grasp, lift, and hold of a 0.07 m cube on a fixed tabletop. “Generated motion” means an upstream reference. It is not a torque command, a dexterous-hand trajectory in either native generator, or a guarantee of successful closed-loop grasping.', PALE)]
    return s

def page2():
    s=page_header(2,'Archon experience and motion-generation models')
    s += [section('2.1  What we actually observed in Archon')]
    s += [P('On 29 September 2026, under staff supervision, we attended humanoid-robot training at Archon Robotics. We experienced PICO human-data and motion-capture collection and watched staff record a real-robot task demonstration and run policy inference. PICO uses first-person images with head and hand pose; teleoperation applies skeleton processing and retargeting to turn operator motion-capture into a robot-trackable reference. The workflow was concrete: wear and calibrate devices, check camera and data streams, connect teleoperation, place the object, record the task, then reset the scene. Operators stay close to the robot to avoid a discontinuous reference at handoff.')]
    s += [P('The training explanation described an episode timeline containing camera images, robot joint and end-effector state, action targets or SONIC motion tokens, human skeleton and hand state, instruction/task text, timestamps, and outcome flags. We did not independently export or verify every field. In autonomous inference, an S1/VLA receives image, instruction, and robot state and predicts a future action chunk; S0/SONIC converts the motion representation into joint targets, tracks and balances at about 50 Hz, and joint PD plus feedback execute it. We saw a camera-disconnect warning. Staff also explained that recovery after an object falls is possible but not guaranteed and requires failure-state demonstrations. These observations motivate our data-quality gates, but they are not evidence that our Kimodo or ARDY policy ran on the physical robot [9].')]
    s += [section('2.2  Two distinct generators')]
    s += [tbl([
        ['Property','Kimodo','ARDY'],
        ['Native input','Text embedding, task pose constraints, root/body constraints; 30 Hz model','Text embedding plus structured constraints; 25 Hz model'],
        ['Native output','Full-body/root motion; final hand schedule supplied downstream','Full-body/root motion; metadata explicitly says model_generated_hands=false'],
        ['Temporal representation','487 authored frames at 30 Hz; segmented export to 810 frames at 50 Hz','Five segments [100,100,50,50,106], five-frame overlap and four-frame history'],
        ['Continuity','Segmented generation followed by deterministic resampling and transition checks','Autoregressive history conditioning and feature-space overlap blending'],
        ['Task conditioning','Wrist/pose constraints and text; task adapter adds approach/contact refinement','Wrist corridor, object case, and text; adapter projects wrist and solves bounded IK'],
        ['Final hand','Independent timed preclose/close reference','Independent cubic preclose/close reference; left hand zeroed'],
    ],[63,188,199],tiny=True), caption('Table 1. Native generator differences. Shared SONIC execution begins only after producer-specific adaptation; these are not two prompts to one generator.')]
    s += [P('Kimodo therefore provides a sampled, segmented body/root trajectory whose canonical exporter performs quaternion SLERP, joint mapping, wrist/contact refinement, clearance, calibration, and per-frame IK. ARDY samples five text-conditioned segments, carries a short native history across boundaries, blends the overlap, and then uses a planned wrist target, bounded projection, calibration, clearance, IK, and an explicit hand schedule. Both expose text-driven semantics, but spatial conditioning and downstream correction carry much of the task specificity. Neither provides real-time language replanning in the formal experiment [4,5].')]
    return s

def page3():
    s=page_header(3,'Motion adaptation and SONIC integration')
    s += [section('3.1  From generated motion to a canonical G1 reference')]
    s += [P('The adapter is a task-conditioned transformation, not a cosmetic file conversion. A canonical episode has a 50 Hz time grid t_k = k/50, 29 body joint positions and velocities, root position and unit wxyz quaternion, and a 14-DoF hand reference. Source trajectories are resampled without padding: scalar joints and root position are linearly interpolated and orientations use quaternion SLERP. Joint names are mapped through the protected G1 order. The adapter blends the wrist toward a structured world-frame target, adds a tested clearance corridor, applies bounded correction and tracking calibration, and solves IK at each frame. A candidate is rejected when IK error or correction bounds fail. This is why the same raw generator can lead to different physical behavior under different frozen adapters.')]
    s += [Diagram('control',450,116), caption('Figure 2. SONIC body control and independent hand control. The 64-D post-FSQ token is an encoded motion representation, not a direct torque vector.')]
    s += [section('3.2  SONIC, dynamics, and feedback')]
    s += [P('At the native boundary, the SONIC release encoder maps a 1,762-D observation to a 64-D post-FSQ motion token; the decoder maps a 994-D history/context vector to 29 body outputs [6]. The native controller receives the 50 Hz body reference, advances MuJoCo at 200 Hz (four physics steps per control tick), and uses state feedback to track the reference. A separate TimestampedHand path applies 14 hand targets through the preserved Hand PD controller. Thus body tracking and dexterous closure are coupled by the scene and timing, but they are not one learned output. The standing/root reference and lower-body stabilizer remain active while upper-body IK moves the right arm; lower-body motion is measured in the trace rather than inferred from a successful grasp.')]
    s += [P('The physical task is staged as raise/prepare, approach, grasp, lift, and hold. For a natural-language instruction such as “grasp the red cube,” external task planning converts the semantic request into a fixed task case, wrist corridor, object pose, and phase schedule. The generator supplies body motion semantics; the adapter supplies deterministic preparation, clearance, IK, calibration, and hand timing. SONIC supplies feedback tracking. The formal replay consumes a prepared reference and does not ask the generator for a new text command after each observation. “Closed loop” therefore refers to low-level state feedback and, later, VA receding-horizon scheduling, not language replanning.')]
    s += [box('<b>Reference versus state.</b> body_ref_q and hand_ref_q are planned targets. MuJoCo qpos, contacts, RGB, decoded SONIC action, and applied hand command are executed evidence. The Canonical Data Factory stores both; substituting the former for the latter would hide tracking error and contact failures.',PALE2)]
    return s

def page4():
    s=page_header(4,'Formal Kimodo versus ARDY comparison')
    s += [section('4.1  Protocol and acceptance criteria')]
    s += [P('We use the completed formal comparison conducted on 8 October 2026, not the unfinished Unified Adapter experiments. Each method used the same predefined seeds 1000-1019, one candidate per seed, a fixed-center cube, 810 frames at 50 Hz (16.2 s), shared SONIC/MuJoCo/Hand PD logic, and producer-specific frozen adapters. Trials were retained in the denominator; only infrastructure failure before a simulation frame could use the same candidate without counting as a task result. Physical success required lift >= 0.10 m, opposing right-thumb and right-index or middle contact, force > 1e-4 N, no cube-table contact during the qualifying interval, and >= 2.0 s continuous hold. Validated success additionally required complete finite synchronized replay, no fall, no weld/attachment/teleportation/artificial force, and all evidence checks. Robot-table contact was recorded but permitted by the primary validator, so 95% is not collision-free grasp success [1].')]
    s += [tbl([
        ['Metric (formal fixed center)','Kimodo','ARDY'],
        ['Physical success','19 / 20 (95%)','20 / 20 (100%)'],
        ['Validated success','19 / 20 (95%)','19 / 20 (95%)'],
        ['95% Wilson interval, validated','76.4%-99.1%','76.4%-99.1%'],
        ['Median end-to-end wall time','162.678 s','182.749 s'],
        ['Mean lift, validated successes (n=19)','0.126150 m','0.139196 m'],
        ['Mean hold, validated successes (n=19)','4.6308 s','4.8466 s'],
        ['Distinct formal failure','Seed 1006: physical grasp failure','Seed 1007: physical success; evidence-integrity failure'],
    ],[168,128,154]), caption('Table 2. Primary results from the historical formal batch. The values describe complete adapted pipelines, not isolated raw motion quality.')]
    s += [section('4.2  Interpretation, timing, and failures')]
    s += [P('The paired result had 18 seeds passing for both methods, one ARDY-only pass, and one Kimodo-only pass; exact McNemar p = 1.0. This small sample does not establish equivalence or a population reliability of 95%. Kimodo seed 1006 did not establish effective grasp contact and had 0 s hold. ARDY seed 1007 lifted 0.137024 m and held 4.820 s but had incomplete state-frame coverage and a stale reference cursor, so evidence integrity failed after physical replay. The primary comparison therefore separates task success from reproducibility of the recorded trace.')]
    s += [P('The 20.071 s median timing difference is descriptive. Kimodo medians included 17.571 s model loading, 50.705 s generation, 5.727 s adapter, 36.721 s SONIC initialization, and 16.200 s replay; ARDY generation was 97.227 s including load, adapter 8.304 s, and replay 56.567 s including initialization. The recorded endpoint is not instruction-to-first-action latency. GPU allocation differed (Kimodo GPU 1, ARDY GPU 7 despite the intended shared allocation), and stage boundaries, batch order, warm-up, DDS waits, and later state recording were unequal. Runtime user corrections were prohibited, so the evaluation count is zero; earlier ARDY development replay also recorded zero. Development prompt/clearance tuning is not an online correction. ARDY passed a strict no-table criterion in 0/20 formal trials; Kimodo did not expose a directly comparable strict field.')]
    s += [box('<b>Attribution caution.</b> Producer-specific adapters materially change wrist targets, clearance, calibration, hand timing, and IK. Success and timing differences therefore belong to the complete adapted pipelines. They cannot be assigned to Kimodo or ARDY architecture alone.', PALE)]
    return s

def page5():
    s=page_header(5,'Physical feasibility and the Canonical Data Factory')
    s += [section('5.1  Generalization and physical evidence')]
    s += [P('The separate spatial pilot is diagnostic, not pooled with the formal center batch. In five positions with two seeds per method, preparation failures were retained: ARDY reached replay in 8/10 and physically succeeded in 2/10; Kimodo reached replay in 6/10 and physically succeeded in 3/10. A fresh ARDY attribution study illustrates the limitation: center passed (0.1371 m lift, 4.84 s hold), near passed (0.12596 m, 3.175 s), left lifted 0.1939 m but held 1.385 s, and right lifted 0.007966 m with 0 hold. This is evidence of reachability and preparation sensitivity, not workspace coverage. Pilot body-tracking RMSE ranged 0.0634-0.1457 rad for ARDY and 0.1118-0.1803 rad for Kimodo, with few finite traces. A fixed-reference Kimodo smoke measured 3.278 ms mean, 3.631 ms P95, and 4.446 ms maximum per profiled control event; it does not certify the complete DDS path.')]
    s += [section('5.2  Canonical Data Factory')]
    s += [Diagram('factory',450,92), caption('Figure 3. Canonicalization is an evidence join, not a row copy. The factory preserves source provenance and refuses missing or late identities.')]
    s += [P('The production chain is <b>Generated Reference -> native SONIC/MuJoCo execution -> physical qualification -> trace alignment -> canonical episode -> VA samples</b>. At each 50 Hz control tick, the factory records the authored reference, MuJoCo body state, root pose, decoded SONIC token/action, independent hand reference, contacts and result flags, while head RGB is recorded at 10 Hz. Native logs carry frame/tick identities. The compiler joins observations by sample_id, observation_id, token_id, and hand_ref_id, checks source hashes, validates producer/case/quaternion/unit metadata, and extracts complete future horizons. It does not repair missing data by array position or pad a short target.')]
    s += [P('This timing distinction matters. Planning time is when a generator or adapter writes a reference; native control time is the authored tick at which SONIC and Hand PD consume it; post-step observation time is after MuJoCo advances. A naive row-index join can pair an observation with the preceding or following action when logging is late, duplicated, or sampled at a different rate. The canonical interface makes the alignment explicit and treats any missing identity as a failed episode. This is the reproducible bridge from motion execution to learning.')]
    s += [tbl([
        ['Verified production evidence','Value'],
        ['Accepted fresh episodes','Kimodo seed 1; ARDY seed 2; both center, right arm, 50 Hz'],
        ['Per-episode accepted rows','161 raw rows each; 152 complete V4 horizons each'],
        ['Mixed V4 manifest','304 samples; RGB [304,3,128,128], body [304,68], hand [304,28], body targets [304,10,64], hand targets [304,50,14]'],
        ['Historical Kimodo merge','19 source episodes, 3,059 samples; episode 0015 explicitly excluded'],
    ],[145,305],tiny=True), caption('Table 3. Counts are taken from manifests; duplicate output roots are not counted as independent episodes [2,3].')]
    return s

def page6():
    s=page_header(6,'VA V4 architecture and training')
    s += [section('6.1  Objective and interface')]
    s += [P('VA asks a different question from Kimodo/ARDY and SONIC: given the current visual observation and robot state, can a policy predict the next SONIC-compatible body motion tokens and dexterous-hand reference? It does not generate the original motion representation, and it does not replace SONIC. Its inputs are RGB [B,3,128,128], a 68-D body state, and a 28-D hand state. The body state is 29 q + 29 dq + root quaternion (4) + angular velocity (3) + gravity (3); the hand state is 14 q + 14 dq. The outputs are future body tokens [B,10,64], future hand references [B,50,14], three-class phase logits, and a sigmoid closure-progress scalar [7,8].')]
    s += [Diagram('va',450,104), caption('Figure 4. V4 implementation from va_research/models/v4/model.py. There is no attention over a temporal observation history; temporal extent is in the target horizons and scheduler.')]
    s += [section('6.2  Network details')]
    s += [P('The visual encoder is four stride-2 3x3 convolutions, 3 -> 32 -> 64 -> 128 -> 256 channels, each followed by GroupNorm(8, channels) and GELU. The 8x8 output becomes 64 visual positions of width 256 and receives a learned [1,64,256] positional parameter. Body and grasp queries each use an MLP 68 -> 256 -> 256, LayerNorm, and GELU. Each query uses a 4-head 256-wide cross-attention block with query/value normalization, residual attention, LayerNorm, and a 256 -> 512 -> 256 feed-forward layer. The hand state uses a 28 -> 128 -> 128 MLP. The body branch maps 256 -> 512 -> 512 and then 512 -> 512 -> 640, reshaped to 10x64. The grasp branch fuses a 256-D cross-attended feature, 64-D phase feature, and 128-D hand feature (448 total), then maps 448 -> 512 -> 512 -> 700, reshaped to 50x14.')]
    s += [section('6.3  Loss and training')]
    s += [P('The implemented objective is exactly: L = SmoothL1(body_tokens, target) + MSE(hand_ref, target) + 0.1 CE(phase_logits, phase) + 0.1 MSE(progress, target). Body and hand normalization statistics are fitted on training data, with standard deviations clamped at 0.05. The configured trainer uses AdamW (learning rate 3e-4), batch size 32, seeded shuffling, gradient clipping at 10, and the offline gates token MAE <= 0.01, FSQ scalar accuracy >= 0.95, hand MAE <= 0.02, moving-right-hand MAE <= 0.03, phase accuracy >= 0.98, and progress MAE <= 0.03. Labels are reference-derived: phase and progress describe motion closure, not an independently sensed physical phase.')]
    s += [box('<b>Target distinction.</b> A body token target is the recorded post-FSQ SONIC token at a future authored tick. A hand target is an absolute 14-DoF reference. Neither target is the executed qpos; MuJoCo state and contacts remain separate supervision and evaluation evidence.',PALE)]
    return s

def page7():
    s=page_header(7,'VA evaluation, closed loop, and failure analysis')
    s += [section('7.1  Four gates, not one score')]
    s += [P('The V4 record supports four distinct claims. <b>Oracle physical replay</b> asks whether a known expert reference can grasp. <b>Offline imitation</b> asks whether V4 predicts held-out rows in the available evaluation protocol. <b>Closed-loop execution integrity</b> asks whether every scheduled token is consumed at the intended absolute tick. <b>Closed-loop task success</b> asks whether the predicted policy actually lifts and holds the cube. A pass at an earlier gate cannot stand in for a pass at a later one.')]
    s += [tbl([
        ['Recorded V4 result','Evidence'],
        ['Offline prediction (epoch 50; 7,530 train / 1,958 val)','Token MAE 0.004183; FSQ scalar accuracy 0.98914; whole-token exact 0.55907; hand MAE 0.002954; phase accuracy 0.99304; progress MAE 0.004347; offline gate true'],
        ['Dataset caveat','The reported validation overlaps the training source in the historical artifact; it is not a clean held-out generalization claim'],
        ['Closed-loop scheduler','810/810 ticks, 162 plans, 0 uncovered/repeated/skipped ticks; 1 overlap fallback and 1 deadline miss; strict runtime pass true'],
        ['Closed-loop task','Physical grasp false; max lift 0.019827 m, final lift -0.000108 m, continuous hold 0 s; 107 robot-table and 640 self-contact steps; no fall'],
    ],[148,302],tiny=True), caption('Table 4. Offline accuracy and runtime completion coexist with a failed physical grasp. Values are from outputs/va_v4_1_kimodo_rule_20261009 [11,12].')]
    s += [section('7.2  Receding-horizon inference')]
    s += [P('At each decision, VA receives current RGB and current body/hand state, predicts 10 body tokens and 50 hand references, and inserts them into an absolute-tick PlanBuffer. Plans start exactly five control ticks apart. The buffer rejects malformed shapes, skipped or repeated ticks, and uncovered targets; it never pads with stale output. An external-token adapter sends each 64-D token with its authored execution index across the ZMQ/DDS boundary. Native SONIC decodes and tracks the body, TimestampedHand applies Hand PD, MuJoCo advances, and the next RGB/state pair feeds the next decision. Synchronization is therefore a model requirement, not a logging convenience.')]
    s += [section('7.3  What failed and what it means')]
    s += [P('The learned rollout completed without falling, but the hand did not form the required opposing contact and the cube remained on the table. Self-contact and table-contact diagnostics are evidence of feedback distribution shift, wrist/body drift, and hand-state mismatch, not proof that the visual encoder is incapable of prediction. Offline behavior cloning sees reference-aligned rows; closed loop sees its own errors, delayed observations, and contacts. Oracle replay and the accepted fresh Kimodo/ARDY episodes show that the native execution stack can grasp under qualified references, while the V4 rollout shows that prediction accuracy and scheduler integrity do not yet transfer to independent grasping. Body-versus-hand isolation, oracle token replay, and phase/progress heads are useful diagnostics, but the recorded artifacts do not establish reliable learned closed-loop grasping or a self-improving data flywheel.')]
    s += [box('<b>Honest conclusion for VA.</b> V4 is a technically specified visuomotor extension with good recorded offline metrics and a valid absolute-tick runtime. Its current evidence is a failed learned physical grasp, not an autonomous manipulation result.', PALE)]
    return s

def page8():
    s=page_header(8,'Discussion, reproducibility, and conclusion')
    s += [section('8.1  What the project establishes')]
    s += [P('The shared reference boundary makes a fair comparison possible: Kimodo and ARDY differ in native temporal generation, history conditioning, and text/spatial interfaces, while the G1 model, tabletop, SONIC, Hand PD, MuJoCo dynamics, initial pose, and primary thresholds are shared. The October 8 center batch found equal validated success counts (19/20 each) with different failure mechanisms and a lower recorded median end-to-end time for Kimodo. That result is descriptive and adapter-dependent. The spatial pilot shows why calibration and preparation must be treated as evidence-producing components, not hidden implementation details.')]
    s += [P('The Canonical Data Factory is the conceptual bridge to the title. A generated reference becomes a learning example only after native execution, physical qualification, tick-identity alignment, and complete future-horizon extraction. V4 then tests whether visual observations and proprioception can recover that action interface. The current answer is partial: expert references can be canonicalized, offline prediction is strong, and timing integrity can be checked, but independent learned grasping remains open.')]
    s += [section('8.2  Reproduction and dependencies')]
    s += [P('The maintained entry point is the workspace experiments interface (for example, <font name="DejaVu-Bold">python -m experiments run --family kimodo</font> or <font name="DejaVu-Bold">--family ardy</font>), followed by canonical replay, dataset compilation, V4 training, offline evaluation, and the closed-loop runner. Native work requires the validated <font name="DejaVu-Bold">va-train</font> environment: Python 3.10.12, PyTorch 2.9.0, torch_musa 2.9.0+8432052, MUSA SDK 4.3.6, MTT S4000 rendering, isolated DDS namespace, protected SONIC sources, the dl-group2 tabletop/controller, and local ARDY/Kimodo checkpoints. The expected input is a text task plus a fixed cube case; the expected output is a provenance-bearing 50 Hz reference, then a qualified trace and canonical episode. Evaluation checks lift, hold, contacts, finite synchronized replay, and evidence integrity.')]
    s += [P('The workspace is a set of preserved repositories, not one verified public GitHub repository. ARDY, Kimodo, SONIC, checkpoints, text encoders, and controller assets are external dependencies; no public URL is asserted here. The report was editorially synthesized with language-model assistance, while numerical claims and architectural statements were checked against maintained source, manifests, and historical reports. No language model performed the robot experiments.')]
    s += [section('8.3  Limitations and conclusion')]
    s += [P('The complete per-trial bundle for the historical 20-seed comparison is unavailable for full independent reconstruction. The timing endpoints are unequal, the sample is small, and the primary validator permits recorded robot-table contact. The formal batch does not measure instruction-to-first-action latency, online corrections, broad spatial generalization, or language replanning. The V4 offline validation overlaps its training source, and its closed-loop physical task failed. These limitations are part of the result.')]
    s += [box('<b>Conclusion.</b> Group 2 moves beyond motion generation by defining and testing the interfaces that make generated motion executable, auditable, and learnable. The evidence supports a reproducible execution-to-data pipeline and a precise V4 research direction. It does not support claiming collision-free universal grasping, statistical equivalence of Kimodo and ARDY, or reliable autonomous VA grasping.', PALE)]
    return s

def refs():
    s=[P('References and source record', 'PageTitle'), HRFlowable(width='100%', thickness=1.1, color=TEAL, spaceAfter=7)]
    refs=[
        '[1] Group 2, <i>Test and Comparison Report v5</i>, formal fixed-center Kimodo/ARDY comparison, 8 October 2026. Source: research/report/Test_and_Comparison_Report_v5.tex.',
        '[2] Group 2, <i>Technical Architecture and Reproducibility Report</i>, 16 October 2026. Source: docs/TECHNICAL_REPORT.md.',
        '[3] Group 2, <i>Canonical Data Factory production manifests</i>, 16 October 2026. Sources: outputs/canonical_production/production_summary.json and mixed_v4_dataset_20261016.',
        '[4] Group 2, <i>Kimodo generation attribution and canonical export</i>. Source: docs/KIMODO_GENERATION_ATTRIBUTION.md and research/kimodo/.',
        '[5] Group 2, <i>ARDY generation attribution and canonical export</i>. Source: docs/ARDY_GENERATION_ATTRIBUTION.md and research/ardy/.',
        '[6] SONIC release and execution contracts. Sources: execution/sonic_mujoco/, va_research/sonic/, and docs/architecture_audit/PIPELINE_CALL_CHAINS.md.',
        '[7] Group 2, <i>V4 policy implementation</i>. Source: va_research/models/v4/model.py, dataset.py, and runtime.py.',
        '[8] Group 2, <i>V4 objectives and training</i>. Source: va_research/models/v4/losses.py, training.py, and configs/token_v4.json.',
        '[9] Group 2, <i>Archon visit: demonstration collection and policy inference</i>, 29 September 2026. Source: research/report/archon.',
        '[10] Group 2, <i>Reproducibility Guide and Index</i>. Sources: docs/REPRODUCIBILITY_GUIDE.md and docs/REPRODUCIBILITY_INDEX.md.',
        '[11] Group 2, <i>V4 offline evaluation artifact</i>. Source: outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json.',
        '[12] Group 2, <i>V4 closed-loop evaluation artifact</i>. Sources: outputs/va_v4_1_kimodo_rule_20261009/closed_loop/report.json and physical_metrics.json.',
        '[13] Group 2, <i>Track 4 unified scene and control interface</i>. Source: docs/TRACK4_INTERFACE.md.',
        '[14] Group 2, <i>Unified expert validation and dataset schema</i>. Sources: docs/UNIFIED_EXPERT_VALIDATION.md and data_factory/dataset/.',
    ]
    for r in refs: s.append(P(r,'Ref'))
    s += [Spacer(1,6), section('Evidence labels used in this report'), P('<b>Verified</b> denotes a maintained source, present artifact, manifest, hash, or inspected execution record. <b>Reported</b> denotes a historical summary whose complete primary bundle is not present in the current workspace. The October 8 comparison is therefore reported historical evidence; fresh canonical production and V4 artifacts are verified current evidence. This distinction prevents newer unfinished Unified Adapter runs from replacing the requested formal comparison.', 'Body2')]
    return s

def build():
    doc=SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=25*mm, rightMargin=25*mm,
        topMargin=16*mm, bottomMargin=18*mm, title='Beyond Motion Generation', author='Group 2')
    story=[]
    pages=[page1(),page2(),page3(),page4(),page5(),page6(),page7(),page8()]
    # Each page is shrink-wrapped independently so the main body is exactly eight pages.
    maxw=A4[0]-50*mm; maxh=A4[1]-34*mm
    for i,flows in enumerate(pages):
        story.append(KeepInFrame(maxw,maxh,flows,mode='shrink',mergeSpace=True))
        story.append(PageBreak())
    story.append(KeepInFrame(maxw,maxh,refs(),mode='shrink',mergeSpace=True))
    doc.build(story,onFirstPage=draw_page,onLaterPages=draw_page)

if __name__=='__main__': build()
