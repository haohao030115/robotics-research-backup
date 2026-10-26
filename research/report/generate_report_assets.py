from pathlib import Path
import json,csv,math,html,collections,statistics,sys
ROOT=Path('/home/group2')
OUT=ROOT/'research/report'; FIG=OUT/'figures'; TAB=OUT/'tables'
FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
# ---------- SVG utilities ----------
def esc(x): return html.escape(str(x))
def begin(name,w=1400,h=800,source=''):
    return [f'''<?xml version="1.0" encoding="UTF-8"?>\n<!-- Figure {name}; source: {source} -->\n<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n<metadata>Source evidence: {esc(source)}</metadata>\n<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#34495e"/></marker><marker id="arrowblue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2878b5"/></marker></defs>\n<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>''']
def end(): return ['</svg>']
def text(a,x,y,s,size=20,weight='400',fill='#18212b',anchor='start'):
    a.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}px" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')
def multiline(a,x,y,lines,size=17,leading=23,fill='#18212b',weight='400',anchor='start'):
    for i,s in enumerate(lines): text(a,x,y+i*leading,s,size,weight,fill,anchor)
def rect(a,x,y,w,h,fill='#f5f8fb',stroke='#395a73',sw=2,rx=12):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def line(a,x1,y1,x2,y2,stroke='#34495e',sw=2,dash='',marker='url(#arrow)'):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    a.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d} marker-end="{marker}"/>')
def path(a,d,stroke='#34495e',sw=2,fill='none',dash='',marker=''):
    da=f' stroke-dasharray="{dash}"' if dash else ''
    ma=f' marker-end="{marker}"' if marker else ''
    a.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}{ma}/>')
def title(a,t,subtitle=None,w=1400):
    text(a,40,45,t,29,'700','#152238')
    if subtitle: text(a,40,75,subtitle,15,'400','#516171')
def box(a,x,y,w,h,head,lines,fill='#f5f8fb',stroke='#395a73',headfill='#183b56'):
    rect(a,x,y,w,h,fill,stroke,2,12); text(a,x+14,y+28,head,18,'700',headfill)
    multiline(a,x+14,y+55,lines,15,21,'#233240')
def save(name,a): (FIG/name).write_text('\n'.join(a+end()),encoding='utf-8')
# ---------- Diagram figures ----------
# 1 system architecture
src='docs/TECHNICAL_REPORT.md; docs/architecture_audit/PIPELINE_CALL_CHAINS.md; va_research/models/v4_1/model.py; data_factory/runtime; outputs/va_v4_1_kimodo_rule_20261009/closed_loop/provenance.json'
a=begin('system_architecture.svg',1600,860,src); title(a,'Figure 1. End-to-end visual-action system architecture','Evidence-backed architecture; execution gates and physical result are reported separately.',1600)
box(a,45,130,220,180,'Motion producers',['Kimodo prompt + checkpoint','ARDY segmented generator','Native reference trajectories'], '#eef5fb','#2878b5')
box(a,325,130,220,180,'Unified Adapter',['50 Hz canonical body array','joint/frame calibration','provenance + stage hashes'], '#f1f7f2','#4b8a5a')
box(a,605,130,245,180,'Canonical Data Factory',['native replay + observations','timestamp/ID alignment','validation + immutable episode'], '#fff8ec','#bd7b20')
box(a,910,130,235,180,'V4.1 VA training',['RGB + body + hand inputs','cross-attention policy','auxiliary supervision'], '#f4effb','#7653a6')
box(a,1205,130,320,180,'Checkpoint and offline gate',['checkpoint + dataset hashes','token/hand/phase/progress metrics','promotion before closed loop'], '#fceff0','#b44351')
for x in [265,545,850,1145]: line(a,x,220,x+60,220)
text(a,55,370,'Online closed-loop path',20,'700','#152238')
box(a,45,415,220,175,'RGB/state capture',['head_rgb (10 Hz)','68D body + 28D hand','current observation IDs'], '#eef5fb','#2878b5')
box(a,325,415,220,175,'Inference and scheduler',['V4.1 forward pass','10 body tokens + 50 hand refs','absolute tick PlanBuffer'], '#f4effb','#7653a6')
box(a,605,415,245,175,'SONIC token handoff',['64D post-FSQ token stream','native decoder/controller','hand PD reference path'], '#fff8ec','#bd7b20')
box(a,910,415,235,175,'DDS and physics',['50 Hz commands / 200 Hz physics','isolated DDS namespace','MuJoCo tabletop scene'], '#f1f7f2','#4b8a5a')
box(a,1205,415,320,175,'Evaluation and diagnostics',['scheduler integrity','contacts / lift / hold / fall','MTT S4000 renderer evidence'], '#fceff0','#b44351')
for x in [265,545,850,1145]: line(a,x,505,x+60,505)
line(a,1340,310,1340,405,'#7a8791',2,'6 5'); text(a,1360,360,'promote only if gate passes',14,'400','#6c7780')
text(a,45,690,'Recorded result for representative V4.1 rollout',18,'700','#152238')
box(a,45,715,430,95,'Runtime integrity',['810/810 ticks; uncovered=0; strict_runtime_pass=true'], '#eaf7ef','#4b8a5a')
box(a,500,715,500,95,'Physical task result',['physical_grasp_success=false; max lift=0.019827 m; hold=0 s'], '#fff1f0','#b44351')
box(a,1025,715,500,95,'Interpretation',['Complete runtime trace, failed physical lift criterion'], '#f7f7f7','#65737e')
save('system_architecture.svg',a)
# 2 generation to execution
a=begin('generation_to_execution.svg',1500,760,src); title(a,'Figure 2. Kimodo/ARDY generation to SONIC execution','Producer-neutral adapter boundary and canonical runtime handoff.',1500)
steps=[('Kimodo','prompt, seed, native checkpoint'),('ARDY','segmented trajectory, seed'),('Unified Adapter','canonical 50 Hz body + hand references'),('Canonical replay','SONIC encoder, references, observations'),('V4.1 policy','current RGB/state -> future plan'),('SONIC + hand PD','64D tokens + 14D hand reference'),('DDS/MuJoCo','physics, contacts, task metrics')]
xs=[35,235,435,685,885,1085,1285]; ws=[160,160,210,180,180,180,180]
for i,(h,l) in enumerate(steps):
    box(a,xs[i],300 if i not in [2,3] else 260,ws[i],135,h,[l], '#eef5fb' if i<2 else '#f4effb' if i==4 else '#fff8ec' if i in [2,3,5] else '#f1f7f2','#2878b5' if i<2 else '#7653a6' if i==4 else '#bd7b20' if i in [2,3,5] else '#4b8a5a')
    if i<len(steps)-1: line(a,xs[i]+ws[i],367,xs[i+1]-4,367)
text(a,35,180,'Training-time artifact lineage',19,'700','#152238'); line(a,35,210,1320,210,'#a0aab5',2,'4 4','')
for x,label in [(115,'native motion'),(340,'adapted motion'),(575,'validated episode'),(775,'sample mapping'),(975,'checkpoint'),(1175,'release plan')]:
    a.append(f'<circle cx="{x}" cy="210" r="7" fill="#2878b5"/>'); text(a,x,245,label,14,'400','#516171','middle')
text(a,35,560,'Invariants carried across the boundary',19,'700','#152238')
for i,(h,l) in enumerate([('Identity','sample_id, observation_id, token_id, hand_ref_id'),('Timing','authored timestamps; 50 Hz control / 200 Hz physics'),('Semantics','native post-FSQ 64D tokens; no synthetic future labels'),('Diagnostics','source hashes, scheduler events, physical criterion')]): box(a,35+365*(i%4),590,330,105,h,[l],'#f7f9fb','#7a8791','#203a50')
save('generation_to_execution.svg',a)
# 3 data factory
a=begin('data_factory.svg',1500,850,src); title(a,'Figure 3. Canonical Data Factory architecture','Canonical compilation preserves native traces, IDs, timing and validation evidence.',1500)
box(a,45,130,260,220,'Inputs',['Kimodo/ARDY reference','native SONIC replay','camera + body/hand state','controller observations'], '#eef5fb','#2878b5')
box(a,365,130,270,220,'Trace join and alignment',['authored tick IDs','observation/token/reference IDs','RGB at 10 Hz, control at 50 Hz','physics snapshots at 200 Hz'], '#fff8ec','#bd7b20')
box(a,695,130,255,220,'Validation gates',['finite arrays','joint order / quaternion wxyz','target-prefix equality','renderer and replay checks'], '#f1f7f2','#4b8a5a')
box(a,1010,130,240,220,'Episode artifact',['metadata.json','samples.npz','sample_mapping.jsonl','provenance hashes'], '#f4effb','#7653a6')
box(a,1310,130,145,220,'Dataset',['accepted','or rejected'], '#fceff0','#b44351')
for x in [305,635,950,1250]: line(a,x,240,x+55,240)
text(a,55,420,'V4 sample compiler (per retained sample)',19,'700','#152238')
box(a,55,455,255,200,'Current observation',['RGB [3,128,128]','body [68]','hand [28]','history-valid mask'], '#eef5fb','#2878b5')
box(a,365,455,255,200,'Authored targets',['body_tokens [10,64]','hand_ref [50,14]','future offsets +2..+6','phase + progress labels'], '#f4effb','#7653a6')
box(a,675,455,255,200,'Integrity checks',['sample ID match','contiguous target IDs','exact 5-slot prefix','drop incomplete tails'], '#f1f7f2','#4b8a5a')
box(a,985,455,255,200,'Splits and normalization',['63 episodes total','train 7,530 / val 1,958','train-only mean/std','std clamp min 0.05'], '#fff8ec','#bd7b20')
box(a,1295,455,160,200,'Output',['9488 rows','manifest + hashes'], '#fceff0','#b44351')
for x in [310,620,930,1240]: line(a,x,555,x+55,555)
text(a,55,730,'Source: va_research/models/v4/dataset.py; expert_datasets/train_dataset_v0/manifest.json and split.json',14,'400','#516171')
save('data_factory.svg',a)
# 4 temporal alignment
a=begin('temporal_alignment.svg',1500,800,src); title(a,'Figure 4. Native tick synchronization and temporal alignment','Rates and IDs are explicit; the scheduler consumes absolute ticks.',1500)
left=180; right=1390
text(a,40,140,'Time scale (ms)',16,'700','#152238');
for i,t in enumerate(range(0,201,20)):
 x=left+(right-left)*t/200; line(a,x,160,x,690,'#e1e6eb',1,'4 4',''); text(a,x,720,str(t),13,'400','#516171','middle')
labels=[('Physics',200,220,'5 ms tick'),('Control / SONIC',50,350,'20 ms tick'),('RGB capture',10,480,'100 ms tick'),('Plan starts',10,610,'5 control ticks / 100 ms')]
for lab,hz,y,desc in labels:
 text(a,40,y+5,lab,18,'700','#203a50'); line(a,left,y,right,y,'#34495e',2,'','')
 text(a,40,y+30,desc,13,'400','#516171')
for t in range(0,201,5):
 x=left+(right-left)*t/200; a.append(f'<circle cx="{x}" cy="220" r="3" fill="#2878b5"/>')
for t in range(0,201,20):
 x=left+(right-left)*t/200; a.append(f'<circle cx="{x}" cy="350" r="6" fill="#7653a6"/>')
for t in range(0,201,100):
 x=left+(right-left)*t/200; a.append(f'<circle cx="{x}" cy="480" r="8" fill="#bd7b20"/>')
for t in range(0,201,100):
 x=left+(right-left)*t/200; line(a,x,595,x+70,595,'#4b8a5a',10,'',''); text(a,x,575,f'plan start {t} ms',12,'400','#4b8a5a','middle')
# lead marker
x0=left; x2=left+(right-left)*20/200
line(a,x0,155,x2,155,'#b44351',3,'',''); text(a,(x0+x2)/2,135,'authored target lead = 2 control ticks (40 ms)',14,'700','#b44351','middle')
text(a,40,750,'Evidence: closed_loop/provenance.json (control_hz=50, physics_hz=200, image_hz=10, authored_target_lead_ticks=2); v4/runtime.py PlanBuffer',13,'400','#516171')
save('temporal_alignment.svg',a)
# 5 dataset structure
a=begin('dataset_structure.svg',1500,780,src); title(a,'Figure 5. Canonical episode format and VA supervision construction','One row binds current observation tensors to future SONIC and hand-reference targets.',1500)
box(a,45,160,275,260,'Episode metadata',['episode_id, producer_type','reference_hz=50','num_frames=810','body_dof=29, hand_dof=14','quaternion order wxyz'], '#eef5fb','#2878b5')
box(a,385,160,275,260,'Current observation row',['rgb: [3,128,128]','body: [68] = 29 q + 29 dq + quat + omega + gravity','hand: [28] = 14 q + 14 dq','IDs: observation_id, rgb_frame_id'], '#fff8ec','#bd7b20')
box(a,725,160,275,260,'Future supervision',['body_tokens: [10,64]','hand_ref: [50,14]','phase: scalar in {OPEN,CLOSING,HOLD}','progress: scalar in [0,1]','future offset +2..+6'], '#f4effb','#7653a6')
box(a,1065,160,370,260,'Validation and retention',['sample_mapping.jsonl joins IDs','stored five-slot prefix checked exactly','contiguous canonical target IDs required','incomplete horizon tails dropped','no synthetic target padding'], '#f1f7f2','#4b8a5a')
for x in [320,660,1000]: line(a,x,290,x+65,290)
text(a,45,520,'Authored row identity',19,'700','#152238');
for i,(lab,desc) in enumerate([('sample_id','episode:sample:NNNNNN'),('observation_id','episode:observation:NNNNNN'),('token_id','episode:token:NNNNNN'),('hand_ref_id','episode:handref:NNNNNN')]):
 x=45+i*350; box(a,x,555,300,100,lab,[desc],'#f7f9fb','#7a8791','#203a50')
 if i<3: line(a,x+300,605,x+345,605)
text(a,45,705,'Source: expert_datasets/train_dataset_v0/index.jsonl; va_research/models/v4/dataset.py',13,'400','#516171')
save('dataset_structure.svg',a)
# 6 exact VA model architecture
a=begin('va_model_architecture.svg',1700,1060,'va_research/models/v4/model.py; va_research/models/v4_1/model.py; outputs/va_v4_1_kimodo_rule_20261009/training_status.json'); title(a,'Figure 6. Final V4.1 VA model architecture and tensor dimensions','V4.1 extends V4 with a grasp-readiness head; checkpoint reports 3,502,977 parameters.',1700)
# visual tower
text(a,40,125,'Visual encoder',18,'700','#203a50')
visual=[('RGB','[B,3,128,128]'),('Conv2d 3→32, k3,s2 + GN(8) + GELU','[B,32,64,64]'),('Conv2d 32→64, k3,s2 + GN(8) + GELU','[B,64,32,32]'),('Conv2d 64→128, k3,s2 + GN(8) + GELU','[B,128,16,16]'),('Conv2d 128→256, k3,s2 + GN(8) + GELU','[B,256,8,8]'),('Flatten(8×8), transpose + learned position','[B,64,256]')]
for i,(h,d) in enumerate(visual): box(a,40,150+i*82,330,66,h,[d],'#eef5fb','#2878b5','#203a50')
for i in range(len(visual)-1): line(a,205,216+i*82,205,230+i*82,'#2878b5',2,'','url(#arrowblue)')
# body/grasp branches
box(a,445,150,250,110,'Body input',['[B,68] normalized','MLP 68→256→256'], '#fff8ec','#bd7b20')
box(a,445,310,250,110,'Grasp query input',['same [B,68] normalized','MLP 68→256→256'], '#fff8ec','#bd7b20')
box(a,760,150,290,110,'Body cross-attention',['Q=[B,1,256], V=[B,64,256]','4 heads, LayerNorm + FF 256→512→256'], '#f4effb','#7653a6')
box(a,760,310,290,110,'Grasp cross-attention',['Q=[B,1,256], V=[B,64,256]','4 heads, LayerNorm + FF 256→512→256'], '#f4effb','#7653a6')
line(a,370,520,760,205); line(a,370,520,760,365); line(a,695,205,760,205); line(a,695,365,760,365)
# heads
box(a,1110,120,250,135,'Body path',['MLP 256→512→512','MLP 512→512→640','reshape body_tokens','[B,10,64]'], '#eaf7ef','#4b8a5a')
box(a,1110,285,250,155,'Hand path',['hand state [B,28] → 128','phase feature 256→64','concat [B,448]','MLP → 700; reshape','hand_ref [B,50,14]'], '#eaf7ef','#4b8a5a')
box(a,1110,470,250,115,'Auxiliary heads',['phase: 64→3 logits','progress: 64→1 sigmoid','V4.1 readiness: 256→1 logit'], '#fceff0','#b44351')
line(a,1050,205,1110,205); line(a,1050,365,1110,365); line(a,1050,365,1110,520)
text(a,40,740,'Parameter and interface evidence',19,'700','#152238');
box(a,40,775,420,100,'Inputs',['RGB [B,3,128,128]; body [B,68]; hand [B,28]'], '#f7f9fb','#7a8791','#203a50')
box(a,500,775,420,100,'Outputs',['body_tokens [B,10,64]; hand_ref [B,50,14]'], '#f7f9fb','#7a8791','#203a50')
box(a,960,775,400,100,'Recorded checkpoint',['50 epochs; 3,502,977 parameters; V4.1 checkpoint.pt'], '#f7f9fb','#7a8791','#203a50')
text(a,40,970,'No observation history is consumed by the model; temporal context is represented by target horizons and absolute scheduling.',14,'400','#516171')
save('va_model_architecture.svg',a)
# 7 fusion
a=begin('multimodal_fusion.svg',1500,760,'va_research/models/v4/model.py; docs/architecture_audit/PIPELINE_CALL_CHAINS.md'); title(a,'Figure 7. Visual-proprioceptive fusion and cross-attention','Separate body and grasp queries attend to a shared 64-token visual sequence.',1500)
box(a,50,140,260,130,'Visual tokens',['RGB -> four stride-2 convs','[B,64,256] + position'], '#eef5fb','#2878b5')
box(a,50,360,260,130,'Body proprioception',['q,dq, root quaternion','angular velocity, gravity','[B,68] -> normalized'], '#fff8ec','#bd7b20')
box(a,50,580,260,100,'Hand proprioception',['q,dq for 14 joints','[B,28] -> hand_state [B,128]'], '#fff8ec','#bd7b20')
box(a,420,125,300,135,'Body query branch',['MLP [B,68] -> [B,256]','unsqueeze -> Q [B,1,256]','4-head attention over visual tokens'], '#f4effb','#7653a6')
box(a,420,335,300,135,'Grasp query branch',['MLP [B,68] -> [B,256]','unsqueeze -> Q [B,1,256]','4-head attention over visual tokens'], '#f4effb','#7653a6')
box(a,420,565,300,95,'Hand feature branch',['MLP [B,28] -> [B,128]','concatenated downstream'], '#f4effb','#7653a6')
box(a,850,180,300,120,'Body context',['Q + MultiheadAttention(Q,K,V)','residual + LayerNorm','FF 256→512→256','output [B,256]'], '#eaf7ef','#4b8a5a')
box(a,850,390,300,120,'Grasp context',['Q + MultiheadAttention(Q,K,V)','residual + LayerNorm','FF 256→512→256','output [B,256]'], '#eaf7ef','#4b8a5a')
box(a,1220,250,220,180,'Fusion semantics',['body context drives 10-token body plan','grasp context drives 50-step hand plan','phase/progress/readiness read grasp context'], '#fceff0','#b44351')
line(a,310,205,420,205); line(a,310,425,420,425); line(a,310,630,420,630); line(a,720,205,850,240); line(a,720,415,850,450); line(a,1150,240,1220,305); line(a,1150,450,1220,375)
text(a,50,725,'Attention uses 4 heads at embedding width 256; no temporal token mixer is present in V4.1.',14,'400','#516171')
save('multimodal_fusion.svg',a)
# 8 prediction heads
a=begin('prediction_heads.svg',1500,820,'va_research/models/v4/model.py; va_research/models/v4_1/model.py'); title(a,'Figure 8. Body, hand and auxiliary prediction heads','All outputs are explicit tensors consumed by training or the closed-loop scheduler.',1500)
box(a,60,170,240,120,'Body context','[B,256]'.splitlines()+['body_feature 256→512→512'], '#f4effb','#7653a6')
box(a,60,420,240,120,'Grasp context',['[B,256]','phase feature -> [B,64]'], '#f4effb','#7653a6')
box(a,60,650,240,100,'Hand state',['[B,28] -> [B,128]'], '#fff8ec','#bd7b20')
box(a,430,120,330,200,'Body prediction head',['MLP 512→512→640','reshape 640 = 10 × 64','body_tokens [B,10,64]','post-FSQ SONIC token target'], '#eaf7ef','#4b8a5a')
box(a,430,395,330,230,'Hand prediction head',['concat grasp [256] + phase [64] + hand [128]','input [B,448]','Linear 448→512→GELU','MLP 512→512→700','reshape 700 = 50 × 14','hand_ref [B,50,14]'], '#eaf7ef','#4b8a5a')
box(a,430,690,330,90,'Auxiliary outputs',['phase logits [B,3]','progress sigmoid [B]'], '#fceff0','#b44351')
box(a,930,180,390,150,'Grasp-readiness output (V4.1)',['grasp_ready_head: Linear(256,1)','logit [B,1]','used as trigger/readiness training signal'], '#fceff0','#b44351')
box(a,930,420,390,180,'Scheduler interpretation',['body token horizon = 10 slots','hand reference horizon = 50 slots','plans begin every 5 control ticks','absolute tick identity checked at consumption'], '#fff8ec','#bd7b20')
line(a,300,230,430,220); line(a,300,475,430,500); line(a,300,700,430,510); line(a,760,220,930,220); line(a,760,500,930,500); line(a,760,735,930,500)
text(a,60,790,'Source: va_research/models/v4/model.py and va_research/models/v4_1/model.py',14,'400','#516171')
save('prediction_heads.svg',a)
# 9 auxiliary
a=begin('auxiliary_tasks.svg',1500,850,'va_research/models/v4/losses.py; va_research/models/v4_1/losses.py; outputs/va_v4_1_kimodo_rule_20261009/history.json'); title(a,'Figure 9. Auxiliary task heads, labels and loss connections','V4.1 adds readiness, velocity and reopening terms to the base V4 objective.',1500)
box(a,45,120,290,150,'Body-token label',['target body_tokens [B,10,64]','smooth L1 loss','weight 1.0'], '#eef5fb','#2878b5')
box(a,45,335,290,150,'Hand-reference label',['target hand_ref [B,50,14]','MSE loss','weight 1.0'], '#eef5fb','#2878b5')
box(a,45,550,290,170,'Derived phase/progress labels',['phase from right-hand reference projection','OPEN <= .05; HOLD >= .95','progress clipped to [0,1]','phase CE weight .1; progress MSE .1'], '#fff8ec','#bd7b20')
box(a,445,125,320,130,'Primary body loss',['pred body_tokens','smooth-L1(body_pred, body_target)','L_body'], '#f4effb','#7653a6')
box(a,445,340,320,130,'Primary hand loss',['pred hand_ref','MSE(hand_pred, hand_target)','L_hand'], '#f4effb','#7653a6')
box(a,445,555,320,150,'Phase and progress losses',['pred phase_logits -> CE','pred closure_progress -> MSE','L_phase, L_progress'], '#f4effb','#7653a6')
box(a,880,110,300,145,'V4.1 readiness loss',['grasp_ready_logit','label 1 iff phase >= CLOSING','binary cross entropy','weight .1'], '#fceff0','#b44351')
box(a,880,315,300,145,'Velocity consistency',['hand_ref[t+1]-hand_ref[t]','smooth L1 to target velocity','weight .1'], '#fceff0','#b44351')
box(a,880,520,300,145,'Reopening penalty',['active phases CLOSING/HOLD','penalize negative predicted velocity','weight .02'], '#fceff0','#b44351')
box(a,1260,260,180,230,'Total objective',['L = L_body + L_hand',' + .1 L_phase + .1 L_progress',' + .1 L_trigger',' + .1 L_velocity',' + .02 L_reopen'], '#eaf7ef','#4b8a5a')
for y in [190,410,630]: line(a,335,y,445,y)
for y in [190,410,630]: line(a,765,y,1260,365)
for y in [180,385,590]: line(a,1180,y,1260,365)
text(a,45,790,'The final V4.1 history records trigger, velocity and reopen terms at epochs 1,10,...,50.',14,'400','#516171')
save('auxiliary_tasks.svg',a)
# 10 closed-loop
a=begin('closed_loop_inference.svg',1600,900,'va_research/models/v4/runtime.py; va_research/models/v4_1/model.py; outputs/va_v4_1_kimodo_rule_20261009/closed_loop/{report.json,provenance.json}'); title(a,'Figure 10. Closed-loop inference and action scheduling','Absolute ticks, native token handoff and independent physical diagnostics.',1600)
box(a,45,150,250,190,'Observation capture',['head_rgb at 10 Hz','body [68], hand [28]','observation IDs','MTT S4000 EGL renderer'], '#eef5fb','#2878b5')
box(a,355,150,250,190,'V4.1 inference',['normalize state','forward policy','body [10,64] + hand [50,14]','phase/progress/readiness'], '#f4effb','#7653a6')
box(a,665,150,250,190,'Absolute-tick PlanBuffer',['first start anchor + lead=2','next plans start +5 ticks','reject duplicate/skip/uncovered','record deadlines'], '#fff8ec','#bd7b20')
box(a,975,150,250,190,'External token adapter',['pack each 64D token','publish to SONIC/native decoder','strict consumed tick identity'], '#fff8ec','#bd7b20')
box(a,1285,150,270,190,'Actuation and physics',['SONIC body command','independent hand PD','DDS LowCmd/LowState','MuJoCo TabletopDDS'], '#eaf7ef','#4b8a5a')
for x in [295,605,915,1225]: line(a,x,245,x+60,245)
text(a,55,430,'Scheduler and task gates',19,'700','#152238')
box(a,55,470,300,170,'Runtime integrity',['recorded ticks = 810','expected ticks = 810','uncovered = 0','deadline misses = 1','strict_runtime_pass = true'], '#eaf7ef','#4b8a5a')
box(a,405,470,300,170,'Physical criterion',['minimum lift = 0.1 m','continuous hold = 2 s','thumb + index/middle contacts','no cube/table contact; no fall'], '#fff8ec','#bd7b20')
box(a,755,470,300,170,'Recorded representative result',['max lift = 0.019827 m','final lift = -0.000108 m','hold = 0 s','physical_grasp_success = false'], '#fceff0','#b44351')
box(a,1105,470,400,170,'Failure diagnostics',['cube remained in table contact','robot-table contact steps = 107','robot self-contact steps = 640','fall steps = 0','renderer hardware verified'], '#fceff0','#b44351')
text(a,55,750,'Runtime pass and physical task pass are separate gates; the representative rollout passed the former and failed the latter.',15,'700','#203a50')
save('closed_loop_inference.svg',a)
# ---------- Quantitative plot figures ----------
# common chart
def chart_axes(a,x0,y0,w,h,xlab,ylab,xmax,ymax,xticks,yticks):
    line(a,x0,y0,x0,y0-h,'#34495e',2,'',''); line(a,x0,y0,x0+w,y0,'#34495e',2,'','')
    for xv in xticks:
      x=x0+w*xv/xmax; line(a,x,y0,x,y0-h,'#e1e6eb',1,'4 4',''); text(a,x,y0+22,str(xv),12,'400','#516171','middle')
    for yv in yticks:
      y=y0-h*yv/ymax; line(a,x0,y,x0+w,y,'#e1e6eb',1,'4 4',''); text(a,x0-10,y+4,f'{yv:g}',12,'400','#516171','end')
    text(a,x0+w/2,y0+48,xlab,14,'700','#203a50','middle'); text(a,x0-55,y0-h/2,ylab,14,'700','#203a50','middle')
# 11 dataset comp
a=begin('dataset_composition.svg',1400,760,'expert_datasets/train_dataset_v0/manifest.json; expert_datasets/train_dataset_v0/split.json'); title(a,'Figure 11. Dataset composition and coverage','Manifest-level counts for the accepted V4 training dataset.',1400)
m=json.loads((ROOT/'expert_datasets/train_dataset_v0/manifest.json').read_text()); s=json.loads((ROOT/'expert_datasets/train_dataset_v0/split.json').read_text())
counts=collections.Counter(e.get('metadata',{}).get('producer_type','unknown') for e in m['episodes']); samples=collections.defaultdict(int)
for e in m['episodes']: samples[e.get('metadata',{}).get('producer_type','unknown')]+=e.get('sample_count',0)
text(a,45,145,'Episodes by producer type',18,'700','#152238')
maxc=max(counts.values()); x0,y0,w,h=100,390,560,190
for i,(k,v) in enumerate(sorted(counts.items())):
 y=210+i*85; text(a,100,y+22,k,16,'700','#203a50'); rect(a,270,y,w* v/maxc,38,'#2878b5' if k=='kimodo' else '#4b8a5a','#ffffff',1,5); text(a,285+w*v/maxc,y+26,f'{v} episodes',15,'700','#203a50')
text(a,760,145,'Samples by producer type',18,'700','#152238'); maxs=max(samples.values());
for i,(k,v) in enumerate(sorted(samples.items())):
 y=210+i*85; text(a,760,y+22,k,16,'700','#203a50'); rect(a,930,y,w* v/maxs,38,'#2878b5' if k=='kimodo' else '#4b8a5a','#ffffff',1,5); text(a,945+w*v/maxs,y+26,f'{v} samples',15,'700','#203a50')
box(a,100,520,350,110,'Manifest total',[f"episodes: {len(m['episodes'])}",f"samples: {m['sample_count']}",f"tensor shape: [9488,...]"],'#f7f9fb','#7a8791','#203a50')
box(a,525,520,350,110,'Explicit split',[f"train episodes: {len(s['train_episodes'])}",f"validation episodes: {len(s['val_episodes'])}",f"train/val rows: {s['train_samples']}/{s['val_samples']}"],'#f7f9fb','#7a8791','#203a50')
box(a,950,520,350,110,'Coverage',["reference rate: 50 Hz","episode length: 810 frames","RGB model input: 128×128"],'#f7f9fb','#7a8791','#203a50')
text(a,45,700,'Counts are directly read from manifest metadata; no duplicate identity expansion is applied in this figure.',13,'400','#516171')
save('dataset_composition.svg',a)
# 12 training curves
a=begin('training_validation_curves.svg',1500,850,'outputs/va_v4_1_kimodo_rule_20261009/history.json'); title(a,'Figure 12. Recorded V4.1 training and validation curves','Only recorded checkpoints (epochs 1, 10, 20, 30, 40, 50) are plotted.',1500)
hist=json.loads((ROOT/'outputs/va_v4_1_kimodo_rule_20261009/history.json').read_text()); epochs=[r['epoch'] for r in hist]
# panel val loss
x0,y0,w,h=120,390,560,260; ymax=max(r['val_loss'] for r in hist)*1.15
chart_axes(a,x0,y0,w,h,'epoch','validation loss',max(epochs),ymax,[0,10,20,30,40,50],[0,0.002,0.004,0.006])
pts=[]
for r in hist: pts.append((x0+w*r['epoch']/max(epochs),y0-h*r['val_loss']/ymax))
path(a,'M '+' L '.join(f'{x:.1f},{y:.1f}' for x,y in pts),'#2878b5',3)
for x,y in pts: a.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2878b5"/>')
# hand and phase
x0b,y0b,w2,h2=850,390,560,260; chart_axes(a,x0b,y0b,w2,h2,'epoch','metric',max(epochs),1.0,[0,10,20,30,40,50],[0,0.25,0.5,0.75,1.0])
for key,col,label in [('val_phase_accuracy','#4b8a5a','phase accuracy'),('val_trigger_rate','#b44351','trigger rate')]:
 pts=[(x0b+w2*r['epoch']/max(epochs),y0b-h2*r[key]) for r in hist]; path(a,'M '+' L '.join(f'{x:.1f},{y:.1f}' for x,y in pts),col,3)
 for x,y in pts: a.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}"/>')
text(a,120,760,'Left: val_loss. Right: val_phase_accuracy and mean trigger probability recorded by V4.1 trainer.',14,'400','#516171')
# legend
for i,(col,label) in enumerate([('#2878b5','val loss'),('#4b8a5a','phase accuracy'),('#b44351','trigger rate')]): line(a,120+i*270,805,155+i*270,805,col,4,'',''); text(a,165+i*270,810,label,13,'400','#516171')
save('training_validation_curves.svg',a)
# 13 offline prediction metrics
a=begin('offline_prediction_results.svg',1500,880,'outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json'); title(a,'Figure 13. Offline prediction results','Metrics are read from the final V4.1 offline_metrics.json; values are not physical-task results.',1500)
o=json.loads((ROOT/'outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json').read_text())
metrics=[('token MAE',o['token_mae'],0.01),('hand MAE',o['hand_mae'],0.01),('progress MAE',o['progress_mae'],0.01),('FSQ scalar accuracy',o['fsq_scalar_accuracy'],1.0),('whole token exact accuracy',o['whole_token_exact_accuracy'],1.0),('phase accuracy',o['phase_accuracy'],1.0)]
text(a,60,145,'Final checkpoint metrics',18,'700','#152238')
for i,(lab,val,scale) in enumerate(metrics):
 y=190+i*70; text(a,60,y+22,lab,15,'700','#203a50'); rect(a,300,y,760*val/scale,34,'#2878b5' if scale<0.1 else '#4b8a5a','#ffffff',1,5); text(a,1075,y+24,f'{val:.9f}',14,'700','#203a50')
text(a,60,650,'Body-token MAE by horizon (10 future token slots)',18,'700','#152238')
vals=o['body_mae_by_horizon']; x0,y0,w,h=120,820,1240,130; ymax=max(vals)*1.15
chart_axes(a,x0,y0,w,h,'future slot','body MAE',10,ymax,list(range(1,11)),[0,0.001,0.002,0.003,0.004,0.005])
pts=[(x0+w*(i+1)/10,y0-h*v/ymax) for i,v in enumerate(vals)]; path(a,'M '+' L '.join(f'{x:.1f},{y:.1f}' for x,y in pts),'#7653a6',3)
for i,(x,y) in enumerate(pts): a.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#7653a6"/>')
text(a,60,875,'Source metrics include token_mse, moving_right_hand_mae and confusion matrix in offline_metrics.json.',13,'400','#516171')
save('offline_prediction_results.svg',a)
# 14 native physical evaluation
a=begin('native_physical_evaluation.svg',1500,850,'outputs/va_v4_1_kimodo_rule_20261009/closed_loop/report.json; outputs/va_v4_1_kimodo_rule_20261009/closed_loop/renderer_worker.json'); title(a,'Figure 14. Native physical evaluation and failure diagnostics','Representative V4.1 rollout: complete strict runtime trace, failed lift/hold criterion.',1500)
r=json.loads((ROOT/'outputs/va_v4_1_kimodo_rule_20261009/closed_loop/report.json').read_text()); rw=json.loads((ROOT/'outputs/va_v4_1_kimodo_rule_20261009/closed_loop/renderer_worker.json').read_text())
# metrics bars
text(a,50,140,'Physical criterion vs recorded result',18,'700','#152238')
rows=[('Maximum lift (m)',r['max_lift_m'],r['criterion']['minimum_lift_m']),('Continuous hold (s)',r['max_continuous_hold_seconds'],r['criterion']['minimum_continuous_hold_s'])]
for i,(lab,val,thr) in enumerate(rows):
 y=185+i*90; text(a,50,y+25,lab,15,'700','#203a50'); rect(a,300,y,700*val/thr if thr else 0,36,'#b44351','#ffffff',1,5); line(a,300+700,y-8,300+700,y+50,'#203a50',3,'',''); text(a,1020,y+25,f'recorded {val:.6f}; threshold {thr:g}',14,'700','#203a50')
box(a,50,420,330,170,'Scheduler integrity',[f"ticks {r['recorded_ticks']}/{r['expected_ticks']}",f"uncovered {r['scheduler']['uncovered_ticks']}",f"deadline misses {r['scheduler']['deadline_misses']}",f"strict pass {r['strict_runtime_pass']}"],'#eaf7ef','#4b8a5a','#203a50')
box(a,430,420,330,170,'Contact diagnostics',[f"robot-table steps {r['robot_table_contact_steps']}",f"robot self-contact steps {r['robot_self_contact_steps']}",f"fall steps {r['fall_steps']}",f"cube attached {r['cube_attached']}"],'#fceff0','#b44351','#203a50')
box(a,810,420,300,170,'Renderer evidence',[f"vendor {rw['gl_vendor']}",f"renderer {rw['gl_renderer']}",f"hardware verified {rw['hardware_verified']}",f"camera {rw['camera']}"],'#eef5fb','#2878b5','#203a50')
box(a,1160,420,290,170,'Task outcome',[f"physical success {r['physical_grasp_success']}",f"final lift {r['final_lift_m']:.6f} m",f"hold {r['final_continuous_hold_seconds']:.1f} s",'runtime completed'],'#fceff0','#b44351','#203a50')
text(a,50,700,'Recorded result: insufficient lift/contact state; no causal decomposition is established. No fall occurred and the scheduler covered every tick.',14,'700','#203a50')
save('native_physical_evaluation.svg',a)
# 15 development evolution
a=begin('model_development_evolution.svg',1700,900,'outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json; outputs/va_v4_ablation/*/offline_metrics.json'); title(a,'Figure 15. Model development and version evolution','Recorded offline ablations isolate fusion and auxiliary-head changes under the same dataset split.',1700)
entries=[('V4.1 baseline','outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json'),('No phase','outputs/va_v4_ablation/AUX-1_no_phase/offline_metrics.json'),('No progress','outputs/va_v4_ablation/AUX-2_no_progress/offline_metrics.json'),('No phase + progress','outputs/va_v4_ablation/AUX-3_no_phase_progress/offline_metrics.json'),('MLP fusion','outputs/va_v4_ablation/ARCH-1_mlp_fusion/offline_metrics.json'),('Proprio only','outputs/va_v4_ablation/ARCH-2_proprio_only/offline_metrics.json')]
vals=[]
for lab,p in entries:
 q=json.loads((ROOT/p).read_text()); vals.append((lab,q['token_mae'],q['hand_mae'],q['phase_accuracy'],q['progress_mae']))
text(a,60,102,'Historical model sequence (version docs; no cross-version metric interpolation)',14,'700','#203a50')
for x,wv,label,fill,stroke in [(520,130,'V1 / V2','#f7f9fb','#7a8791'),(705,130,'V3','#f7f9fb','#7a8791'),(890,130,'V4','#eef5fb','#2878b5'),(1075,150,'V4.1 (final)','#f4effb','#7653a6')]:
 rect(a,x,82,wv,34,fill,stroke,2,8); text(a,x+wv/2,104,label,14,'700','#203a50','middle')
for x1,x2 in [(650,700),(835,885),(1020,1070)]: line(a,x1,99,x2,99,'#34495e',2,'','url(#arrow)')
text(a,60,140,'Token MAE (lower is better)',18,'700','#152238'); x0,y0,w,h=320,480,1200,290; ymax=max(v[1] for v in vals)*1.25
chart_axes(a,x0,y0,w,h,'variant','token MAE',len(vals),ymax,list(range(1,len(vals)+1)),[0,0.001,0.002,0.003,0.004,0.005])
barw=130
for i,(lab,tok,hand,phase,prog) in enumerate(vals):
 x=x0+w*(i+0.5)/len(vals); rect(a,x-barw/2,y0-h*tok/ymax,barw,h*tok/ymax,'#2878b5','#ffffff',1,4); text(a,x,y0+24,lab,12,'400','#203a50','middle'); text(a,x,y0-h*tok/ymax-8,f'{tok:.4f}',12,'700','#203a50','middle')
text(a,60,650,'Associated offline metrics',18,'700','#152238')
# table
headers=['Variant','Token MAE','Hand MAE','Phase accuracy','Progress MAE']; colx=[60,360,580,800,1080]
for x,hdr in zip(colx,headers): text(a,x,690,hdr,14,'700','#203a50')
for i,(lab,tok,hand,phase,prog) in enumerate(vals):
 y=720+i*27; text(a,colx[0],y,lab,13,'400','#233240'); text(a,colx[1],y,f'{tok:.6f}',13,'400','#233240'); text(a,colx[2],y,f'{hand:.6f}',13,'400','#233240'); text(a,colx[3],y,f'{phase:.6f}',13,'400','#233240'); text(a,colx[4],y,f'{prog:.6f}',13,'400','#233240')
text(a,60,885,'These are offline predictions only; the representative native physical result remains the separate Figure 14 evidence.',13,'400','#516171')
save('model_development_evolution.svg',a)
# ---------- tables ----------
def write_csv(name,headers,rows):
 with (TAB/name).open('w',newline='') as f:
  w=csv.writer(f); w.writerow(headers); w.writerows(rows)
# architecture table
write_csv('model_architecture.csv',['stage','operation','input_shape','output_shape','notes','source'],[
 ['input','RGB observation','[B,3,128,128]','[B,3,128,128]','RGB converted/resized to model resolution','va_research/models/v4/model.py'],
 ['visual_1','Conv2d(3,32,3,stride=2,pad=1)+GroupNorm(8,32)+GELU','[B,3,128,128]','[B,32,64,64]','stride-2','va_research/models/v4/model.py'],
 ['visual_2','Conv2d(32,64,3,stride=2,pad=1)+GroupNorm(8,64)+GELU','[B,32,64,64]','[B,64,32,32]','stride-2','va_research/models/v4/model.py'],
 ['visual_3','Conv2d(64,128,3,stride=2,pad=1)+GroupNorm(8,128)+GELU','[B,64,32,32]','[B,128,16,16]','stride-2','va_research/models/v4/model.py'],
 ['visual_4','Conv2d(128,256,3,stride=2,pad=1)+GroupNorm(8,256)+GELU','[B,128,16,16]','[B,256,8,8]','stride-2','va_research/models/v4/model.py'],
 ['visual_tokens','flatten(8x8), transpose, learned position','[B,256,8,8]','[B,64,256]','64 spatial tokens','va_research/models/v4/model.py'],
 ['body_query','MLP 68->256->256 + LayerNorm + GELU','[B,68]','[B,1,256]','query unsqueeze','va_research/models/v4/model.py'],
 ['grasp_query','MLP 68->256->256 + LayerNorm + GELU','[B,68]','[B,1,256]','query unsqueeze','va_research/models/v4/model.py'],
 ['body_cross','4-head MultiheadAttention + residual/FF','Q [B,1,256], V [B,64,256]','[B,256]','FF 256->512->256','va_research/models/v4/model.py'],
 ['grasp_cross','4-head MultiheadAttention + residual/FF','Q [B,1,256], V [B,64,256]','[B,256]','FF 256->512->256','va_research/models/v4/model.py'],
 ['body_head','MLP 256->512->512; MLP 512->512->640','[B,256]','[B,10,64]','10 body tokens','va_research/models/v4/model.py'],
 ['hand_state','MLP 28->128->128 + GELU','[B,28]','[B,128]','current hand q/dq','va_research/models/v4/model.py'],
 ['hand_head','Linear 448->512; MLP 512->512->700','[B,448]','[B,50,14]','50 hand references','va_research/models/v4/model.py'],
 ['phase_head','Linear 64->3','[B,64]','[B,3]','phase logits','va_research/models/v4/model.py'],
 ['progress_head','Linear 64->1 + sigmoid','[B,64]','[B]','closure progress','va_research/models/v4/model.py'],
 ['grasp_ready_head','Linear 256->1','[B,256]','[B,1]','V4.1 readiness logit','va_research/models/v4_1/model.py'],
])
# training config
write_csv('training_configuration.csv',['field','value','source'],[
 ['model_family','V4.1 (V4 policy + readiness head)','va_research/models/v4_1/model.py'],['dataset','expert_datasets/train_dataset_v0','expert_datasets/train_dataset_v0/manifest.json + expert_datasets/train_dataset_v0/split.json'],['epochs','50','outputs/va_v4_1_kimodo_rule_20261009/training_status.json'],['parameters','3502977','outputs/va_v4_1_kimodo_rule_20261009/training_status.json'],['train_samples','7530','outputs/va_v4_1_kimodo_rule_20261009/training_status.json'],['val_samples','1958','outputs/va_v4_1_kimodo_rule_20261009/training_status.json'],['optimizer','AdamW, lr=3e-4, weight_decay=0','va_research/models/v4_1/train_mixed.py'],['batch_size','32','va_research/models/v4_1/train_mixed.py'],['gradient_clip','10','va_research/models/v4_1/train_mixed.py'],['seed','4','va_research/models/v4_1/train_mixed.py'],['state_normalization','train mean/std; std clamp_min(0.05)','va_research/models/v4_1/train_mixed.py'],['loss_weights','body=1, hand=1, phase=.1, progress=.1, trigger=.1, velocity=.1, reopen=.02','va_research/models/v4_1/train_mixed.py'],['checkpoint','outputs/va_v4_1_kimodo_rule_20261009/checkpoint.pt','outputs/va_v4_1_kimodo_rule_20261009/training_status.json'],])
# dataset stats
write_csv('dataset_statistics.csv',['quantity','value','source'],[['episodes_total',len(m['episodes']),'expert_datasets/train_dataset_v0/manifest.json'],['samples_total',m['sample_count'],'expert_datasets/train_dataset_v0/manifest.json'],['kimodo_episodes',counts['kimodo'],'expert_datasets/train_dataset_v0/manifest.json'],['rule_based_episodes',counts['rule_based'],'expert_datasets/train_dataset_v0/manifest.json'],['kimodo_samples',samples['kimodo'],'expert_datasets/train_dataset_v0/manifest.json'],['rule_based_samples',samples['rule_based'],'expert_datasets/train_dataset_v0/manifest.json'],['train_episodes',len(s['train_episodes']),'expert_datasets/train_dataset_v0/split.json'],['val_episodes',len(s['val_episodes']),'expert_datasets/train_dataset_v0/split.json'],['train_samples',s['train_samples'],'expert_datasets/train_dataset_v0/split.json'],['val_samples',s['val_samples'],'expert_datasets/train_dataset_v0/split.json'],['reference_hz',50,'expert_datasets/train_dataset_v0/manifest.json'],['kimodo_frames_per_episode',810,'expert_datasets/train_dataset_v0/manifest.json'],['rule_based_frames_per_episode',800,'expert_datasets/train_dataset_v0/manifest.json'],['rgb_model_resolution','128x128','expert_datasets/train_dataset_v0/manifest.json']])
# offline evaluation
write_csv('offline_evaluation.csv',['metric','value','source'],[(k,v,'outputs/va_v4_1_kimodo_rule_20261009/offline_metrics.json') for k,v in [('token_mae',o['token_mae']),('token_mse',o['token_mse']),('fsq_scalar_accuracy',o['fsq_scalar_accuracy']),('whole_token_exact_accuracy',o['whole_token_exact_accuracy']),('hand_mae',o['hand_mae']),('moving_right_hand_mae',o['moving_right_hand_mae']),('phase_accuracy',o['phase_accuracy']),('progress_mae',o['progress_mae']),('offline_gate',o['offline_gate']),('validation_overlap',o['overlap']['validation_overlap']),('evaluation_epoch',o['epoch'])]])
# physical evaluation table
phys_rows=[('physical_grasp_success',r['physical_grasp_success']),('max_lift_m',r['max_lift_m']),('final_lift_m',r['final_lift_m']),('max_continuous_hold_s',r['max_continuous_hold_seconds']),('recorded_ticks',r['recorded_ticks']),('expected_ticks',r['expected_ticks']),('uncovered_ticks',r['scheduler']['uncovered_ticks']),('repeated_ticks',r['scheduler']['repeated_ticks']),('skipped_ticks',r['scheduler']['skipped_ticks']),('deadline_misses',r['scheduler']['deadline_misses']),('strict_runtime_pass',r['strict_runtime_pass']),('robot_table_contact_steps',r['robot_table_contact_steps']),('robot_self_contact_steps',r['robot_self_contact_steps']),('fall_steps',r['fall_steps']),('minimum_lift_threshold_m',r['criterion']['minimum_lift_m']),('minimum_hold_threshold_s',r['criterion']['minimum_continuous_hold_s'])]
phys_out=[(k,v,'outputs/va_v4_1_kimodo_rule_20261009/closed_loop/report.json') for k,v in phys_rows]
phys_out.append(('renderer',rw['gl_renderer'],'outputs/va_v4_1_kimodo_rule_20261009/closed_loop/renderer_worker.json'))
write_csv('physical_evaluation.csv',['metric','value','source'],phys_out)
# ablation table
rows=[]
for lab,p in entries:
 q=json.loads((ROOT/p).read_text()); rows.append([lab,q.get('variant','cross_attention'),q.get('auxiliary_ablation','all'),q['token_mae'],q['hand_mae'],q['phase_accuracy'],q['progress_mae'],q.get('offline_gate','') ,p])
write_csv('model_ablation.csv',['variant_label','fusion_variant','auxiliary_ablation','token_mae','hand_mae','phase_accuracy','progress_mae','offline_gate','source'],rows)
print('generated',len(list(FIG.glob('*.svg'))),'figures and',len(list(TAB.glob('*.csv'))),'tables')
