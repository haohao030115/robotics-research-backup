import html,json
from pathlib import Path
from .contracts import read

def training_report(folder):
 p=Path(folder);m=read(p/'manifest.json');h=read(p/'history.json');s=read(p/'status.json');
 metrics=h[-1] if h else {}; text='<html><meta charset="utf-8"><title>V4 Training Report</title><body><h1>V4 Training Report</h1>'
 text+=f'<p>Status: {html.escape(str(s))}</p><h2>History</h2><table border=1><tr><th>epoch</th><th>train loss</th><th>val loss</th><th>token MAE</th><th>hand MAE</th><th>phase acc</th><th>progress MAE</th><th>lr</th></tr>'
 for r in h:
  t=r['train_metrics'];v=r.get('val_metrics') or {};text+=f"<tr><td>{r['epoch']}</td><td>{r['train_loss']:.6g}</td><td>{r['val_loss'] if r['val_loss'] is not None else 'unavailable'}</td><td>{t['token_mae']:.6g}</td><td>{t['hand_mae']:.6g}</td><td>{t['phase_accuracy']:.6g}</td><td>{t['progress_mae']:.6g}</td><td>{r['learning_rate']}</td></tr>"
 text+='</table><h2>Provenance</h2><pre>'+html.escape(json.dumps(m,indent=2))+'</pre><p>Unavailable metrics are not reconstructed by changing training semantics.</p></body></html>'; (p/'report.html').write_text(text);return s
