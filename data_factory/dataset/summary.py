import html,json
from pathlib import Path
def build(experiment,links,output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=''.join(f'<li><a href="{html.escape(str(v))}">{html.escape(k)}</a></li>' for k,v in links.items());(out/'index.html').write_text(f'<html><meta charset="utf-8"><h1>Experiment {html.escape(experiment)}</h1><ul>{rows}</ul></html>');return out/'index.html'
