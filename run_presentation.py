"""Execute the notebook from a fresh kernel and export a readable HTML copy."""
from pathlib import Path
import argparse
import json
import os
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--in-process",action="store_true",help="Execute cells with IPython without opening kernel sockets.")
args=parser.parse_args()
path=ROOT/"VAERS_Presentation.ipynb"
nb=nbformat.read(path,as_version=4)
for cell in nb.cells:
    if cell.cell_type=="code":
        cell.outputs=[]
        cell.execution_count=None
if args.in_process:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT)
    shell=TerminalInteractiveShell.instance()
    shell.display_formatter.active_types=["text/plain","text/html","image/png","image/svg+xml","text/latex"]
    count=0
    for cell in nb.cells:
        if cell.cell_type!="code":
            continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        cell.execution_count=count
        stdout=captured.stdout.replace("No event loop hook running.\n","")
        if stdout:
            cell.outputs.append(nbformat.v4.new_output("stream",name="stdout",text=stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream",name="stderr",text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data",data=output.data,metadata=output.metadata))
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout,captured.stderr)
            raise RuntimeError(f"Notebook cell {count} failed") from (result.error_before_exec or result.error_in_exec)
        print(f"Executed code cell {count}",flush=True)
    engine="IPython in a fresh Python process; no kernel sockets"
else:
    client=NotebookClient(nb,timeout=1200,kernel_name="python3",allow_errors=False,
                          resources={"metadata":{"path":str(ROOT)}})
    client.execute()
    engine="Jupyter kernel via nbclient"
nb.metadata["execution_engine"]=engine
report_path=ROOT/"execution_report.json"
report=json.loads(report_path.read_text())
report["notebook_execution_engine"]=engine
report_path.write_text(json.dumps(report,indent=2)+"\n")
nbformat.validate(nb)
nbformat.write(nb,path)
exporter=HTMLExporter(template_name="lab")
exporter.exclude_input=True
exporter.exclude_input_prompt=True
exporter.exclude_output_prompt=True
html,_=exporter.from_notebook_node(nb)
style='''<style>
body {background:#f4f7fa!important;color:#17364d;}
.jp-Notebook {max-width:1120px;margin:28px auto;padding:38px 48px!important;background:white;border-radius:12px;box-shadow:0 3px 24px #17364d0c;}
.jp-RenderedHTMLCommon {font-size:16px;line-height:1.65;}
.jp-RenderedHTMLCommon h1 {font-size:34px;color:#17364d;line-height:1.2;}
.jp-RenderedHTMLCommon h2 {font-size:24px;color:#17364d;margin-top:34px;}
.jp-RenderedHTMLCommon table {font-size:14px!important;margin:16px 0!important;}
.jp-RenderedHTMLCommon th {background:#edf3f8!important;}
.jp-RenderedHTMLCommon td,.jp-RenderedHTMLCommon th {padding:10px 14px!important;}
.jp-OutputArea-output img {max-width:100%;height:auto;}
.jp-InputArea {display:none;}
@media print {.jp-Notebook{box-shadow:none;margin:0;padding:10px!important;}h2{break-after:avoid;}img,table{break-inside:avoid;}}
</style>'''
html=html.replace('</head>',style+'\n</head>')
(ROOT/"VAERS_Presentation.html").write_text(html)
print(json.dumps({"executed_code_cells":sum(c.cell_type=="code" for c in nb.cells),
    "errors":sum(o.output_type=="error" for c in nb.cells if c.cell_type=="code" for o in c.outputs),
    "embedded_figures":sum("image/png" in o.get("data",{}) for c in nb.cells if c.cell_type=="code" for o in c.outputs),
    "notebook_bytes":path.stat().st_size,"html_bytes":(ROOT/"VAERS_Presentation.html").stat().st_size},indent=2))
