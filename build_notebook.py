"""Execute the analysis in notebook cells and save actual captured outputs."""
from pathlib import Path
import base64
import contextlib
import io
import nbformat

root = Path(__file__).resolve().parent
source = (root / 'analysis.py').read_text(encoding='utf-8')
chunks = source.split('# %% ')[1:]
notebook = nbformat.v4.new_notebook()
notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}}
notebook.cells.append(nbformat.v4.new_markdown_cell(
    '# Customer churn homework\n\nExecuted Python analysis covering all 12 tasks. '
    'Run from the project directory; install requirements.txt first. '
    'churn_flag=1 is the positive class. All preprocessing and model selection use training data only.'))
namespace = {'__name__': '__main__'}
figures = {3: ['age_histograms.png'], 10: ['validation_curves.png', 'confusion_matrix.png']}
for count, chunk in enumerate(chunks, 1):
    title, code = chunk.split('\n', 1)
    notebook.cells.append(nbformat.v4.new_markdown_cell('## '+title))
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exec(compile(code, str(root / 'analysis.py'), 'exec'), namespace)
    outputs = []
    if stdout.getvalue():
        outputs.append(nbformat.v4.new_output('stream', name='stdout', text=stdout.getvalue()))
    if stderr.getvalue():
        outputs.append(nbformat.v4.new_output('stream', name='stderr', text=stderr.getvalue()))
    for filename in figures.get(count, []):
        outputs.append(nbformat.v4.new_output('display_data', data={
            'image/png': base64.b64encode((root / 'outputs' / filename).read_bytes()).decode(),
            'text/plain': filename}))
    notebook.cells.append(nbformat.v4.new_code_cell(code, execution_count=count, outputs=outputs))
report = (root / 'outputs' / 'report.md').read_text(encoding='utf-8')
for filename in ['age_histograms.png', 'validation_curves.png', 'confusion_matrix.png']:
    report = report.replace(']('+filename+')', '](data:image/png;base64,'+
        base64.b64encode((root / 'outputs' / filename).read_bytes()).decode()+')')
notebook.cells.append(nbformat.v4.new_markdown_cell(report))
nbformat.validate(notebook)
nbformat.write(notebook, root / 'customer_churn_analysis.ipynb')
print('Saved executed notebook with', len(chunks), 'code cells.')
