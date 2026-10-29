"""Package the native app, complete browser app, trained models and experiment records."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent.parent
native=root/'streamlit' if (root/'streamlit').exists() else root
web=root/'dist' if (root/'dist').exists() else root/'web'
models=web/'model' if native!=root else root/'model'
output=web/'spectraleaf-streamlit.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
    archive.write(root/'COPYRIGHT.txt','COPYRIGHT.txt')
    if native!=root:
        native_files=[p for p in native.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    else:
        native_files=[root/n for n in ['app.py','lab_tools.py','requirements.txt','model_config.example.json','README.md','.streamlit/config.toml'] if (root/n).exists()]
    for p in native_files:archive.write(p,p.relative_to(native))
    for p in models.rglob('*'):
        if p.is_file():archive.write(p,Path('model')/p.relative_to(models))
    for directory in ['ml','docs','scripts']:
        for p in (root/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:archive.write(p,p.relative_to(root))
    for p in web.rglob('*'):
        if p.is_file() and p!=output and not p.suffix=='.zip' and 'model' not in p.relative_to(web).parts:
            archive.write(p,Path('web')/p.relative_to(web))
    for p in models.rglob('*'):
        if p.is_file():archive.write(p,Path('web/model')/p.relative_to(models))
    for p in (root/'tests').glob('*.cjs'):
        archive.writestr(str(Path('tests')/p.name),p.read_text().replace('../dist/','../web/'))
    archive.writestr('.gitignore','__pycache__/\n*.py[cod]\n.venv/\ndata/\n*.partial\n')
    archive.writestr('.github/workflows/validate.yml','''name: Validate model and optical workbench
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - uses: actions/setup-node@v4
        with:
          node-version: '22'
      - run: node --check web/lab.js
      - run: node tests/optics-core.cjs
      - run: node tests/evaluation-core.cjs
      - run: node tests/study-core.cjs
      - run: node tests/study-replay.cjs
      - run: python -m py_compile app.py lab_tools.py
      - run: python ml/validate_bundle.py --public model
      - run: python -m pip install numpy==2.3.5 onnxruntime==1.31.1
      - run: python ml/check_runtime.py --public model
      - run: python ml/check_ablation.py --public model --runtime
''')
    assert archive.testzip() is None
print(f'{output}: {output.stat().st_size} bytes')
