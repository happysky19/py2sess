# PyPI Release Checklist

This repository is not published to PyPI yet. When you are ready, use this checklist.

## 1. Finalize package metadata

Make sure the following fields in `pyproject.toml` are correct:

- `version`
- `description`
- `authors`
- `license`
- `requires-python`
- `project.urls`
- `classifiers`
- dependency extras

## 2. Build and verify distributions locally

From the repository root:

```bash
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine check dist/*
```

This should create:

- a source distribution (`sdist`)
- a wheel (`bdist_wheel`)

## 3. Test an install from the built wheel

Use a clean virtual environment:

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python3 -m pip install dist/*.whl
python3 -c "import py2sess; print(py2sess.__all__)"
```

Then run a small smoke test or one of the example scripts.

## 4. Upload to TestPyPI first

```bash
python3 -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment and verify:

- imports work
- packaged benchmark fixtures are included
- examples run

## 5. Upload to PyPI

```bash
python3 -m twine upload dist/*
```

## 6. Tag the release in Git

After the upload succeeds, create a matching Git tag, for example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Recommended pre-release checks

- run `python3 -m unittest discover -s tests -v`
- run all example scripts
- verify the package data files are present in the built wheel
- confirm the README renders correctly on PyPI
