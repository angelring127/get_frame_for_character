# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-based Python project for worksheet frame detection, character matching, and OCR.

- Core entry point: `main.py` (`flag=1` writing, `flag=2` reading).
- Detection/processing modules: `reading_detector.py`, `writing_detector.py`, `image_preprocessor.py`, `proc*.py`.
- OCR integration: `ocr_api.py` (external API call).
- Test and validation scripts: `test_kanji_matching.py`, `test.py`.
- Data and generated artifacts:
  - Inputs/samples: `images/`, `samples/`
  - Intermediate outputs: `extracted/`
  - Template/output images: `output/`
  - Docs/notes: `*.md` in repo root.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt`  
  Install runtime dependencies (OpenCV, NumPy, scikit-learn, pytesseract, etc.).
- `python main.py <image_path> <flag> [template_name]`  
  Run detection pipeline. Example: `python main.py images/answer01_01.jpg 1 template`.
- `python test_kanji_matching.py`  
  Run kanji similarity validation against files in `images/`.
- `python test.py`  
  Run template-fill smoke test.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables and lowercase module filenames.
- Keep image-processing steps explicit and small; prefer helper functions over long inline blocks.
- Add concise comments only where algorithm intent is non-obvious.

## Testing Guidelines
- Tests are currently script-based (not `pytest` suites).
- Name new test scripts `test_*.py` and keep required sample data under `images/` or `samples/`.
- Verify both console output and generated artifacts in `extracted/` and `output/`.

## Commit & Pull Request Guidelines
- Current history uses short, purpose-driven subjects (often multilingual), e.g. fixing template placement or matching order.
- Recommended commit format: `<area>: <what changed>` (imperative, specific).
- PRs should include:
  - What changed and why
  - How to run/verify (`python ...` commands)
  - Before/after images when detection or layout behavior changes
  - Linked issue or task ID when available

## Security & Configuration Tips
- Do not commit API keys or private endpoints.
- Avoid committing large generated images unless needed for regression evidence.
