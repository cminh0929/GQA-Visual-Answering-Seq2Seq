# Plan: Code Reorganization

## Summary
Reorganize the project structure by moving supplementary scripts (`evaluate.py`, `visualize.py`, `comparison.py`) into appropriate subdirectories (`scripts/`, `tools/`). This will clean up the root directory and separate core execution entries from analytical tools.

## User Story
As a developer, I want a well-organized project structure, so that supporting scripts don't clutter the main root directory and follow a logical categorization.

## Problem → Solution
Messy root with 6+ `.py` files → Clean root with only main entries (`train.py`, `app.py`) and centralized config.

## Metadata
- **Complexity**: Small
- **Source PRD**: N/A
- **PRD Phase**: N/A
- **Estimated Files**: 5 (3 Moved, 1 Updated README, 1 New Directory)

---

## UX Design
N/A — internal change.

---

## Mandatory Reading

Files that MUST be read before implementing:

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 (critical) | `scripts/extract_features.py` | 24-26 | Pattern for relative imports in sub-scripts |
| P1 (important) | `config.py` | 1-20 | Verify BASE_DIR logic |
| P2 (reference) | `README.md` | all | Update usage instructions |

---

## Patterns to Mirror

### SCRIPT_IMPORT_PATTERN
// SOURCE: scripts/extract_features.py:24-26
```python
# Add root directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `evaluate.py` | MOVE to `scripts/` | Analytical script, belongs with other procedural scripts |
| `visualize.py` | MOVE to `tools/` | Visualization tool for results |
| `comparison.py` | MOVE to `tools/` | Analytical tool for comparing models |
| `README.md` | UPDATE | Reflected new paths in usage documentation |

## NOT Building
- Turning the project into a formal Python package (`pip install -e .`).
- Moving core modules (`data/`, `models/`, `utils/`) into a `src/` folder (keep them at root for now as they are well-organized).

---

## Step-by-Step Tasks

### Task 1: Create Directories
- **ACTION**: Create `tools/` directory.
- **IMPLEMENT**: `mkdir tools`
- **VALIDATE**: `ls tools` exists.

### Task 2: Move and Update `evaluate.py`
- **ACTION**: Move `evaluate.py` to `scripts/evaluate.py`.
- **IMPLEMENT**: Update its `sys.path.insert` to point one level up.
- **MIRROR**: `scripts/extract_features.py` import pattern.
- **VALIDATE**: Run `python scripts/evaluate.py --help` from root.

### Task 3: Move and Update `visualize.py`
- **ACTION**: Move `visualize.py` to `tools/visualize.py`.
- **IMPLEMENT**: Update its `sys.path.insert` to point one level up.
- **MIRROR**: `scripts/extract_features.py` import pattern.
- **VALIDATE**: Run `python tools/visualize.py --help` from root.

### Task 4: Move and Update `comparison.py`
- **ACTION**: Move `comparison.py` to `tools/comparison.py`.
- **IMPLEMENT**: Update its `sys.path.insert` to point one level up.
- **MIRROR**: `scripts/extract_features.py` import pattern.
- **VALIDATE**: Run `streamlit run tools/comparison.py` from root (manually).

### Task 5: Update README.md
- **ACTION**: Update all commands in `README.md` to reflect new paths.
- **IMPLEMENT**: Search and replace `python evaluate.py` with `python scripts/evaluate.py`, etc.
- **VALIDATE**: Read `README.md` and check all paths.

---

## Testing Strategy

### Validation Commands
- `python scripts/evaluate.py --help`
- `python tools/visualize.py --help`
- `streamlit run tools/comparison.py --help` (Check if st app starts)

### Manual Validation
- [ ] Check if `train.py` still runs correctly (root script).
- [ ] Check if `app.py` still runs correctly (root script).
- [ ] Ensure `results/` are still saved to the correct location via `config.py`.

---

## Acceptance Criteria
- [ ] Root directory contains only `app.py`, `train.py`, `config.py`, `README.md`, `requirements.txt`.
- [ ] `evaluate.py` is in `scripts/`.
- [ ] `visualize.py` and `comparison.py` are in `tools/`.
- [ ] All moved scripts run correctly without import errors.
- [ ] `README.md` accurately reflects the new structure.
