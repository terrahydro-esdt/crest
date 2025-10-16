# CREST timeline for GitHub release

## Documentation Organization Philosophy

1. **Tutorials** - Learning-oriented, step-by-step lessons (e.g. the notebooks)
2. **How-to Guides** - Task-oriented, problem-solving recipes (more notebooks?)
3. **Reference** - Information-oriented, API documentation (we've got this already)
4. **Explanation** - Understanding-oriented, conceptual background (we could use the AI-generated explanation)
5. **Model Zoo / Example Models** - Pre-trained models and configurations (expected in AI projects)

How much of this can we do?

## Tentative Implementation Roadmap for GitHub Release

### Phase 0: Documentation updates (2 weeks)

- [ ] **Update docstrings** - see DOC_GUIDE.md
- [ ] **Update notebooks** - see DOC_GUIDE.md
- [ ] **Update RST files**
- [ ] **Clean up code-base** - Understanding-oriented, conceptual background
  - Need a branching model
  - Remove all branches, leave `main`, `develop`, and "tags"

### Phase 1: Getting started (1 week)

- [x] Create DOC_GUIDE.md
- [x] Create CONTRIBUTING.md
- [x] Create CITATION.cff
- [x] Create PR template
- [ ] Update README.md
- [ ] Create CHANGELOG.md
- [ ] Create CODE_OF_CONDUCT.md
- [ ] Add issue templates

### Phase 2: Content enhancement (2-4 weeks)

- [ ] Add notebook headers to all examples
- [ ] Organize examples by difficulty
- [ ] Write installation guide
- [ ] Create 3-5 core tutorials
- [ ] Enhance API documentation with examples
- [ ] Add docstrings to all public functions

### Phase 3: Advanced features (4-6 weeks)

- [ ] Set up automated notebook testing?
- [ ] Add Binder integration?
- [ ] Create example gallery
- [ ] Write developer guide
- [ ] Add benchmarks page
- [ ] Set up CI/CD for docs (partly done)

## Assignments

| Team Member | Responsibility |
|-------------|----------------|
| TBD | API documentation, developer guide |
| TBD | Tutorial content, conceptual explanations |
| TBD | Data format docs, pipeline examples |
| TBD | Model examples, training guides |
| All | Docstrings for own code |

## Quality metrics

```bash
# Docstring coverage
interrogate crest/ --verbose

# Sphinx build without warnings
cd docs/ && make clean && make html

# Notebook execution
pytest examples/ -v
```

**Target metrics**:

- Docstring coverage: >85%
- All notebooks executable: 100%
- Sphinx warnings: 0
- Broken links: 0

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/econchick/interrogate
    rev: 1.5.0
    hooks:
      - id: interrogate
        args: [--fail-under=80]
```

