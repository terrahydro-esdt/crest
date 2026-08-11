# Changelog

All notable changes to CREST will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `GriddedModel` no longer hardcodes a filesystem path as the default `SysMetrics` 
  run directory; falls back to `SysMetrics`'s own default (`logs/<timestamp>`).
- `pyproject.toml`'s `docs` extra was missing packages `docs/conf.py` actually
  requires (`myst-nb`, `nbsphinx`, `nbsphinx-link`, `pydata-sphinx-theme`,
  `sphinxcontrib-mermaid`), so `pip install ".[docs]"` alone could not build the
  documentation.
