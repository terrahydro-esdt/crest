# Architecture Decision Records

An Architecture Decision Record (ADR) is a short document that records one design
decision: the problem, the options considered, what we chose and what follows from it.
ADRs keep the *why* behind the code, which merge request comments and people's
memories don't hold for long.

## When to write one

Write an ADR when a change:

- adds or replaces a core abstraction (for example, Workflow replacing Engine),
- changes how other repos use CREST (terrahydro, terrahweb, cloud-framework), or
- is expensive to reverse.

Bug fixes and routine features don't need one.

## How

1. Copy `template.md` to `NNNN-short-title.md`, using the next number.
2. Set **Status: Proposed** and open a merge request. Discuss it in the MR,
   ideally next to a working example.
3. When the MR merges, set the status to **Accepted**. If a later ADR replaces it,
   set **Superseded by NNNN**. Don't delete old ADRs: the history is the point.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-workflow-runner-layer.md) | Workflow / WorkflowRunner layer | Proposed |
