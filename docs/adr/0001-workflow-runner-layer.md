# 0001. Workflow / WorkflowRunner layer

- **Status:** Proposed
- **Date:** 2026-11-10
- **Deciders:** CREST team (author: Deepthi Raghunandan)
- **Issue:** #77 (work split into #78–#83)

## Context

CREST runs work through `Engine` / `ModelEngine`. Each live task (forecast, hindcast,
MERRA-2 ingest) subclasses `ModelEngine`, and the code combines three concerns:

1. **What** the work is: load data, run a model, write outputs.
2. **Where and how** it runs: in-process, as a Slurm job, or on a remote host.
3. **What happens around it**: retries, notifying other systems, starting the next step.

Because of this, the same task can't be run locally, under Slurm or in a container
without changing code. Retry and chaining logic is re-implemented per task. Other
systems, such as monitoring, dashboards or an AI assistant, have no standard way to
observe a run.

We also want containerized, reproducible workflows, configuration-driven training and
benchmarking, and an assistant layer that can start and watch work. All three need one
contract for "a unit of work".

## Decision

Replace `Engine` / `ModelEngine` with four pieces that each have one job:

| Piece | Responsibility |
|---|---|
| `Workflow` (`crest/engine/Workflow.py`) | **What**: an abstract `setup()` → `run()` → `teardown()` lifecycle. `setup()` must be idempotent. Keyword arguments are routed to each method by signature. `Workflow.main()` / `as_command()` build a click CLI, so every workflow runs standalone and can be a container entrypoint. |
| `WorkflowRunner` (`crest/engine/WorkflowRunner.py`) | **Where and how**: executes a workflow, owns success, failure and cleanup, and returns an immutable `WorkflowResult`. Runner-scoped resources go in `open()` / `close()`. Concrete runners: `LocalRunner`, `SlurmRunner` (#79). |
| Events (`crest/engine/Events.py`) | **What happened**: an `EventBus` publishes `WorkflowEvent`s (STARTED, SUBMITTED, FINISHED, FAILED, INGESTED) to `Subscriber`s. FINISHED is published only after `teardown()`. |
| `DeploymentConfig` (`crest/configuration/ConfigManagement.py`) | **Configuration**: one validated (pydantic) YAML file declares the workflows, runner kind (`local` / `slurm` / `remote`), retry policy, path templates and trigger chains. |

Retries (#80) and chaining (#81) are runner and subscriber concerns, not workflow code.
A retry re-runs the whole lifecycle, which is why `setup()` must be idempotent.

## Options considered

- **Keep `ModelEngine` and add hooks.** Least change, but it keeps "what" and "where"
  together, so a task still can't move between local, Slurm and container without
  code changes.
- **Adopt an external orchestrator now** (e.g. Airflow, Prefect). Powerful, but it adds
  a service to operate on the live servers and a second way to define work. The
  lifecycle and event contract here is small enough to adapt to one later: a runner or
  subscriber can hand off to an orchestrator.
- **Review the abstract classes first, then build** (the original !75). Replaced by
  reviewing them together with a working example: the live hindcast running as a
  `Workflow` (#82).

## Consequences

- A workflow is written once and runs locally, under Slurm or in a container, chosen
  by configuration.
- Retry, chaining and notifications are written once, in runners and subscribers.
- Monitoring, dashboards and an assistant layer can subscribe to events and read
  `WorkflowResult`s without knowing scheduler details.
- Every live task has to be ported from `ModelEngine` (#82, #83). Until the port is
  finished, both patterns coexist on `working_live`.
- Consumers of CREST must move to a CREST version that includes this layer. See the
  "Version compatibility" page in the TERRAHydro wiki.

## Open questions

- Triggers with several sources or prerequisites per step, and triggers whose source
  runs under a different configuration (#81).
- Whether the `remote` runner kind is needed now or later (#79).
