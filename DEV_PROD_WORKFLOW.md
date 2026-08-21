# Working with dev/ and prod/

This describes how the team should work across our two repos for CREST, and is meant
as the template to reuse for future repos (`terrahydro`, `fews-net`, etc.) once they're
ready for the same treatment.

**GitLab is the source of truth.** All real development and review — feature branches,
merge requests into `develop`, and eventually `develop` into `main` — happens on
GitLab. GitHub is not a second place where day-to-day review happens; it's the
public-facing destination that `main` gets automatically mirrored to whenever the
`develop` → `main` MR is accepted on GitLab. Nobody interacts with GitHub directly —
not even to promote a release; that's a GitLab push mirror, not a manual step (see
"Getting to GitHub" below).

## The pieces: dev/, GitHub, and prod/

| | dev/ | GitHub | prod/ |
|---|---|---|---|
| Host | GitLab (`astg/terrahydro/development/crest`) | github.com (`terrahydro-esdt/crest`) | GitLab (`astg/terrahydro/production/crest`) |
| Visibility | Private | Public | Internal |
| Role | **Source of truth** — all review happens here, via GitLab merge requests, for both `develop` and `main` | Public-facing home for released code — where the latest release lives | Internal-network reflection of GitHub |
| Updated by | Team members, via GitLab merge requests | Automatically, by GitLab's own push-mirror job, whenever `main` changes on GitLab | Automatically, by GitLab's own pull-mirror job |

Code flows one direction only: `dev/` *(GitLab MRs into `develop`, then `develop` into
`main`)* → *(automatic push-mirror, triggered by the `main` MR)* → GitHub →
*(automatic pull-mirror)* → `prod/`. GitHub and `prod/` each only update reactively —
GitHub whenever GitLab's `main` changes, `prod/` whenever GitLab's mirror job next
polls GitHub.

`prod/`'s pull-mirror is configured **server-side** by whoever administers the
GitLab-production project (Settings → Repository → Mirroring repositories, direction
"Pull", source `https://github.com/terrahydro-esdt/crest.git`) — nothing about it lives
in your local git config. If you have a local clone of `prod/`, treat it as strictly
read-only: anything committed there directly would just get silently overwritten on
the next pull sync. It is not a place to work!

## Day-to-day workflow (on GitLab)

1. Work as normal in `dev/` (GitLab) — feature branches, local iteration, collaborating
   with teammates there.
2. When something is ready, open a **GitLab merge request against `develop`**.
3. The MR should require, before it can merge:
   - **CI passing** (the `test`/`doctest` jobs from `.gitlab-ci.yml`).
   - **An approval from someone other than the author.**
   > Note: `main` is protected but a decision to protect `develop`, currently    
     unprotected, still needs to be made.
4. Merge strategy: squash feature MRs into `develop`  (one clean commit per MR). The    
   `develop` → `main` MR is the exception — see below.
5. Clean up your local branch afterward, same as always:
   ```
   git checkout develop
   git pull
   git branch -d <branch>
   git fetch --prune
   ```

### `develop` → `main`, still on GitLab

`main` doesn't move for every change — only when it's time to prepare a release, via a
GitLab merge request from `develop` into `main`. **Accepting this MR is the actual
release gate**: it's the decision that something is ready to go out, not a separate
step afterward.

## Getting to GitHub: a push mirror, not a PR

There is no GitHub-side PR for promotion, and no manual push. GitLab is to be configured
with a **push mirror** on the `astg/terrahydro/development/crest` project (Settings →
Repository → Mirroring repositories, direction "Push", target
`https://github.com/terrahydro-esdt/crest.git`) that automatically propagates `main`
to GitHub whenever it changes. This is the opposite direction from `prod/`'s mirror
described earlier (`prod/` *pulls* from GitHub; this one *pushes* to GitHub).

The version tag also originates on GitLab: right after (or as part of) accepting the
`develop` → `main` MR, tag GitLab's `main` with the release version. The push mirror
propagates both the branch update and the new tag to GitHub automatically.

**TODO:**

- **GitHub's branch protection on `main` currently blocks this push outright** —
  `enforce_admins: true` means *no one* — not even admins — can push directly to
  `main` on GitHub. Therefore, the GitHub account must be setup with an exception,
  added to a GitHub **Ruleset** bypass list, that will keep the pretected branches blocked from
  direct pushes exactly as now, but allow a `terrahydro-gitlab-mirror` exception to
  push on updates. Furthermore, we still need to decide on whether to protect the `develop `
  branch on GitLab (it is already protected on GitHb).
- **A mirrored tag alone likely won't trigger Zenodo.** Zenodo's GitHub integration
  listens for a GitHub *Release* being published. A push mirror only replicates git refs (branches, tags), it doesn't create GitHub Release objects, which are a GitHub-specific concept layered on top of a tag. So even once the tag shows up on GitHub via the mirror, someone likely still needs to turn it into an actual Release for Zenodo to pick it up:
  ```
  gh release create vX.Y.Z --repo terrahydro-esdt/crest --target main --title "vX.Y.Z" --notes "..."
  ```
  Worth verifying directly rather than assuming either way.

## Branch protection on GitHub

GitHub's protected-branch rules on `main`/`develop` still exist as the technical
backstop against ad-hoc direct pushes — but see above: as currently configured
(classic branch protection), they also block the push mirror itself. Migrating `main`
to a Ruleset with `terrahydro-gitlab-mirror` on the bypass list (once that's done) resolves this without weakening the rule for anyone else.
- No direct pushes — for anyone, including admins (`enforce_admins: true`).
- 1 required approving review (relevant if `develop` on GitHub is ever used directly;
  not part of the mirror-based `main` update path).
- Required status checks: `test`, `doctest` (must pass and be up to date with the base
  branch).
- All PR conversations must be resolved before merging.
- No force-pushes, no branch deletion.

If `main` and `develop` on GitHub ever need a sanity check on where they stand
relative to each other:
```
git fetch github
git log github/develop..github/main --oneline   # commits main has that develop doesn't
git log github/main..github/develop --oneline   # commits develop has that main doesn't
```

## Cutting a release

1. Merge the `develop` → `main` MR on GitLab (this is the actual release decision).
2. Tag GitLab's `main` with the version.
3. The push mirror propagates the branch and tag to GitHub automatically (once the
   branch-protection conflict above is resolved).
4. Create the GitHub Release from that tag to trigger Zenodo (see above) — until/unless
   this step is confirmed to also happen automatically:
   ```
   gh release create vX.Y.Z --repo terrahydro-esdt/crest --target main --title "vX.Y.Z" --notes "..."
   ```

If the GitHub↔Zenodo integration is toggled on for the repo (Zenodo account settings →
GitHub → toggle the repo on — this only needs doing once, and must happen *before* the
release you want archived), Zenodo automatically archives the release and mints a DOI.

Afterward:
- Verify the DOI actually resolves (`https://zenodo.org/records/<id>`) before using it
  anywhere — a reserved-but-unpublished draft won't resolve.
- Update `CITATION.cff`'s `doi:` field via the normal MR process, on GitLab like
  everything else. Note: the archived snapshot on Zenodo is immutable — fixing
  `CITATION.cff` later doesn't retroactively update a DOI/version that's already been
  published.

## Gotchas

- **Force-pushing a protected branch is rejected server-side**, no matter who's
  pushing or what client-side flags are used (`--force-with-lease` included, and this
  applies to the push mirror too — see above). If you actually need to overwrite a
  protected branch's history, that requires an admin temporarily adjusting the branch
  protection settings first.
- **GitHub outages happen.** If pages/PRs/API calls start behaving strangely, check
  `https://www.githubstatus.com/` before assuming the repo or org is broken. A
  degraded "Actions" component also means required CI checks may not complete, which
  will block merging even on an approved PR — just wait it out!


## Applying this to future repos (`terrahydro`, `fews-net`, ...)

The plan is to replicate this same GitLab-source-of-truth / GitHub-public-release
split, branch protection, and release process per-repo as each becomes ready for
public release — each gets its own `terrahydro-esdt` GitHub repo, its own branch
protection, its own versioned releases. A `prod/` GitLab-production mirror can be
layered on top the same way once it's worth setting up for a given repo. Before any
given repo goes down this path, it needs the same treatment CREST got first: a
secrets-in-history audit, and a check for any placeholder/incorrect content.

## Branch naming convention

Format: `<type>/<short-kebab-case-description>` — lowercase, hyphen-separated, no
spaces or underscores. The description should be specific enough that someone
browsing the branch list understands the intent without opening it — `chore/fix` or
`chore/update` alone isn't enough, say what's actually changing.

Types:
- **`feature/`** — new functionality.
- **`fix/`** — bug fixes.
- **`refactor/`** — restructuring code without changing behavior.
- **`chore/`** — maintenance that isn't a feature or bug fix: docs, config, CI/CD,
  repo settings, cleanup. Note all recent branches in CREST have been this type —
  `chore/update-doi`, `chore/readme-review`, `chore/doc-cleanup`,
  `chore/sync-doi-to-develop`, `chore/establish-main-ancestry`.
- **`docs/`** — optional, if you want something more specific than `chore/` for
  documentation-only changes. 

Whichever type, delete the branch once its MR/PR merges — see "Clean up your local
branch afterward" above.
