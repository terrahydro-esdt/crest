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
   > Note: `main` and `develop` are both protected on GitLab — `develop` has to be,
     for the push mirror to update it on GitHub (see "One-time setup" below).
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

Merge this MR with a **merge commit — never squash or rebase it**. Squashing creates a
new commit that `develop` doesn't have, and the two branches (and GitHub, which mirrors
both) drift apart from then on.

## Getting to GitHub: a push mirror, not a PR

There is no GitHub-side PR for promotion, and no manual push. GitLab's
`astg/terrahydro/development/crest` project has a **push mirror** to
`github.com/terrahydro-esdt/crest` that propagates `main`, `develop` and the release
tags to GitHub whenever they change on GitLab. This is the opposite direction from
`prod/`'s mirror described earlier (`prod/` *pulls* from GitHub; this one *pushes* to
GitHub).

A tag on its own doesn't get a release archived: Zenodo listens for a GitHub *Release*
being published, and a mirror only copies git refs. The `Release` GitHub Actions
workflow (`.github/workflows/release.yml`) closes that gap — when a `vX.Y.Z` tag arrives
it checks the tag against the version files, runs `test` and `doctest` on the tagged
commit, builds the sdist and wheel, and publishes the GitHub Release with the
version's `CHANGELOG.md` section as its text. Zenodo then archives it and mints the
version's DOI.

### One-time setup

Done once per repo, in this order (the GitLab mirror generates the key that GitHub
then needs).

**On GitLab** (needs Maintainer on `astg/terrahydro/development/crest`):

1. Settings → Repository → **Protected branches**: protect `develop` as well as `main`
   (allowed to push: no one; allowed to merge: Developers + Maintainers). Only
   protected refs are mirrored — see step 3.
2. Settings → Repository → **Protected tags**: protect `v*` (allowed to create:
   Maintainers). Release tags then can't be moved or deleted, and they count as
   protected refs for the mirror.
3. Settings → Repository → **Mirroring repositories** → Add new:
   - Git repository URL: `ssh://git@github.com/terrahydro-esdt/crest.git`
   - Mirror direction: **Push**
   - Detect host keys (should show GitHub's keys)
   - Authentication method: **SSH public key**
   - **Mirror only protected branches: on.** GitLab has many internal working
     branches that must never be pushed to the public GitHub repo.
   - Keep divergent refs: on (a branch that has diverged on GitHub is reported as a
     mirror error instead of being overwritten).
4. After saving, use the mirror's **Copy SSH public key** button.

**On GitHub** (needs admin on `terrahydro-esdt/crest`):

5. Settings → **Deploy keys** → Add deploy key: title `gitlab-push-mirror`, paste the
   key from step 4, tick **Allow write access**.
6. Settings → Rules → **Rulesets** → New branch ruleset, named `main and develop`:
   - Enforcement: Active
   - Bypass list: **Deploy keys** (always allow)
   - Target branches: `main`, `develop`
   - Rules: restrict deletions; block force pushes; require a pull request (1
     approval, dismiss stale approvals, require conversation resolution); require
     status checks `test` and `doctest`, up to date with the base branch.
7. Settings → **Branches**: delete the classic protection rules for `main` and
   `develop`. Classic rules and rulesets both apply and the classic ones have no
   bypass list, so leaving them would keep blocking the mirror.
8. Settings → **Actions** → General → Workflow permissions: leave the default
   (read). `release.yml` asks for `contents: write` only in the job that publishes.

**Check it:** in GitLab's mirror list, **Update now** should succeed, and
`gh api repos/terrahydro-esdt/crest/rulesets` should list the ruleset.

GitHub then behaves as before for people: nobody, admins included, can push to `main`
or `develop`; only the mirror's deploy key can.

### If a mirror update fails

The mirror only fast-forwards. If someone merged a PR on GitHub directly, GitHub's
branch has commits GitLab doesn't, and the mirror reports that branch as divergent.
Fix it on GitLab by merging the GitHub branch into a GitLab branch without changing
any files, then let it go through review as usual:

```bash
git fetch github
git switch -c chore/merge-github-history origin/develop
git merge -s ours github/main github/develop    # records GitHub's commits, changes nothing
```

(This is how `v0.2.0` reconciled the PRs merged on GitHub after `v0.1.0`.)

## Branch protection on GitHub

The `main and develop` ruleset is the backstop against direct pushes:
- No direct pushes for anyone, including admins — only the mirror's deploy key
  bypasses it.
- 1 required approving review (relevant only if a PR is ever opened on GitHub; not
  part of the mirror path).
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

The version lives in `pyproject.toml`, `crest/__about__.py`, `CITATION.cff`, the BibTeX
block in `README.rst` and `CHANGELOG.md`. Don't edit those by hand —
`scripts/release.py` keeps them in step, and CI refuses a release where they disagree.

Throughout development, every MR adds its user-visible changes under `## [Unreleased]`
in `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/) sections:
Added / Changed / Fixed / Removed). That section becomes the release notes.

1. **Release branch, on GitLab.** Off the latest `develop`:
   ```bash
   git fetch origin
   git switch -c release/vX.Y.Z origin/develop
   scripts/release.py bump X.Y.Z          # versions, date, CHANGELOG heading and links
   git diff                               # review
   scripts/release.py check               # must print "release check: OK"
   ```
   Commit, push, and open an MR into `develop`. The `release-check` CI job runs on
   `release/*` branches. Pick X.Y.Z by [semantic versioning](https://semver.org/);
   before 1.0, new features or behavior changes bump Y, fixes only bump Z.
2. **Freeze `develop`** from the moment the release MR is merged until the tag exists —
   anything merged in between goes into this release.
3. **`develop` → `main` MR**, merged with a merge commit (see above). This is the
   release decision.
4. **Tag `main`** on GitLab: Code → Tags → New tag, name `vX.Y.Z`, create from `main`,
   message `CREST vX.Y.Z`. Or locally:
   ```bash
   git fetch origin
   git tag -a vX.Y.Z origin/main -m "CREST vX.Y.Z"
   git push origin vX.Y.Z
   ```
   Push one tag at a time: GitHub doesn't run workflows for a push of more than three
   tags at once.
5. **Everything after that is automatic:** the mirror pushes `main` and the tag to
   GitHub, the `Release` workflow publishes the GitHub Release (~20 min, most of it
   tests), and Zenodo archives it.
6. **Check:** the release appears at
   `https://github.com/terrahydro-esdt/crest/releases`, and a few minutes later a new
   version shows up under the concept DOI's Zenodo record.

If the `Release` workflow fails (e.g. a flaky test), open the run on GitHub and use
**Re-run failed jobs**; it can't publish twice, since `gh release create` refuses an
existing release. If the tag itself is wrong (the `check` job says so), fix the files
through a new MR, delete the tag on GitLab and on GitHub, and tag again.

`CITATION.cff` and the README cite the Zenodo **concept DOI**
(`10.5281/zenodo.21939176`), which always resolves to the latest version, so a release
doesn't need a DOI update afterward. Each version also gets its own DOI on Zenodo for
anyone who needs to cite one specific version.

The GitHub↔Zenodo integration is already switched on for this repo. For a new repo,
switch it on first (Zenodo → account settings → GitHub → toggle the repo); Zenodo
only archives releases published after that.

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
