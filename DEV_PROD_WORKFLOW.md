# Working with dev/ and prod/

This describes how the team should work across our two repos for CREST, and is meant
as the template to reuse for future repos (`terrahydro`, `fews-net`, etc.) once they're
ready for the same treatment.

## The pieces: dev/, GitHub, and prod/

There are three things involved, not two — `prod/` is not GitHub itself, it's a
separate internal mirror *of* GitHub:

| | dev/ | GitHub | prod/ |
|---|---|---|---|
| Host | GitLab (`astg/terrahydro/development/crest`) | github.com (`terrahydro-esdt/crest`) | GitLab (`astg/terrahydro/production/crest`) |
| Visibility | Private | Public | Internal |
| Purpose | Active development, collaboration | The actual source of truth — released, citable, public-facing code | Internal-network reflection of GitHub |
| Updated by | You, directly — commits, feature branches | PRs from `dev/`, reviewed and merged | Automatically, by GitLab's own pull-mirror job |

Code flows one direction only: `dev/` → *(deliberate PR)* → GitHub → *(automatic
pull-mirror)* → `prod/`. None of the three stay continuously in sync with each other —
`dev/` and GitHub only sync when someone opens and merges a PR; `prod/` only catches up
with GitHub whenever GitLab's mirror job next runs (its own polling interval, not
instant).

`prod/`'s pull-mirror is configured **server-side** — nothing about it lives
in your local git config, and it isn't something you set up per-clone. If you have a
local clone of `prod/`, treat it as strictly read-only!

`dev/` is where day-to-day work happens; code only reaches GitHub — and from there,
eventually `prod/` — when someone deliberately promotes it via a PR.

## Day-to-day workflow

1. Work as normal in `dev/` (GitLab) — feature branches, local iteration, collaborating
   with teammates there.
2. When something is ready to become public: push the branch to GitHub and open a PR
   **against `develop`**. `main` is not touched for day-to-day work — see "Release
   branch model" below. Also the following assumes that you are using `gh` (recommended).
   ```
   git push github <branch>:<branch>
   gh pr create --repo terrahydro-esdt/crest --base develop --head <branch> \
     --reviewer <someone-other-than-you>
   ```
3. The PR needs, before it can merge:
   - **CI passing** — the `test` and `doctest` checks (from `.github/workflows/tests.yml`)
     run automatically on every PR.
   - **1 approval from someone other than the PR author.** GitHub does not allow
     self-approval, and this applies to everyone, including admins — there is no bypass.
4. **Explicitly request a reviewer when you open the PR** (`--reviewer <name>`, or
   `gh pr edit <number> --add-reviewer <name>` after the fact, where `<name>` is 
   the GitHub username). Also note that being a repo admin does *not* mean you get
   notified about new PRs automatically — only an explicit review request reliably
   triggers a notification. Don't rely on people noticing on their own.
5. **Squash-merge feature PRs into `develop`.** Each PR becomes one clean commit —
   this is still the right default for day-to-day work. The one exception is the
   `develop` → `main` release PR — see below.
6. **Clean up your local branch afterward.** The repo has `delete_branch_on_merge`
   enabled, so GitHub already deletes the *remote* branch on merge — but your local
   copy and local remote-tracking ref stick around until you clean them up:
   ```
   git checkout develop
   git merge github/develop --ff-only
   git branch -d <branch>
   ```
   Since feature PRs are squash-merged, `-d` (safe delete) will often refuse with
   "not fully merged" — squashing deliberately breaks the ancestry link `-d` checks
   for, even though the content is safely in `develop`. That's expected, not a
   warning sign: try `-d` first as a habit, then fall back to `-D` (force) once
   you've confirmed via GitHub that the PR actually merged. Also periodically prune
   stale remote-tracking refs:
   ```
   git fetch github --prune
   ```
   A bulk (magical) cleanup one-liner, once you're comfortable with the above — deletes every
   local branch already merged into `develop`, excluding `develop`/`main` themselves:
   ```
   git branch --merged develop | grep -v -E '^\*|develop|main' | xargs -r git branch -D
   ```

### Organizing your local workspace

One local clone is enough for `dev/` work, with **two remotes** configured in it — one
for each system you actually push to:

```
❯ git remote -v
github	git@github.com:terrahydro-esdt/crest.git (fetch)
github	git@github.com:terrahydro-esdt/crest.git (push)
origin	git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest.git (fetch)
origin	git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest.git (push)
```

`origin` is GitLab-development — your default working remote for day-to-day commits.
`github` is added on top so that, from that same clone, you can push a finished branch
straight to GitHub and open a PR (`git push github <branch>:<branch>`) without needing
a second directory or any bridging step between two separate clones.

If you also want a local clone of `prod/` (GitLab-production), that's a separate,
second directory — just a plain `git clone` of the GitLab-production project, with only
its own `origin` remote. It doesn't need a `github` remote added, since you never push
to it directly — GitLab's pull-mirror job keeps it current on its own, as described
above.

## Branch protection (`main` and `develop`, both on GitHub)

Both branches currently enforce:
- No direct pushes — for anyone, including admins (`enforce_admins: true`).
- 1 required approving review.
- Required status checks: `test`, `doctest` (must pass and be up to date with the base
  branch).
- All PR conversations must be resolved before merging.
- No force-pushes, no branch deletion.


## Release branch model: `develop` is where work lands, `main` only moves at release time

`main` is **not** updated for every PR. Day-to-day work only ever targets `develop` —
that's the integration branch carrying the most recent state. `main` stays put between
releases and only advances when it's actually time to cut one, via a single PR merging
`develop` into `main` (this will not be done frequently):

```
gh pr create --repo terrahydro-esdt/crest --base main --head develop \
  --title "Release vX.Y.Z" --reviewer <someone-other-than-you>
```

That PR still goes through the normal review + CI gate (branch protection applies to
`main` same as always) — it's just infrequent instead of per-change. Once it's merged,
cut the release/tag from `main` as usual (see "Cutting a release" below).

**This PR must be merged with a real merge commit, not squashed.** Squashing collapses
the merge down to a single-parent commit and silently discards the second-parent link
that makes `develop`'s history genuinely ancestral to `main`'s. I found this out ealier:
squash-merging a `develop`→`main` PR left `main`'s next release PR permanently stuck in
GitHub's `behind` state, because `main` had a commit that wasn't actually in `develop`'s
ancestry (even though the *content* matched) — and there's no way to fix that after the
fact without another PR into the protected branch. Therefore, use:
```
gh pr merge <number> --repo terrahydro-esdt/crest --merge
```
The repo has both squash and merge-commit strategies enabled for exactly this reason —
squash for feature PRs into `develop`, real merge for the `develop`→`main` release PR.
Don't let anyone squash-merge a release PR "for consistency" with everything else.

This avoids needing a companion sync PR for every single change. If `main` and `develop`
ever need a sanity check on where they stand relative to each other do this:
```
git fetch github
git log github/develop..github/main --oneline   # commits main has that develop doesn't
git log github/main..github/develop --oneline   # commits develop has that main doesn't
```

## Cutting a release

First, merge the `develop` → `main` release PR described above so `main` actually has
the work you're releasing. Then tag from `main`. Tags/releases themselves don't need to
go through branch protection — a release just points at an already-existing commit, it
doesn't push new commits to a protected branch.

```
gh release create vX.Y.Z --repo terrahydro-esdt/crest --target main --title "vX.Y.Z" --notes "..."
```

If the GitHub↔Zenodo integration is toggled on for the repo (Zenodo account settings →
GitHub → toggle the repo on — this only needs doing once, and must happen *before* the
release you want archived), Zenodo automatically archives the release and mints a DOI.

Afterward:
- Verify the DOI actually resolves (`https://zenodo.org/records/<id>`) before using it
  anywhere — a reserved-but-unpublished draft won't resolve.
- Update `CITATION.cff`'s `doi:` field via the normal PR process. Note: the archived
  snapshot on Zenodo is immutable — fixing `CITATION.cff` later doesn't retroactively
  update a DOI/version that's already been published.

## Gotchas

- **Refspec ambiguity when pushing.** `git push github <branch>` (no colon) can resolve
  to the wrong destination if that local branch has upstream-tracking configured against
  a *different* branch name (e.g. it was created with `git checkout -b foo github/develop`).
  Always use the explicit form to be unambiguous:
  ```
  git push github <local-branch>:<remote-branch>
  ```
- **Force-pushing a protected branch is rejected server-side**, no matter who's pushing
  or what client-side flags are used (`--force-with-lease` included). If you actually
  need to overwrite a protected branch's history, that requires an admin temporarily
  adjusting the branch protection settings first.
- **GitHub outages happen.** If pages/PRs/API calls start behaving strangely, check
  `https://www.githubstatus.com/` before assuming the repo or org is broken. A degraded
  "Actions" component also means required CI checks may not complete, which will block
  merging even on an approved PR — just wait it out!

## Applying this to future repos (`terrahydro`, `fews-net`, ...)

The plan is to replicate this same dev(GitLab)/GitHub split, branch protection, and
release process per-repo as each becomes ready for public release — each gets its own
`terrahydro-esdt` GitHub repo, its own branch protection, its own versioned releases. A
`prod/` GitLab-production mirror can be layered on top the same way once it's worth
setting up for a given repo. Before any given repo goes down this path, it needs the
same treatment CREST got first: a secrets-in-history audit, and a check for any
placeholder/incorrect content.
