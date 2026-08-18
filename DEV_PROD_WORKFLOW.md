# Working with dev/ and prod/

This describes how the team should work across our two repos for CREST, and is meant
as the template to reuse for future repos (`terrahydro`, `fews-net`, etc.) once they're
ready for the same treatment.

**GitLab is the source of truth.** All real development and review — feature branches,
merge requests into `develop`, and eventually `develop` into `main` — happens on
GitLab. GitHub is not a second place where day-to-day review happens; it's the
public-facing destination that receives `main` periodically, at release time, paired
with a new tag. Nobody interacts with GitHub directly outside of that periodic
promotion.

## The pieces: dev/, GitHub, and prod/

| | dev/ | GitHub | prod/ |
|---|---|---|---|
| Host | GitLab (`astg/terrahydro/development/crest`) | github.com (`terrahydro-esdt/crest`) | GitLab (`astg/terrahydro/production/crest`) |
| Visibility | Private | Public | Internal |
| Role | **Source of truth** — all review happens here, via GitLab merge requests, for both `develop` and `main` | Public-facing home for released code — where the latest release lives | Internal-network reflection of GitHub |
| Updated by | Team members, via GitLab merge requests | A single periodic promotion from GitLab's `main`, paired with a new release tag | Automatically, by GitLab's own pull-mirror job |

Code flows one direction only: `dev/` *(GitLab MRs into `develop`, then `develop` into
`main`)* → *(one periodic promotion + new tag)* → GitHub → *(automatic pull-mirror)* →
`prod/`. None of these stay continuously in sync — GitHub only updates when someone
does the deliberate release-time promotion; `prod/` only catches up with GitHub
whenever GitLab's mirror job next runs.

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

Note `main` doesn't move for every change, only when it's time to prepare a release, via a GitLab merge request from `develop` into `main`. This is the last step before the GitHub promotion below.

## Promoting to GitHub (periodic, at release time)

Once GitLab's own `main` has what you want to release, promote it to GitHub. Today
this is a manual step; the plan is to automate the push + PR-creation part once that
automation exists and is verified — but the review/merge gate on the GitHub side
doesn't go away even once that happens; only the "someone runs the push command" part
gets automated.

```
git push github <gitlab-main-local-branch>:promote-vX.Y.Z
gh pr create --repo terrahydro-esdt/crest --base main --head promote-vX.Y.Z \
  --title "Release vX.Y.Z" --reviewer <someone-other-than-you>
```

**Merge this with a real merge commit, not squashed.** This lesson came from direct
experience: squash-merging a promotion PR once collapsed the merge down to a
single-parent commit, silently discarding the ancestry link, which left the *next*
promotion PR permanently stuck in GitHub's `behind` state — content matched, but
GitHub's `main` had a commit that technically wasn't in the promoted branch's history,
and there was no way to fix that after the fact without another PR into the protected
branch.
```
gh pr merge <number> --repo terrahydro-esdt/crest --merge
```

### Optional organization of your local workspace

Regular day-to-day contributors never need this — it's only relevant if you directly perform the GitLab → GitHub promotion. To do so, add a second remote to your `dev/` clone:
```
❯ git remote -v
github	git@github.com:terrahydro-esdt/crest.git (fetch)
github	git@github.com:terrahydro-esdt/crest.git (push)
origin	git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest.git (fetch)
origin	git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest.git (push)
```
`origin` stays your normal GitLab remote. `github` is added on top, used only for the
promotion push above. Everyone else can ignore this entirely.

## Branch protection on GitHub

Even though real review happens on GitLab, we have GitHub's protected-branch rules in place as a technical safety net to prevent anyone from
pushing straight to `main`/`develop` on GitHub, bypassing the promotion PR:
- No direct pushes — for anyone, including admins (`enforce_admins: true`).
- 1 required approving review.
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

After the GitHub promotion PR (above) is merged, `main` on GitHub actually has the
work you're releasing. Tag from `main` — tags/releases don't need to go through branch
protection, since a release just points at an already-existing commit, it doesn't push
new commits to a protected branch, e.g.:

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

- **Refspec ambiguity when pushing** (relevant to whoever's doing the GitHub
  promotion). `git push github <branch>` (no colon) can resolve to the wrong
  destination if that local branch has upstream-tracking configured against a
  *different* branch name. Always use the explicit form to be unambiguous:
  ```
  git push github <local-branch>:<remote-branch>
  ```
- **Force-pushing a protected branch is rejected server-side**, no matter who's
  pushing or what client-side flags are used (`--force-with-lease` included). If you
  actually need to overwrite a protected branch's history, that requires an admin
  temporarily adjusting the branch protection settings first.
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
