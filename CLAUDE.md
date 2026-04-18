# Project instructions — `mlx-swift-sam3`

**Before doing anything, read these in order:**

1. `AGENT.md` — mission, ground rules, per-module workflow, MLX-Swift gotchas, do-not-do list.
2. `TODO.md` — 12-phase roadmap, per-module checklist, parity results log.

Then check `TaskList` for in-progress work, and pick up at the next unchecked item in `TODO.md`.

Global rules in `~/.claude/CLAUDE.md` still apply (plan mode, parity-first, minimal diffs, etc.).

## Checkpoint rule (every 3 phases)

After completing **Phase 3, 6, 9, and 12**, stop and post a checkpoint report. Do NOT start the next phase on your own.

Checkpoint report format (≤10 bullets total):

- **Phases completed this block**: which phases, one line each.
- **Parity summary**: worst `max |Δ|` across all modules this block, dtype used, any fp16 fallbacks.
- **Deviations from `python/sam3`**: any intentional slims or known mismatches (should also be in `TODO.md`).
- **Weight / fixture status**: any new HF params that needed `sanitize` remapping; any fixtures that couldn't be generated.
- **Blockers / open questions**: things that need user input before next block.
- **Next block preview**: one line on what phases 4–6 / 7–9 / 10–12 will tackle.

Then wait. Don't touch phase N+1 until the user says go.

Rationale: keeps the user in the loop at natural integration seams (Phase 3 = all encoders done; Phase 6 = image e2e working; Phase 9 = video math done; Phase 12 = agent layer).
