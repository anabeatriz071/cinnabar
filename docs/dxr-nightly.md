# DXR Windows nightly

Experimental sunlight shadows for nearby opaque cube blocks. This is not a port of
Bedrock RTX's full lighting or material system. Non-cube and cutout casters,
reflections, emissive lighting and RTX material maps remain incomplete.

Download the Windows x64 setup executable or extract the portable ZIP and open
`cinnabar.exe`. The first launch fetches the game's resource pack after EULA acceptance.
This unsigned test build comes from the source commit recorded in `build.txt`.
Keep automatic updates Off in Settings → Updates while testing this fork; the
upstream updater does not publish these experimental builds.

Open **Settings → Video → DXR Sunlight Shadows (Experimental)**. It defaults to Off
and applies immediately; no command prompt or environment variable is required.
The option requires Windows, DirectX 12, the bundled DXC and a GPU with inline ray
queries (DXR Tier 1.1). Keep Java shader packs Off. Turning shadows Off releases
their cached terrain structures and restores ordinary rendering.

Terrain uploads run before optional shadow builds. During continued uploads or
deferred terrain work, one shadow section may build every eight eligible frames;
otherwise one may build per frame. Shadow builds wait while the terrain arena grows.
Edited or unloaded casters are masked individually while the previous scene stays
visible during replacement builds. Equivalent opaque geometry reuses its acceleration
structure, and a retention margin stabilizes caster selection during movement.
Only one shadow build submission may remain in flight.

Sunlight uses a fixed world-space finite-sun pattern: two rays per shaded sample,
with up to four where the opposing probes disagree. This softens shadow edges
without animated noise, history trails or screen-space blur across surfaces.
Shadow darkening follows the receiver's sun-facing angle and retains at least 60%
of the existing scene lighting. This is approximate direct-light compositing;
indirect lighting, denoising and thin-caster sampling improvements remain open.
These limits do not establish a hardware performance pass.

To test, join a local world during daylight, compare shadows Off and On, and break
an elevated opaque cube that casts a clear shadow. Check that the block changes,
its shadow clears, and newly visited chunks load. Compare a raised block's shadow
edge with its contact shadow and check that surfaces facing away from the sun retain their
existing light. Report FPS with shadows Off and On, including the anti-aliasing
setting; soft-shadow sampling increases ray work. Also try placing the cube again
and turning shadows Off while travelling. Record the source commit, resolution,
anti-aliasing, FPS and observed result with any report. The earlier prototype's
slowdown and stalled terrain updates have not yet been reproduced on this build.

The headless DXR probe checks occlusion and composite pixels after removal. A
hosted runner without a DXR device reports a missing fixture; that is not a GPU
pass. Fresh in-game removal, streaming, visual quality and RTX 5060 performance
acceptance remain pending.

The workflow validates the selected Windows regression tests alongside the release
build, then publishes only when both jobs succeed. Test and release build caches
are separate and retain workspace outputs as well as dependencies. Content hashes
restore timestamps only for unchanged source inputs, so Cargo can reuse unchanged
crates while rebuilding edits. Complete snapshots use per-commit keys and restore
the latest compatible snapshot; the first snapshot bootstraps from the existing
dependency cache. Go caching belongs to the packaging job, which builds the core. Hosted-runner capacity, cache eviction
and optimized linking still affect build time.
