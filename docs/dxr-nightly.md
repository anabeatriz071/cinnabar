# DXR Windows nightly

Experimental opaque-terrain sunlight shadows, PBR materials, metal reflections,
and placed/held torch and lantern lighting. Full Bedrock RTX path tracing,
indirect light bounces, non-cube/cutout ray geometry, and actor/water receivers
remain incomplete.

Download the Windows x64 setup executable or extract the portable ZIP and open
`cinnabar.exe`. The source commit is recorded in `build.txt`. First launch fetches
the resource pack after EULA acceptance. Keep automatic updates Off under Updates
while testing this fork; the upstream updater does not publish these builds.

Enable **Settings → Video → DXR Lighting (Experimental)**. It defaults to Off and
requires Windows, DirectX 12, the bundled DXC and DXR Tier 1.1 inline ray queries.
Keep Java shader packs Off. The older saved `dxr_terrain_shadows` preference remains
compatible. The legacy Enhanced renderer remains disabled.

## PBR packs

Sunlight and local torch/lantern lighting need no RTX pack. Normal detail, mapped
emission and reflective metals use authored PBR materials. Metal reflections follow
the pack's metalness and roughness maps.

Import the RTX `.mcpack` through **Settings → Global Resources**, activate it,
and apply the selection. With DXR Lighting enabled, its texture sets load automatically,
including normal, metalness, emission and roughness maps. Changing or removing the
active pack reloads those maps; no JSON file or restart is needed.

The optional `dxr-pbr.json` configuration remains available for loose development
packs when no global resource pack is active. Its `format: "bedrock"` spelling is
accepted. The default in-game texture cap is 128 pixels per tile.

`max_tile_size` defaults to 128 and accepts powers of two from 16 through 512.
Bigger maps cost memory; this cap minifies maps without changing authoritative
block data. Texture decoding runs off the frame thread and GPU publication uses
the existing bounded, complete-map upload path.

## Lighting and reflection scope

Authored albedo modulates the existing scene colour, preserving the native lightmap,
biome tint, fog and sunlight shadows. Normal maps use a linear mipmapped sampler;
roughness and metalness feed GGX/Smith/Schlick direct-light terms. Metals trace one
reflection bounce against the cached opaque cube scene, including off-screen
geometry within that cache. Misses use an approximation of the current sky colour.
Roughness controls reflection weight and texture mip detail; geometry reflections
are not yet filtered into smooth rough-surface lobes.

Placed torches, lanterns, soul torches and soul lanterns publish local lights from
actual block states. Both of the local player's hands can emit, including when a
server omits the held block-runtime field. Held emitters use approximate hand
anchors, not animated model bones. Other players' held lights are not implemented.
Warm and soul colours are provisional. Radius comes from the source state's native
emission level. These optional coloured direct lights supplement the existing
vanilla block-light solution; they do not change packets or server light state.

Emitter discovery scans at most two changed resident sections per frame and
prioritizes nearby sections. Up to 32 lights are retained, with both hands first;
at most four eligible lights are evaluated per shaded sample. Shadows use opaque
terrain occlusion, so fences, leaves, lantern meshes and actors are not blockers in
this prototype. Terrain within the 32-block receiver region gets these effects;
models, grass, water and actors keep their existing shading.

Sunlight uses a stable finite-sun pattern: two rays per shaded sample, up to four at
mixed edges. Near-tangent faces fade their extra shadow darkening; other sun-facing
faces retain readable contrast. Existing casters survive movement and equivalent
mesh publications. Edited/unloaded instances are masked individually during rebuilds.
Continued terrain uploads admit one shadow build every eight eligible frames;
arena growth suspends builds and only one build submission remains in flight.
Ray-hit material slots reserve approximately 24 MiB and upload only changed sections.

## Testing

Compare DXR Off/On in daylight and at night. Break an elevated cube and verify its
shadow clears. Move around and visit new chunks. With a suitable PBR pack, compare
normal detail and rough versus polished metal; verify a nearby block appears in a
metal reflection even when it is outside the screen.

Place and break a torch and lantern in a dark area. Hold each in the main hand and
offhand, walk, switch to an unlit item, and leave/rejoin the world. Verify there are
no leftover lights. Also check soul variants. Record the source commit, resolution,
anti-aliasing, FPS and GPU timings. More ray work, especially with MSAA, can reduce
FPS; this batch has no RTX 5060 visual/performance acceptance yet.

Hosted runners without DXR report named missing fixtures for hardware probes.
Those results are not GPU passes. CPU checks cover map decoding, emitter removal,
hand-light selection, shader bindings and cache invalidation. The nightly builds
validation and packaging in parallel and publishes only after both succeed.
Full build snapshots retain workspace outputs; content-checked timestamps and
Cargo fingerprints preserve unchanged crates while rebuilding changed inputs.

## RT pipeline migration status

The client pins `anabeatriz071/wgpu:cinnabar-v30` at
`0a632ff196d098525327489831447e88e84ff0a4`. This applies two borrowed DX12 HAL
accessors and an empty-TLAS DX12 workaround to the exact former pin `a13a9d6dad54aeebc4d1445425ec1c3adbba2fd5`;
existing public signatures and non-empty rendering behavior are unchanged. All wgpu-family patches
share that revision and pass the existing release-branch pin guard. Development
and validation use the owner's forks; an upstream PR is not a prerequisite.

The independent `dxr-pipeline` backend implements native state objects, triangle
hit groups, optional any-hit shaders, shader identifier tables, checked dispatch
limits and `DispatchRays`. Windows cross-compilation, CPU layout tests and the
native marker-dispatch/readback probe pass. The wgpu interoperability probe builds
an acceleration structure through the public wgpu API, traces a hit and miss through
native HLSL `TraceRay`, then removes the instance and checks that both rays miss.
This new bridge probe requires Windows DXR validation; missing fixtures are named
skips and do not establish acceptance. If GitHub rejects dispatch because Actions
is disabled, enable Actions in the fork's repository settings before running the
backend check. This does not require starting an installer build.

Empty DX12 scene builds use an inactive zeroed instance, preserving all-miss
semantics while avoiding WARP's zero-instance divide-by-zero crash. The dependency
lock preserves the existing Windows interface versions.

Native recording uses a dedicated raw encoder. Resource transitions and result
copies use separate tracked wgpu encoders, submitted in order on the same queue.
The pipeline and all referenced buffers/acceleration structures remain alive through
submission completion. Native handles come from borrowed HAL accessors, without
private struct reinterpretation or unsynchronized external queue submissions.

The in-game renderer still uses inline ray queries. World-pass integration, PBR
migration, shadowed volumetric light shafts, stochastic area-light shadows and
accumulation/denoising remain incomplete. GI, water transport and DLSS are separate
work. The native marker probe alone does not establish terrain rendering or RTX
5060 visual/performance acceptance.

The backend-only Windows workflow compiles DXIL fixtures and tests this small crate;
it does not build an installer or publish a release. Installer builds remain manual
while migration is pending. They create new `dxr-preview-<run>-<attempt>-<commit>`
prereleases and never move or overwrite `nightly-dxr`.
