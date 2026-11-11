//! Bounded Windows DXR sunlight prototype for resident opaque cube terrain.
#![cfg_attr(
    not(windows),
    allow(
        dead_code,
        reason = "Windows pass compiled on other hosts for regression checks"
    )
)]

use super::*;
use bevy::{
    camera::{MainPassResolutionOverride, primitives::Frustum, visibility::RenderLayers},
    core_pipeline::{Core3d, Core3dSystems},
    render::{
        renderer::{RenderContext, ViewQuery},
        view::ViewDepthStencilTexture,
    },
};
use render_scene::{AtmosphereFrame, ChunkRenderInstance, SkyKind};
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

mod cache;
mod geometry;
#[cfg(windows)]
mod native;
mod pipeline;
mod surfaces;
use pipeline::{Frame, SUN_ANGULAR_RADIUS, create_pipeline};

use cache::{MAX_SECTIONS, invalid_instance_mask, retain_caster, select_casters};
const BUILDS_PER_FRAME: usize = 1;
const BUSY_BUILD_INTERVAL: u8 = 8;
const CASTER_RADIUS: f32 = 72.0;
const RECEIVER_RADIUS: f32 = render_scene::terrain_shadows::DXR_RECEIVER_RADIUS;
const RAY_LENGTH: f32 = CASTER_RADIUS - RECEIVER_RADIUS;

#[derive(Resource, Default)]
struct Shadows {
    sections: HashMap<Entity, Section>,
    ordered: Vec<Entity>,
    scene: Option<Arc<wgpu::Tlas>>,
    surfaces: Option<surfaces::SceneSurfaces>,
    surface_quads: Option<Buffer>,
    surface_allocation_failed: bool,
    surface_pipelines: Mutex<HashMap<(wgpu::TextureFormat, u32), surfaces::SurfacePipelines>>,
    view: Option<Entity>,
    pipelines: Mutex<HashMap<(wgpu::TextureFormat, u32), wgpu::RenderPipeline>>,
    pending: Arc<AtomicBool>,
    build_budget: ShadowBuildBudget,
    active: bool,
    budget_wait_reported: bool,
    composite_reported: AtomicBool,
}

struct Section {
    slot: u32,
    mesh: Arc<[PackedQuad]>,
    layout: CubeQuadLayout,
    origin: [i32; 3],
    blas: wgpu::Blas,
}

/// Inputs that suspend optional shadow work independently of world publication.
#[derive(SystemParam)]
struct Preparation<'w> {
    uploads: Res<'w, ChunkGpuUploadStats>,
    preference: Res<'w, render_scene::terrain_shadows::DxrTerrainShadows>,
    panorama: Option<Res<'w, render_scene::PanoramaScene>>,
}

#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum DxrPass {
    #[cfg(windows)]
    Native,
    Sunlight,
    Materials,
}

/// Registers a separate pass after opaque terrain, including late GPU-culled draws.
pub(super) fn install(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    let supported = render
        .world()
        .resource::<RenderAdapter>()
        .get_info()
        .backend
        == wgpu::Backend::Dx12
        && render
            .world()
            .resource::<RenderDevice>()
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    if render
        .world_mut()
        .try_schedule_scope(Core3d, |_, schedule| {
            schedule.add_systems(
                crate::gpu_timing::profiled(
                    composite,
                    Some(render_scene::RuntimeStage::GpuShadows),
                    "DXR terrain sunlight",
                )
                .in_set(Core3dSystems::MainPass)
                .in_set(DxrPass::Sunlight)
                .after(crate::scene_target::ScenePass::Opaque)
                .after(super::GpuCullLateLabel)
                .after(crate::entity_shadow_render::EntityShadowLabel)
                .before(crate::scene_target::ScenePass::Transparent),
            );
        })
        .is_err()
    {
        return;
    }
    #[cfg(any(windows, feature = "enhanced"))]
    render.world_mut().schedule_scope(Core3d, |_, schedule| {
        schedule.add_systems(
            crate::gpu_timing::profiled(
                surfaces::composite,
                Some(render_scene::RuntimeStage::GpuShadows),
                "DXR PBR and local lights",
            )
            .in_set(Core3dSystems::MainPass)
            .in_set(DxrPass::Materials)
            .after(DxrPass::Sunlight)
            .before(crate::scene_target::ScenePass::Transparent),
        );
    });
    #[cfg(windows)]
    {
        render.init_resource::<native::NativeTerrain>();
        render.add_systems(
            Render,
            native::reset.in_set(RenderSystems::PrepareResources),
        );
        render.add_systems(Render, native::retain.in_set(RenderSystems::Cleanup));
        render.world_mut().schedule_scope(Core3d, |_, schedule| {
            schedule.add_systems(
                native::record::composite
                    .in_set(Core3dSystems::MainPass)
                    .in_set(DxrPass::Native)
                    .after(crate::scene_target::ScenePass::Opaque)
                    .after(super::GpuCullLateLabel)
                    .after(crate::entity_shadow_render::EntityShadowLabel)
                    .before(DxrPass::Sunlight),
            );
        });
    }
    install_preparation(render);
    app.add_systems(Last, super::admit_depth_sampling);
    app.insert_resource(render_scene::terrain_shadows::DxrTerrainShadowsSupported(
        supported,
    ));
}

/// Keeps preference cleanup active even while menus suspend ordinary world passes.
fn install_preparation(render: &mut SubApp) {
    render.add_systems(RenderStartup, init).add_systems(
        Render,
        prepare
            .run_if(bevy::prelude::resource_exists::<Shadows>)
            .in_set(RenderSystems::PrepareResources)
            .after(prepare_gpu_chunks),
    );
}

/// Leaves the ordinary renderer active when the selected device lacks DXR.
fn init(mut commands: Commands, adapter: Res<RenderAdapter>, device: Res<RenderDevice>) {
    if adapter.get_info().backend == wgpu::Backend::Dx12
        && device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    {
        commands.init_resource::<Shadows>();
        bevy::log::info!(
            "experimental DXR terrain shadows available in Video settings (off by default)"
        );
    } else {
        bevy::log::debug!("DXR terrain shadows require a DX12 device with inline ray queries");
    }
}

/// Retains one acceleration structure per nearby published mesh and rebuilds the scene on edits.
fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    arena: Res<ChunkGpuArena>,
    control: Preparation,
    mut shadows: ResMut<Shadows>,
    views: Query<(
        Entity,
        &MainEntity,
        &ExtractedView,
        Option<&Frustum>,
        Option<&RenderLayers>,
    )>,
    terrain: Query<(Entity, &ChunkRenderInstance, &GpuChunkAllocation)>,
) {
    if shadows.active != control.preference.0 {
        shadows.active = control.preference.0;
        bevy::log::info!(
            enabled = shadows.active,
            "DXR terrain shadow preference applied"
        );
    }
    if !control.preference.0 {
        shadows.surface_quads = None;
        shadows.surface_allocation_failed = false;
        shadows.build_budget = default();
        shadows.budget_wait_reported = false;
        shadows.composite_reported.store(false, Ordering::Relaxed);
        shadows.sections.clear();
        shadows.ordered.clear();
        shadows.scene = None;
        shadows.surfaces = None;
        shadows.view = None;
        return;
    }
    if control
        .panorama
        .as_ref()
        .is_some_and(|panorama| !panorama.game_visible())
    {
        return;
    }
    let selected = super::gpu_cull::select_gpu_cull_view(
        views
            .iter()
            .map(|(entity, main, view, ..)| (entity, main, view, false)),
        |entity| {
            views.get(entity).is_ok_and(|(_, _, _, frustum, layers)| {
                super::gpu_cull::gpu_cull_view_eligible(frustum, layers)
            })
        },
    );
    shadows.view = selected;
    let Some((_, _, view, ..)) = selected.and_then(|entity| views.get(entity).ok()) else {
        shadows.sections.clear();
        shadows.ordered.clear();
        shadows.scene = None;
        shadows.surfaces = None;
        return;
    };
    let eye = view.world_from_view.translation();
    let candidates: HashMap<_, _> = terrain
        .iter()
        .filter(|(entity, chunk, _)| {
            !arena.pending_removals.contains(entity) && chunk.cube_layout.solid_len() > 0
        })
        .filter_map(|(entity, chunk, allocation)| {
            let origin = Vec3::from_array(chunk.origin.map(|axis| axis as f32));
            let extent = Vec3::splat(world::SUB_CHUNK_SIDE as f32);
            let distance = (eye - eye.clamp(origin, origin + extent)).length();
            let uploaded = allocation.key == chunk.key && allocation.generation == chunk.generation;
            retain_caster(distance, true, CASTER_RADIUS)
                .then_some((entity, (distance, chunk, uploaded)))
        })
        .collect();
    let material_buffer = shadows.surface_quads.clone();
    shadows.sections.retain(|entity, section| {
        candidates.get(entity).is_some_and(|(_, chunk, uploaded)| {
            let valid = section.origin == chunk.origin
                && opaque_geometry_matches(
                    &section.mesh,
                    section.layout,
                    &chunk.cube_quads,
                    chunk.cube_layout,
                );
            if valid && *uploaded {
                if section.mesh != chunk.cube_quads {
                    section.mesh = chunk.cube_quads.clone();
                    section.layout = chunk.cube_layout;
                    if let Some(buffer) = &material_buffer {
                        surfaces::SceneSurfaces::write_section(&queue, buffer, section);
                    }
                }
                // Refresh identity after equivalent mesh publications without rebuilding its BLAS.
                section.mesh = chunk.cube_quads.clone();
                section.layout = chunk.cube_layout;
            }
            valid
        })
    });
    let desired = select_casters(
        candidates
            .iter()
            .filter_map(|(&entity, &(distance, _, uploaded))| {
                let resident = shadows.sections.contains_key(&entity);
                // Equivalent resident geometry is safe across upload generations; new geometry
                // enters the ray scene only after its terrain allocation has been published.
                (uploaded || resident).then_some((distance, entity, resident))
            }),
        CASTER_RADIUS,
        MAX_SECTIONS,
    );
    // Keep the previous TLAS while replacements build. Its per-instance mask withdraws edited
    // or unloaded casters immediately without switching every other shadow off.
    if shadows.pending.load(Ordering::Acquire) {
        return;
    }
    let build_limit = shadows.build_budget.limit(&control.uploads);
    if build_limit == 0 {
        if !shadows.budget_wait_reported {
            shadows.budget_wait_reported = true;
            bevy::log::info!(
                uploads = control.uploads.chunk_updates,
                deferred = control.uploads.candidates_deferred,
                growing = control.uploads.uploads_waiting_for_growth,
                time_limited = control.uploads.time_limited,
                "DXR shadow builds waiting for terrain budget"
            );
        }
        return;
    }
    let gpu = device.wgpu_device();
    let mut encoder = None;
    let mut inserted = Vec::new();
    for entity in desired
        .iter()
        .copied()
        .filter(|entity| !shadows.sections.contains_key(entity))
        .take(build_limit)
        .collect::<Vec<_>>()
    {
        inserted.push(entity);
        let chunk = candidates[&entity].1;
        let encoder =
            encoder.get_or_insert_with(|| gpu.create_command_encoder(&Default::default()));
        let blas = build_section(gpu, encoder, chunk);
        if shadows.sections.len() == MAX_SECTIONS {
            let victim = shadows
                .sections
                .keys()
                .filter(|entity| !desired.contains(entity))
                .max()
                .copied()
                .expect("a new desired caster replaces one resident outside the desired set");
            shadows.sections.remove(&victim);
        }
        let slot = (0..MAX_SECTIONS as u32)
            .find(|slot| {
                shadows
                    .sections
                    .values()
                    .all(|section| section.slot != *slot)
            })
            .expect("a bounded caster replacement frees one material slot");
        shadows.sections.insert(
            entity,
            Section {
                slot,
                mesh: chunk.cube_quads.clone(),
                layout: chunk.cube_layout,
                origin: chunk.origin,
                blas,
            },
        );
    }
    if shadows.surface_quads.is_none() && !shadows.surface_allocation_failed {
        shadows.surface_quads = surfaces::SceneSurfaces::quad_buffer(&device);
        shadows.surface_allocation_failed = shadows.surface_quads.is_none();
    }
    if let Some(buffer) = shadows.surface_quads.as_ref() {
        // Newly inserted slots are the only full geometry transfer in an ordinary build frame.
        for entity in inserted {
            surfaces::SceneSurfaces::write_section(&queue, buffer, &shadows.sections[&entity]);
        }
    }
    let mut ordered: Vec<_> = shadows.sections.keys().copied().collect();
    ordered.sort_unstable();
    if ordered.is_empty() {
        shadows.ordered.clear();
        shadows.scene = None;
        shadows.surfaces = None;
        return;
    }
    if encoder.is_none() && ordered == shadows.ordered && shadows.scene.is_some() {
        return;
    }
    let mut scene = gpu.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("DXR terrain scene"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: ordered.len() as u32,
    });
    for (index, entity) in ordered.iter().enumerate() {
        let section = &shadows.sections[entity];
        let [x, y, z] = section.origin.map(|axis| axis as f32);
        let transform = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z];
        scene[index] = Some(wgpu::TlasInstance::new(
            &section.blas,
            transform,
            index as u32,
            0xff,
        ));
    }
    let mut encoder = encoder.unwrap_or_else(|| gpu.create_command_encoder(&Default::default()));
    encoder.build_acceleration_structures([], [&scene]);
    queue.submit([encoder.finish()]);
    shadows.pending.store(true, Ordering::Release);
    let pending = shadows.pending.clone();
    queue.on_submitted_work_done(move || pending.store(false, Ordering::Release));
    shadows.ordered = ordered;
    shadows.surfaces = shadows.surface_quads.as_ref().map(|buffer| {
        surfaces::SceneSurfaces::new(gpu, &shadows.ordered, &shadows.sections, buffer)
    });
    shadows.scene = Some(Arc::new(scene));
}

/// Gives terrain most busy frames while guaranteeing bounded shadow progress during streaming.
#[derive(Default)]
struct ShadowBuildBudget {
    busy_frames: u8,
}

impl ShadowBuildBudget {
    fn limit(&mut self, uploads: &ChunkGpuUploadStats) -> usize {
        // Arena growth is a memory-pressure signal, so wait until its allocation completes.
        if uploads.uploads_waiting_for_growth > 0 {
            self.busy_frames = 0;
            return 0;
        }
        if uploads.chunk_updates > 0 || uploads.candidates_deferred > 0 || uploads.time_limited {
            self.busy_frames += 1;
            if self.busy_frames < BUSY_BUILD_INTERVAL {
                return 0;
            }
        }
        self.busy_frames = 0;
        BUILDS_PER_FRAME
    }
}

/// Light, material or allocation identity changes do not change the opaque ray geometry.
fn opaque_geometry_matches(
    cached: &Arc<[PackedQuad]>,
    cached_layout: CubeQuadLayout,
    current: &Arc<[PackedQuad]>,
    current_layout: CubeQuadLayout,
) -> bool {
    let len = cached_layout.solid_len() as usize;
    len == current_layout.solid_len() as usize
        && (Arc::ptr_eq(cached, current)
            || cached[..len]
                .iter()
                .zip(&current[..len])
                .all(|(a, b)| a.words()[0] == b.words()[0]))
}

/// Expands only the mesher's opaque runs. Each greedy quad becomes two triangles.
fn build_section(
    gpu: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    chunk: &ChunkRenderInstance,
) -> wgpu::Blas {
    let solid = &chunk.cube_quads[..chunk.cube_layout.solid_len() as usize];
    let mut positions = Vec::with_capacity(solid.len() * 4);
    let mut indices = Vec::with_capacity(solid.len() * 6);
    for quad in solid {
        let base = positions.len() as u32;
        positions.extend(quad_corners(quad));
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    geometry::build_opaque_mesh(gpu, encoder, &positions, &indices)
}

/// Expands the packed face plane. Ray queries treat both triangle sides as opaque.
fn quad_corners(quad: &PackedQuad) -> [[f32; 3]; 4] {
    let (axis, positive, width, height) = match quad.face() {
        Face::NegativeX => (0, false, 2, 1),
        Face::PositiveX => (0, true, 2, 1),
        Face::NegativeY => (1, false, 0, 2),
        Face::PositiveY => (1, true, 0, 2),
        Face::NegativeZ => (2, false, 0, 1),
        Face::PositiveZ => (2, true, 0, 1),
    };
    let mut a = quad.origin().map(f32::from);
    a[axis] += f32::from(u8::from(positive));
    let mut b = a;
    b[width] += f32::from(quad.width());
    let mut d = a;
    d[height] += f32::from(quad.height());
    let mut c = b;
    c[height] = d[height];
    [a, b, c, d]
}

type ShadowView = (
    &'static ExtractedCamera,
    &'static ExtractedView,
    &'static ViewTarget,
    &'static ViewDepthStencilTexture,
    &'static crate::scene_target::SceneTarget,
    Option<&'static MainPassResolutionOverride>,
    Has<render_scene::shaderpack::ShaderPackRendering>,
);

/// Darkens scene surfaces hit by a sun ray against nearby opaque cube terrain.
fn composite(world: &World, query: ViewQuery<ShadowView>, mut context: RenderContext) -> Result {
    if !world
        .get_resource::<render_scene::terrain_shadows::DxrTerrainShadows>()
        .is_some_and(|preference| preference.0)
    {
        return Ok(());
    }
    let entity = query.entity();
    #[cfg(windows)]
    if native::encoded(world, entity) {
        return Ok(());
    }
    let (camera, view, target, depth, color, resolution, shader_pack) = query.into_inner();
    let Some(shadows) = world.get_resource::<Shadows>() else {
        return Ok(());
    };
    let Some(scene) = shadows
        .scene
        .as_ref()
        .filter(|_| shadows.view == Some(entity))
    else {
        return Ok(());
    };
    let Some(atmosphere) = world.get_resource::<AtmosphereFrame>() else {
        return Ok(());
    };
    let sun = atmosphere.sun_direction();
    if shader_pack
        || world
            .get_resource::<render_scene::PanoramaScene>()
            .is_some_and(|panorama| !panorama.game_visible())
        || atmosphere.sky_kind() != SkyKind::Overworld
        || sun[1] <= 0.0
        || !depth
            .texture()
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
    {
        return Ok(());
    }
    let samples = color.texture.sample_count();
    if samples != depth.texture().sample_count() {
        return Ok(());
    }
    let Some(viewport) = crate::render_bounds::camera_viewport(
        camera.viewport.as_ref(),
        resolution,
        crate::render_bounds::extent(color.color_view(false)),
    ) else {
        return Ok(());
    };
    if viewport.depth != (0.0..1.0) {
        return Ok(());
    }
    let device = context.render_device().clone();
    let gpu = device.wgpu_device();
    let mut pipelines = shadows
        .pipelines
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pipeline = pipelines
        .entry((color.format(), samples))
        .or_insert_with(|| create_pipeline(gpu, color.format(), samples));
    let eye = view.world_from_view.translation();
    let clip_from_world = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
    let frame = Frame {
        world_from_clip: clip_from_world.inverse().to_cols_array_2d(),
        eye_radius: [eye.x, eye.y, eye.z, RECEIVER_RADIUS],
        sun_strength: [sun[0], sun[1], sun[2], 0.55 * atmosphere.daylight()],
        viewport: [
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32,
            viewport.physical_size.y as f32,
        ],
        ray_length: [RAY_LENGTH, SUN_ANGULAR_RADIUS, 0.0, 0.0],
        invalid_instances: invalid_instance_mask(&shadows.ordered, |entity| {
            shadows.sections.contains_key(entity)
        }),
    };
    let uniform = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DXR shadow camera"),
        contents: bytemuck::bytes_of(&frame),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DXR sunlight inputs"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(scene),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: if samples > 1 { 3 } else { 2 },
                resource: wgpu::BindingResource::TextureView(crate::scene_sampling::view_depth(
                    depth,
                )),
            },
        ],
    });
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DXR terrain sunlight"),
            color_attachments: &[Some(color.color_attachment(target, false))],
            ..Default::default()
        });
    let [x, y, width, height] = frame.viewport;
    pass.set_viewport(x, y, width, height, 0.0, 1.0);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.draw(0..3, 0..1);
    if !shadows.composite_reported.swap(true, Ordering::Relaxed) {
        bevy::log::info!(
            sections = shadows.sections.len(),
            "DXR terrain sunlight composite encoded"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equivalent_mesh_publications_reuse_opaque_casters_but_edits_invalidate_them() {
        let layout = CubeQuadLayout::from_solid_counts([1, 0, 0, 0, 0, 0]);
        let mesh: Arc<[PackedQuad]> =
            Arc::from([PackedQuad::new([0; 3], Face::NegativeX, 1, 1, 0)]);
        let material: Arc<[PackedQuad]> =
            Arc::from([PackedQuad::new([0; 3], Face::NegativeX, 1, 1, 1)]);
        let edit: Arc<[PackedQuad]> =
            Arc::from([PackedQuad::new([1, 0, 0], Face::NegativeX, 1, 1, 0)]);
        assert!(opaque_geometry_matches(&mesh, layout, &material, layout));
        assert!(!opaque_geometry_matches(&mesh, layout, &edit, layout));
        assert!(!opaque_geometry_matches(
            &mesh,
            layout,
            &Arc::from([]),
            CubeQuadLayout::default()
        ));
    }

    #[test]
    fn sunlight_shader_accepts_selective_instance_invalidation() {
        let shader = naga::front::wgsl::parse_str(include_str!("dxr.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&shader)
        .unwrap();
    }

    /// Streaming cannot indefinitely starve shadows; only one build is admitted per busy interval.
    #[test]
    fn streaming_has_priority_without_starving_shadow_builds() {
        assert_eq!(
            ShadowBuildBudget::default().limit(&default()),
            BUILDS_PER_FRAME
        );
        for stats in [
            ChunkGpuUploadStats {
                chunk_updates: 1,
                ..default()
            },
            ChunkGpuUploadStats {
                candidates_deferred: 1,
                ..default()
            },
            ChunkGpuUploadStats {
                time_limited: true,
                ..default()
            },
        ] {
            let mut budget = ShadowBuildBudget::default();
            let admissions: Vec<_> = (0..BUSY_BUILD_INTERVAL * 2)
                .map(|_| budget.limit(&stats))
                .collect();
            assert!(
                admissions[..usize::from(BUSY_BUILD_INTERVAL - 1)]
                    .iter()
                    .all(|&count| count == 0)
            );
            assert_eq!(admissions.iter().sum::<usize>(), 2 * BUILDS_PER_FRAME);
            assert!(admissions.iter().all(|&count| count <= BUILDS_PER_FRAME));
        }
    }

    #[test]
    fn arena_growth_waits_and_idle_frames_resume_shadow_builds() {
        let mut budget = ShadowBuildBudget::default();
        let growth = ChunkGpuUploadStats {
            uploads_waiting_for_growth: 1,
            ..default()
        };
        for _ in 0..BUSY_BUILD_INTERVAL * 2 {
            assert_eq!(budget.limit(&growth), 0);
        }
        assert_eq!(budget.limit(&default()), BUILDS_PER_FRAME);
    }

    /// Turning off the menu setting clears cached drawing state without waiting on the GPU.
    #[test]
    fn disabling_shadows_with_an_in_flight_build_releases_the_hidden_view() {
        let (gpu, queue) = wgpu::Device::noop(&Default::default());
        let device = RenderDevice::from(gpu);
        let mut app = SubApp::new();
        app.add_schedule(Render::base_schedule());
        let view = app.world_mut().spawn_empty().id();
        let shadows = Shadows {
            view: Some(view),
            ordered: vec![view],
            pending: Arc::new(AtomicBool::new(true)),
            ..default()
        };
        app.insert_resource(ChunkGpuArena::new(&device))
            .insert_resource(device)
            .insert_resource(RenderQueue::new(queue))
            .insert_resource(ChunkGpuUploadStats::default())
            .insert_resource(render_scene::terrain_shadows::DxrTerrainShadows(false))
            .insert_resource(shadows)
            .configure_sets(
                Render,
                crate::world_passes::WorldPassSystems.run_if(|| false),
            );
        install_preparation(&mut app);
        app.world_mut().run_schedule(Render);
        let shadows = app.world().resource::<Shadows>();
        assert!(shadows.view.is_none());
        assert!(shadows.scene.is_none());
        assert!(shadows.ordered.is_empty());
        assert!(shadows.pending.load(Ordering::Acquire));
    }
}
