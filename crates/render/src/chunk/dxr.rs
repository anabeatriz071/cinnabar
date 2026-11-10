//! Bounded Windows DXR sunlight prototype for resident opaque cube terrain.

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
use wgpu::util::DeviceExt;

mod geometry;
mod pipeline;
use pipeline::{Frame, create_pipeline};

const MAX_SECTIONS: u32 = 128;
const BUILDS_PER_FRAME: usize = 6;
const CASTER_RADIUS: f32 = 72.0;
const RECEIVER_RADIUS: f32 = 32.0;
const RAY_LENGTH: f32 = CASTER_RADIUS - RECEIVER_RADIUS;

#[derive(Resource, Default)]
struct Shadows {
    sections: HashMap<Entity, Section>,
    ordered: Vec<Entity>,
    scene: Option<wgpu::Tlas>,
    view: Option<Entity>,
    pipelines: Mutex<HashMap<(wgpu::TextureFormat, u32), wgpu::RenderPipeline>>,
}

struct Section {
    mesh: Arc<[PackedQuad]>,
    layout: CubeQuadLayout,
    origin: [i32; 3],
    blas: wgpu::Blas,
}

/// Registers a separate pass after opaque terrain, including late GPU-culled draws.
pub(super) fn install(app: &mut App) {
    if !crate::device_settings::dxr_terrain_shadows_requested() {
        return;
    }
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
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
    render.add_systems(RenderStartup, init).add_systems(
        Render,
        prepare
            .run_if(bevy::prelude::resource_exists::<Shadows>)
            .in_set(crate::world_passes::WorldPassSystems)
            .in_set(RenderSystems::PrepareResources),
    );
    app.add_systems(Last, super::admit_depth_sampling);
}

/// Leaves the ordinary renderer active when the selected device lacks DXR.
fn init(mut commands: Commands, adapter: Res<RenderAdapter>, device: Res<RenderDevice>) {
    if adapter.get_info().backend == wgpu::Backend::Dx12
        && device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    {
        commands.init_resource::<Shadows>();
        bevy::log::info!("experimental DXR terrain shadows enabled");
    } else {
        bevy::log::warn!("DXR terrain shadows require a DX12 device with inline ray queries");
    }
}

/// Retains one acceleration structure per nearby published mesh and rebuilds the scene on edits.
fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    arena: Res<ChunkGpuArena>,
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
        return;
    };
    let eye = view.world_from_view.translation();
    let mut candidates: Vec<_> = terrain
        .iter()
        .filter(|(entity, chunk, allocation)| {
            !arena.pending_removals.contains(entity)
                && allocation.key == chunk.key
                && allocation.generation == chunk.generation
                && chunk.cube_layout.solid_len() > 0
        })
        .filter_map(|(entity, chunk, _)| {
            let origin = Vec3::from_array(chunk.origin.map(|axis| axis as f32));
            let extent = Vec3::splat(world::SUB_CHUNK_SIDE as f32);
            let distance = (eye - eye.clamp(origin, origin + extent)).length_squared();
            (distance < CASTER_RADIUS * CASTER_RADIUS).then_some((distance, entity, chunk))
        })
        .collect();
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    candidates.truncate(MAX_SECTIONS as usize);
    let keep: HashMap<_, _> = candidates
        .iter()
        .map(|&(_, entity, chunk)| (entity, chunk))
        .collect();
    shadows.sections.retain(|entity, section| {
        keep.get(entity).is_some_and(|chunk| {
            section.origin == chunk.origin
                && section.layout == chunk.cube_layout
                && Arc::ptr_eq(&section.mesh, &chunk.cube_quads)
        })
    });
    let gpu = device.wgpu_device();
    let mut encoder = None;
    for (_, entity, chunk) in candidates
        .into_iter()
        .filter(|(_, entity, _)| !shadows.sections.contains_key(entity))
        .take(BUILDS_PER_FRAME)
        .collect::<Vec<_>>()
    {
        let encoder =
            encoder.get_or_insert_with(|| gpu.create_command_encoder(&Default::default()));
        let blas = build_section(gpu, encoder, chunk);
        shadows.sections.insert(
            entity,
            Section {
                mesh: chunk.cube_quads.clone(),
                layout: chunk.cube_layout,
                origin: chunk.origin,
                blas,
            },
        );
    }
    let mut ordered: Vec<_> = shadows.sections.keys().copied().collect();
    ordered.sort_unstable();
    if ordered.is_empty() {
        shadows.ordered.clear();
        shadows.scene = None;
        return;
    }
    if encoder.is_none() && ordered == shadows.ordered {
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
        scene[index] = Some(wgpu::TlasInstance::new(&section.blas, transform, 0, 0xff));
    }
    let mut encoder = encoder.unwrap_or_else(|| gpu.create_command_encoder(&Default::default()));
    encoder.build_acceleration_structures([], [&scene]);
    queue.submit([encoder.finish()]);
    shadows.ordered = ordered;
    shadows.scene = Some(scene);
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
    let entity = query.entity();
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
        ray_length: [RAY_LENGTH, 0.0, 0.0, 0.0],
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
    Ok(())
}
