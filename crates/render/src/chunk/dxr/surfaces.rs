//! Ray-hit material addressing shared by PBR receivers and reflected opaque terrain.
use super::*;
#[cfg(any(windows, feature = "enhanced"))]
use bytemuck::Zeroable;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceGpu {
    origin: [f32; 3],
    quad_start: u32,
}

pub(super) struct SurfacePipelines {
    lighting: wgpu::RenderPipeline,
    albedo: wgpu::RenderPipeline,
    authored_sampler: wgpu::Sampler,
    reported: std::sync::atomic::AtomicU8,
}

const _: () = assert!(std::mem::size_of::<PackedQuad>() == std::mem::size_of::<[u32; 2]>());

pub(super) struct SceneSurfaces {
    pub(super) instances: Buffer,
    pub(super) quads: Buffer,
}

const QUADS_PER_SECTION: usize =
    world::SUB_CHUNK_SIDE * world::SUB_CHUNK_SIDE * world::SUB_CHUNK_SIDE * 6;

impl SceneSurfaces {
    /// Reserves bounded stable slots; only a newly built or changed section uploads its words.
    pub(super) fn quad_buffer(device: &RenderDevice) -> Option<Buffer> {
        let gpu = device.wgpu_device();
        let bytes = (MAX_SECTIONS * QUADS_PER_SECTION * std::mem::size_of::<PackedQuad>()) as u64;
        crate::gpu_allocation::checked(device, "DXR stable material slots", bytes, || {
            gpu.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DXR stable material slots"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
            .into()
        })
    }

    pub(super) fn write_section(queue: &RenderQueue, buffer: &Buffer, section: &Section) {
        let quads = &section.mesh[..section.layout.solid_len() as usize];
        assert!(quads.len() <= QUADS_PER_SECTION);
        let words: Vec<_> = quads.iter().map(PackedQuad::words).collect();
        queue.write_buffer(
            buffer,
            (section.slot as usize * QUADS_PER_SECTION * std::mem::size_of::<PackedQuad>()) as u64,
            bytemuck::cast_slice(&words),
        );
    }

    /// Instance addresses are tiny and travel with the TLAS; shared quad slots stay resident.
    pub(super) fn new(
        device: &wgpu::Device,
        ordered: &[Entity],
        sections: &HashMap<Entity, Section>,
        quads: &Buffer,
    ) -> Self {
        let instances: Vec<_> = ordered
            .iter()
            .map(|entity| {
                let section = &sections[entity];
                InstanceGpu {
                    origin: section.origin.map(|x| x as f32),
                    quad_start: section.slot * QUADS_PER_SECTION as u32,
                }
            })
            .collect();
        Self {
            instances: device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DXR scene material instances"),
                    contents: bytemuck::cast_slice(&instances),
                    usage: wgpu::BufferUsages::STORAGE,
                })
                .into(),
            quads: quads.clone(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LightGpu {
    pub(super) position_radius: [f32; 4],
    pub(super) colour_intensity: [f32; 4],
}

/// Records PBR/metal and local-light compositing separately from the sunlight shadow pass.
#[cfg(any(windows, feature = "enhanced"))]
pub(super) fn composite(
    world: &World,
    query: ViewQuery<ShadowView>,
    mut context: RenderContext,
) -> Result {
    if !world
        .get_resource::<render_scene::terrain_shadows::DxrTerrainShadows>()
        .is_some_and(|value| value.0)
        || world
            .get_resource::<render_scene::PanoramaScene>()
            .is_some_and(|value| !value.game_visible())
    {
        return Ok(());
    }
    let entity = query.entity();
    #[cfg(windows)]
    if native::encoded(world, entity) {
        return Ok(());
    }
    let (camera, view, target, depth, colour, resolution, shader_pack) = query.into_inner();
    if shader_pack
        || !depth
            .texture()
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
    {
        return Ok(());
    }
    let Some(shadows) = world
        .get_resource::<Shadows>()
        .filter(|shadows| shadows.view == Some(entity))
    else {
        return Ok(());
    };
    let Some((scene, surfaces)) = shadows.scene.as_ref().zip(shadows.surfaces.as_ref()) else {
        return Ok(());
    };
    let Some(textures) = world
        .get_resource::<super::super::gpu::bind_groups::ChunkGpuTextureAssets>()
        .and_then(|assets| assets.prepared.as_ref())
    else {
        return Ok(());
    };
    let lights = world.get_resource::<render_scene::terrain_shadows::DxrLocalLights>();
    if textures.authored_source.is_none() && lights.is_none_or(|lights| lights.0.is_empty()) {
        return Ok(());
    }
    let Some(atmosphere) = world.get_resource::<AtmosphereFrame>() else {
        return Ok(());
    };
    let Some(viewport) = crate::render_bounds::camera_viewport(
        camera.viewport.as_ref(),
        resolution,
        crate::render_bounds::extent(colour.color_view(false)),
    ) else {
        return Ok(());
    };
    if viewport.depth != (0.0..1.0) {
        return Ok(());
    }
    let samples = colour.texture.sample_count();
    if samples != depth.texture().sample_count() {
        return Ok(());
    }
    let device = context.render_device().clone();
    let gpu = device.wgpu_device();
    let mut pipelines = shadows
        .surface_pipelines
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pipeline = pipelines
        .entry((colour.format(), samples))
        .or_insert_with(|| create_pipeline(gpu, colour.format(), samples));
    let eye = view.world_from_view.translation();
    let sun = atmosphere.sun_direction();
    let mut frame = Frame {
        world_from_clip: view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse())
            .inverse()
            .to_cols_array_2d(),
        eye_radius: [eye.x, eye.y, eye.z, RECEIVER_RADIUS],
        sun_strength: [sun[0], sun[1], sun[2], atmosphere.daylight()],
        viewport: [
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32,
            viewport.physical_size.y as f32,
        ],
        ray_length: [RAY_LENGTH, 0.0, 0.0, 0.0],
        invalid_instances: invalid_instance_mask(&shadows.ordered, |id| {
            shadows.sections.contains_key(id)
        }),
    };
    let sky = atmosphere
        .sky_zenith()
        .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u32);
    frame.ray_length[1] = (sky[0] | sky[1] << 8 | sky[2] << 16) as f32;
    let mut light_values: Vec<_> = lights
        .into_iter()
        .flat_map(|lights| &lights.0)
        .map(|light| LightGpu {
            position_radius: [
                light.position[0],
                light.position[1],
                light.position[2],
                light.radius,
            ],
            colour_intensity: [
                light.colour[0],
                light.colour[1],
                light.colour[2],
                light.intensity,
            ],
        })
        .collect();
    frame.ray_length[2] = light_values.len() as f32;
    if light_values.is_empty() {
        light_values.push(LightGpu::zeroed());
    }
    let uniform = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DXR PBR frame"),
        contents: bytemuck::bytes_of(&frame),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let light_buffer = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DXR local lights"),
        contents: bytemuck::cast_slice(&light_values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut entries = vec![
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
            resource: wgpu::BindingResource::TextureView(crate::scene_sampling::view_depth(depth)),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: surfaces.instances.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: surfaces.quads.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: textures.material_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: textures.enhanced_texture_refs.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 8,
            resource: light_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&pipeline.authored_sampler),
        },
    ];
    entries.push(wgpu::BindGroupEntry {
        binding: 21,
        resource: wgpu::BindingResource::Sampler(&textures.sampler),
    });
    for (index, view) in textures
        .views
        .iter()
        .chain(textures.enhanced_views.iter())
        .enumerate()
    {
        entries.push(wgpu::BindGroupEntry {
            binding: 10 + index as u32,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    let group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DXR PBR materials and lights"),
        layout: &pipeline.lighting.get_bind_group_layout(0),
        entries: &entries,
    });
    let albedo_group = textures.authored_source.as_ref().map(|_| {
        let entries: Vec<_> = entries
            .iter()
            .filter(|entry| entry.binding != 8)
            .cloned()
            .collect();
        gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DXR authored albedo"),
            layout: &pipeline.albedo.get_bind_group_layout(0),
            entries: &entries,
        })
    });
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DXR PBR reflections and local lights"),
            color_attachments: &[Some(colour.color_attachment(target, false))],
            ..default()
        });
    let [x, y, width, height] = frame.viewport;
    pass.set_viewport(x, y, width, height, 0.0, 1.0);
    if let Some(albedo_group) = &albedo_group {
        pass.set_pipeline(&pipeline.albedo);
        pass.set_bind_group(0, albedo_group, &[]);
        pass.draw(0..3, 0..1);
    }
    pass.set_pipeline(&pipeline.lighting);
    pass.set_bind_group(0, &group, &[]);
    pass.draw(0..3, 0..1);
    let features =
        u8::from(textures.authored_source.is_some()) | (u8::from(frame.ray_length[2] > 0.0) << 1);
    if pipeline.reported.fetch_or(features, Ordering::Relaxed) & features != features {
        bevy::log::info!(
            authored_materials = textures.authored_source.is_some(),
            local_lights = frame.ray_length[2] as u32,
            "DXR material/reflection/local-light composite encoded"
        );
    }
    Ok(())
}

pub(super) fn shader_source() -> String {
    material_source(include_str!("surfaces.wgsl"))
}

pub(super) fn material_source(source: &str) -> String {
    source
        .replace("// DXR_MATERIAL_FUNCTIONS", include_str!("material.wgsl"))
        .replace(
            "DXR_LAYER_MASK",
            &format!("{}u", assets::TEXTURE_LAYER_MASK),
        )
        .replace(
            "DXR_PAGE_SHIFT",
            &format!("{}u", assets::TEXTURE_PAGE_SHIFT),
        )
        .replace(
            "DXR_WIDTH_SHIFT",
            &format!("{}u", crate::material_shader::TEXTURE_WIDTH_SHIFT),
        )
        .replace(
            "DXR_HEIGHT_SHIFT",
            &format!("{}u", crate::material_shader::TEXTURE_HEIGHT_SHIFT),
        )
        .replace(
            "DXR_GRID_SHIFT",
            &format!("{}u", crate::material_shader::TEXTURE_GRID_SHIFT),
        )
        .replace(
            "DXR_AUTHORED_MASK",
            &format!("{}u", assets::texture::PBR_REF_COLOR - 1),
        )
        .replace(
            "DXR_REFERENCE_WORDS",
            &format!(
                "{}u",
                render_scene::EnhancedTextureAssets::REFERENCE_BASE_WORDS
            ),
        )
}

fn create_pipeline(
    gpu: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> SurfacePipelines {
    let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("DXR PBR surfaces"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });
    let pipeline = |albedo| {
        gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DXR PBR surfaces"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(match (albedo, samples > 1) {
                    (true, true) => "albedo_multisampled",
                    (true, false) => "albedo_single",
                    (false, true) => "multisampled",
                    (false, false) => "single",
                }),
                compilation_options: default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(if albedo {
                        wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Dst,
                                dst_factor: wgpu::BlendFactor::Zero,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }
                    } else {
                        wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: samples,
                ..default()
            },
            multiview_mask: None,
            cache: None,
        })
    };
    SurfacePipelines {
        albedo: pipeline(true),
        lighting: pipeline(false),
        reported: std::sync::atomic::AtomicU8::new(0),
        authored_sampler: gpu.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DXR authored linear material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "names the missing optional hardware fixture"
    )]
    fn dxr_pbr_pipelines_compile_on_inline_query_hardware() {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::DX12;
        descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::default_dynamic_dxc();
        let instance = wgpu::Instance::new(descriptor);
        let Some(adapter) =
            bevy::tasks::block_on(instance.enumerate_adapters(wgpu::Backends::DX12))
                .into_iter()
                .find(|adapter| {
                    adapter
                        .features()
                        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                })
        else {
            eprintln!(
                "missing fixture: DX12 DXC and inline ray-query GPU for PBR pipeline validation"
            );
            return;
        };
        let (device, _) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits:
                wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
            // SAFETY: This optional hardware probe explicitly opts into the supported experimental feature.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..default()
        }))
        .unwrap();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        for format in [
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        ] {
            for samples in [1, 4] {
                let _ = create_pipeline(&device, format, samples);
            }
        }
        assert!(bevy::tasks::block_on(scope.pop()).is_none());
    }

    #[test]
    fn dxr_pbr_shader_bindings_match_both_composites_and_sample_counts() {
        let source = shader_source();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        for (index, entry) in module
            .entry_points
            .iter()
            .enumerate()
            .filter(|(_, ep)| ep.stage == naga::ShaderStage::Fragment)
        {
            let used = info.get_entry_point(index);
            let mut actual: Vec<_> = module
                .global_variables
                .iter()
                .filter_map(|(handle, global)| {
                    global
                        .binding
                        .as_ref()
                        .filter(|_| !used[handle].is_empty())
                        .map(|binding| binding.binding)
                })
                .collect();
            actual.sort_unstable();
            let albedo = entry.name.starts_with("albedo");
            let msaa = entry.name.ends_with("multisampled");
            let expected: Vec<_> = (0..=21)
                .filter(|&id| id != if msaa { 2 } else { 3 })
                .filter(|&id| !albedo || id != 8)
                .collect();
            assert_eq!(actual, expected, "{}", entry.name);
        }
    }
}
