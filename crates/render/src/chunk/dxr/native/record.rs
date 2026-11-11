//! Connects the native passes to the live view without taking over world publication.
use super::*;
use surfaces::LightGpu;

pub(in super::super) fn composite(
    world: &World,
    query: ViewQuery<ShadowView>,
    mut context: RenderContext,
) -> Result {
    if !world
        .get_resource::<render_scene::terrain_shadows::DxrTerrainShadows>()
        .is_some_and(|p| p.0)
        || world
            .get_resource::<render_scene::PanoramaScene>()
            .is_some_and(|p| !p.game_visible())
    {
        return Ok(());
    }
    let entity = query.entity();
    let (camera, view, target, depth, colour, resolution, shader_pack) = query.into_inner();
    let Some(shadows) = world
        .get_resource::<Shadows>()
        .filter(|s| s.view == Some(entity))
    else {
        return Ok(());
    };
    let Some((scene, surfaces)) = shadows.scene.as_ref().zip(shadows.surfaces.as_ref()) else {
        return Ok(());
    };
    if shadows.native_address == 0 {
        return Ok(());
    }
    let Some(state) = world.get_resource::<NativeTerrain>() else {
        return Ok(());
    };
    if shader_pack
        || !depth
            .texture()
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
    {
        return Ok(());
    }
    let samples = colour.texture.sample_count();
    if samples != depth.texture().sample_count() {
        return Ok(());
    }
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
    let device = context.render_device().clone();
    let gpu = device.wgpu_device();
    let Some(programs) = state.programs(&device, colour.format(), samples) else {
        return Ok(());
    };
    let Some(atmosphere) = world.get_resource::<AtmosphereFrame>() else {
        return Ok(());
    };
    let Some(textures) = world
        .get_resource::<crate::chunk::gpu::bind_groups::ChunkGpuTextureAssets>()
        .and_then(|a| a.prepared.as_ref())
    else {
        return Ok(());
    };
    let Some(size) = grid(
        viewport.physical_size.to_array(),
        u64::from(gpu.limits().max_storage_buffer_binding_size).min(gpu.limits().max_buffer_size),
    ) else {
        return Ok(());
    };
    if *state
        .failed_size
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        == Some(size)
    {
        return Ok(());
    }
    let buffers = {
        let mut cache = state
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.as_ref().is_none_or(|b| b.size != size) {
            let Some(buffers) = Buffers::new(&device, size) else {
                *state
                    .failed_size
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(size);
                return Ok(());
            };
            *state
                .failed_size
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            buffers.initialize(context.command_encoder());
            *cache = Some(Arc::new(buffers));
        }
        cache.as_ref().unwrap().clone()
    };
    let eye = view.world_from_view.translation();
    let sun = atmosphere.sun_direction();
    let fog = atmosphere.fog_color();
    let sky = atmosphere.sky_zenith();
    let mut lights: Vec<_> = world
        .get_resource::<render_scene::terrain_shadows::DxrLocalLights>()
        .into_iter()
        .flat_map(|l| l.0.iter().take(32))
        .map(|l| LightGpu {
            position_radius: [l.position[0], l.position[1], l.position[2], l.radius],
            colour_intensity: [l.colour[0], l.colour[1], l.colour[2], l.intensity],
        })
        .collect();
    let params = Params {
        frame: Frame {
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
            ray_length: [RAY_LENGTH, SUN_ANGULAR_RADIUS, lights.len() as f32, 0.0],
            invalid_instances: invalid_instance_mask(&shadows.ordered, |id| {
                shadows.sections.contains_key(id)
            }),
        },
        grid: [
            size[0],
            size[1],
            samples,
            u32::from(atmosphere.sky_kind() == SkyKind::Overworld && sun[1] > 0.0),
        ],
        fog: [
            fog[0],
            fog[1],
            fog[2],
            (0.6 / (atmosphere.fog_end() - atmosphere.fog_start()).max(8.0)).min(0.035),
        ],
        sky: [sky[0], sky[1], sky[2], 0.55],
    };
    if lights.is_empty() {
        lights.push(LightGpu {
            position_radius: [0.0; 4],
            colour_intensity: [0.0; 4],
        });
    }
    let uniform = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Native DXR view"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let light_buffer = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Native DXR lights"),
        contents: bytemuck::cast_slice(&lights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut material_entries = vec![
        entry(1, uniform.as_entire_binding()),
        entry(
            if samples > 1 { 3 } else { 2 },
            wgpu::BindingResource::TextureView(crate::scene_sampling::view_depth(depth)),
        ),
        entry(4, surfaces.instances.as_entire_binding()),
        entry(5, surfaces.quads.as_entire_binding()),
        entry(6, textures.material_buffer.as_entire_binding()),
        entry(7, textures.enhanced_texture_refs.as_entire_binding()),
        entry(8, light_buffer.as_entire_binding()),
        entry(9, wgpu::BindingResource::Sampler(&programs.sampler)),
        entry(21, wgpu::BindingResource::Sampler(&textures.sampler)),
    ];
    for (index, texture) in textures
        .views
        .iter()
        .chain(textures.enhanced_views.iter())
        .enumerate()
    {
        material_entries.push(entry(
            10 + index as u32,
            wgpu::BindingResource::TextureView(texture),
        ));
    }
    let deferred_entries = [
        entry(0, uniform.as_entire_binding()),
        entry(1, buffers.receivers.as_entire_binding()),
        entry(2, buffers.hits.as_entire_binding()),
        entry(4, buffers.lighting.as_entire_binding()),
        entry(5, buffers.fog.as_entire_binding()),
    ];
    let output_entries = [
        entry(0, buffers.receivers.as_entire_binding()),
        entry(1, buffers.normals.as_entire_binding()),
    ];
    let groups = |layouts: [wgpu::BindGroupLayout; 3], ids: [&[u32]; 3]| {
        std::array::from_fn::<_, 3, _>(|index| {
            let source = match index {
                0 => material_entries.as_slice(),
                1 => deferred_entries.as_slice(),
                _ => output_entries.as_slice(),
            };
            group(gpu, &layouts[index], source, ids[index])
        })
    };
    // Resolve every fallible native command before publishing any of them to the graph.
    let commands = [
        dispatch::encode(
            gpu,
            &programs,
            0,
            shadows.native_address,
            &uniform,
            &light_buffer,
            &buffers,
        )?,
        dispatch::encode(
            gpu,
            &programs,
            1,
            shadows.native_address,
            &uniform,
            &light_buffer,
            &buffers,
        )?,
        dispatch::encode(
            gpu,
            &programs,
            2,
            shadows.native_address,
            &uniform,
            &light_buffer,
            &buffers,
        )?,
    ];
    let receive_entry = if samples > 1 {
        "receivers_msaa"
    } else {
        "receivers_single"
    };
    let receive_ids =
        std::array::from_fn::<_, 3, _>(|group| programs::bindings(receive_entry, group));
    let receivers = groups(
        std::array::from_fn(|i| programs.receivers.get_bind_group_layout(i as u32)),
        std::array::from_fn(|i| receive_ids[i].as_slice()),
    );
    compute(&mut context, &programs.receivers, &receivers, size);
    let mut commands = commands.into_iter();
    for buffer in commands.next().unwrap() {
        context.add_command_buffer(buffer);
    }
    let normal_ids =
        std::array::from_fn::<_, 3, _>(|group| programs::bindings("mapped_normals", group));
    let normals = groups(
        std::array::from_fn(|i| programs.normals.get_bind_group_layout(i as u32)),
        std::array::from_fn(|i| normal_ids[i].as_slice()),
    );
    compute(&mut context, &programs.normals, &normals, size);
    for phase in commands {
        for buffer in phase {
            context.add_command_buffer(buffer);
        }
    }
    let render_groups = |pipeline: &wgpu::RenderPipeline, name| {
        std::array::from_fn::<_, 2, _>(|index| {
            group(
                gpu,
                &pipeline.get_bind_group_layout(index as u32),
                if index == 0 {
                    &material_entries
                } else {
                    &deferred_entries
                },
                &programs::bindings(name, index),
            )
        })
    };
    let albedo = render_groups(&programs.albedo, "albedo");
    let lighting = render_groups(&programs.lighting, "lighting");
    let fog = render_groups(&programs.fog, "fog_light");
    {
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Native DXR materials, shadows, local lighting and beams"),
                color_attachments: &[Some(colour.color_attachment(target, false))],
                ..Default::default()
            });
        let [x, y, w, h] = params.frame.viewport;
        pass.set_viewport(x, y, w, h, 0.0, 1.0);
        for (pipeline, groups) in [
            (&programs.albedo, &albedo),
            (&programs.lighting, &lighting),
            (&programs.fog, &fog),
        ] {
            pass.set_pipeline(pipeline);
            for (index, group) in groups.iter().enumerate() {
                pass.set_bind_group(index as u32, group, &[]);
            }
            pass.draw(0..3, 0..1);
        }
    }
    state
        .pending
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(KeepAlive {
            _programs: programs,
            _scene: scene.clone(),
            _buffers: buffers,
            _uniform: uniform,
            _lights: light_buffer,
        });
    *state
        .encoded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(entity);
    if !state.reported.swap(true, Ordering::Relaxed) {
        bevy::log::info!(
            width = size[0],
            height = size[1],
            "Native DXR DispatchRays active: PBR, reflections, soft shadows, AO, local lights and volumetric beams"
        );
    }
    Ok(())
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}
fn group(
    gpu: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
    ids: &[u32],
) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .filter(|entry| ids.contains(&entry.binding))
        .cloned()
        .collect();
    gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Native DXR deferred inputs"),
        layout,
        entries: &entries,
    })
}
fn compute(
    context: &mut RenderContext,
    pipeline: &wgpu::ComputePipeline,
    groups: &[wgpu::BindGroup; 3],
    size: [u32; 2],
) {
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Native DXR receiver materials"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    for (index, group) in groups.iter().enumerate() {
        pass.set_bind_group(index as u32, group, &[]);
    }
    pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
}
