//! Headless DXR probe for sunlight occlusion and terrain removal.
#![cfg(windows)]

use bevy::tasks::block_on;
use std::time::Duration;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureUpdateMode, Backends, BindGroupDescriptor,
    BindGroupEntry, BindingResource, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, CreateTlasDescriptor,
    Device, DeviceDescriptor, Dx12Compiler, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, Limits, MapMode, PollType, Queue, Tlas, TlasInstance, include_wgsl,
};

#[path = "../src/chunk/dxr/geometry.rs"]
mod geometry;
#[path = "../src/chunk/dxr/pipeline.rs"]
mod pipeline;

const VERTICES: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];
const INDICES: [u32; 36] = [
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7,
    6, 3, 6, 2,
];
const SAMPLES: usize = 3;
const RESULT_BYTES: u64 = size_of::<[u32; SAMPLES]>() as u64;

#[test]
#[expect(clippy::print_stderr, reason = "reports the GPU or missing fixture")]
fn dx12_terrain_shadow_updates_after_removal() {
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = Backends::DX12;
    descriptor.backend_options.dx12.shader_compiler = Dx12Compiler::default_dynamic_dxc();
    let instance = Instance::new(descriptor);
    let ray_query = Features::EXPERIMENTAL_RAY_QUERY;
    let Some(adapter) = block_on(instance.enumerate_adapters(Backends::DX12))
        .into_iter()
        .find(|adapter| adapter.features().contains(ray_query))
    else {
        eprintln!("missing fixture: DX12 adapter with DXC and DXR Tier 1.1");
        return;
    };
    eprintln!("DXR terrain probe adapter: {}", adapter.get_info().name);
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_features: ray_query,
        required_limits: Limits::default().using_minimum_supported_acceleration_structure_values(),
        // SAFETY: The probe explicitly opts into experimental ray queries on a supporting adapter.
        experimental_features: unsafe { ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .expect("DXR test device");

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    let mut scene = device.create_tlas(&CreateTlasDescriptor {
        label: Some("DXR shadow scene"),
        flags: AccelerationStructureFlags::empty(),
        update_mode: AccelerationStructureUpdateMode::Build,
        max_instances: 2,
    });
    for (index, x) in [0.0, 8.0].into_iter().enumerate() {
        let block = geometry::build_opaque_mesh(&device, &mut encoder, &VERTICES, &INDICES);
        let transform = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0];
        scene[index] = Some(TlasInstance::new(&block, transform, index as u32, 0xff));
    }
    encoder.build_acceleration_structures([], [&scene]);
    queue.submit([encoder.finish()]);

    let shader = device.create_shader_module(include_wgsl!("fixtures/dx12_terrain_shadow.wgsl"));
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("DXR sunlight occlusion"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    assert_eq!(trace(&device, &queue, &pipeline, &scene), [1, 0, 1]);
    for samples in [1, 4] {
        assert_eq!(
            composite_pixel(&device, &queue, &scene, samples, 0.15, [0; 4]),
            [188, 188, 188, 255]
        );
    }

    // A finite sun produces partial coverage at the block edge without blurring lit neighbors.
    for samples in [1, 4] {
        let edge = composite_pixel(&device, &queue, &scene, samples, 1.0, [0; 4]);
        assert!(
            edge[0] > 188 && edge[0] < 255,
            "partial sun coverage: {edge:?}"
        );
        assert_eq!(edge[0], edge[1]);
        assert_eq!(edge[1], edge[2]);
        assert_eq!(
            composite_pixel(&device, &queue, &scene, samples, 1.15, [0; 4]),
            [255; 4]
        );
        assert_eq!(
            composite_surface(&device, &queue, &scene, samples, 0.15, [0; 4], (-2.0, 0.0)),
            [255; 4],
            "a face turned away from the sun keeps its existing ambient lighting"
        );
    }

    // A steep sun-facing plane must retain readable occlusion rather than multiplying
    // its already lightmapped colour by another weak cosine-weighted shadow factor.
    for samples in [1, 4] {
        assert_eq!(
            composite_surface(&device, &queue, &scene, samples, 0.15, [0; 4], (2.0, 4.0)),
            [188, 188, 188, 255],
            "sloped receivers retain full interior shadow contrast"
        );
    }

    // The submitted TLAS still contains both blocks. Withdrawing one instance through the
    // frame mask removes only its shadow, without waiting for a replacement GPU build.
    for samples in [1, 4] {
        assert_eq!(
            composite_pixel(&device, &queue, &scene, samples, 0.15, [1, 0, 0, 0]),
            [255; 4]
        );
        assert_eq!(
            composite_pixel(&device, &queue, &scene, samples, 8.15, [1, 0, 0, 0]),
            [188, 188, 188, 255]
        );
    }

    scene[0] = None;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures([], [&scene]);
    queue.submit([encoder.finish()]);
    assert_eq!(trace(&device, &queue, &pipeline, &scene), [0, 0, 1]);
    for samples in [1, 4] {
        assert_eq!(
            composite_pixel(&device, &queue, &scene, samples, 0.15, [0; 4]),
            [255; 4]
        );
    }
}

/// Runs the game shader over a known depth sample and reads the blended scene color.
fn composite_pixel(
    device: &Device,
    queue: &Queue,
    scene: &Tlas,
    samples: u32,
    ground_x: f32,
    invalid_instances: [u32; 4],
) -> [u8; 4] {
    composite_surface(
        device,
        queue,
        scene,
        samples,
        ground_x,
        invalid_instances,
        (2.0, 0.0),
    )
}

/// Varies the visible side of the receiver plane while keeping the occluding scene fixed.
fn composite_surface(
    device: &Device,
    queue: &Queue,
    scene: &Tlas,
    samples: u32,
    ground_x: f32,
    invalid_instances: [u32; 4],
    receiver: (f32, f32),
) -> [u8; 4] {
    let (eye_y, slope) = receiver;
    use wgpu::{
        Color, Extent3d, LoadOp, Operations, RenderPassColorAttachment,
        RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TexelCopyBufferInfo,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    };
    let size = Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let format = TextureFormat::Rgba8UnormSrgb;
    let texture = |format, sample_count, usage| {
        device.create_texture(&TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let resolved = texture(
        format,
        1,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    );
    let msaa = (samples > 1).then(|| texture(format, samples, TextureUsages::RENDER_ATTACHMENT));
    let resolved_view = resolved.create_view(&Default::default());
    let msaa_view = msaa
        .as_ref()
        .map(|texture| texture.create_view(&Default::default()));
    let depth = texture(
        TextureFormat::Depth32Float,
        samples,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
    );
    let depth_view = depth.create_view(&Default::default());
    let frame = pipeline::Frame {
        world_from_clip: [
            [0.02, 0.02 * slope, 0.0, 0.0],
            [0.0, 0.0, 0.02, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [ground_x, -0.5, 0.35, 1.0],
        ],
        eye_radius: [ground_x, eye_y, 0.35, 32.0],
        sun_strength: [0.0, 1.0, 0.0, 0.5],
        viewport: [0.0, 0.0, 1.0, 1.0],
        ray_length: [32.0, pipeline::SUN_ANGULAR_RADIUS, 0.0, 0.0],
        invalid_instances,
    };
    let uniform = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&frame),
        usage: BufferUsages::UNIFORM,
    });
    let pipeline = pipeline::create_pipeline(device, format, samples);
    let group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(scene),
            },
            BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: if samples > 1 { 3 } else { 2 },
                resource: BindingResource::TextureView(&depth_view),
            },
        ],
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let _clear = encoder.begin_render_pass(&RenderPassDescriptor {
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.5),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: msaa_view.as_ref().unwrap_or(&resolved_view),
                depth_slice: None,
                resolve_target: msaa_view.as_ref().map(|_| &resolved_view),
                ops: Operations {
                    load: LoadOp::Clear(Color::WHITE),
                    store: StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        resolved.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: Default::default(),
        },
        size,
    );
    let submission = queue.submit([encoder.finish()]);
    read_gpu(device, &readback, submission).try_into().unwrap()
}

/// Returns one GPU-written occlusion flag per ground sample.
fn trace(device: &Device, queue: &Queue, pipeline: &ComputePipeline, scene: &Tlas) -> Vec<u32> {
    let result = device.create_buffer(&BufferDescriptor {
        label: Some("DXR shadow flags"),
        size: RESULT_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("DXR shadow readback"),
        size: RESULT_BYTES,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(scene),
            },
            BindGroupEntry {
                binding: 1,
                resource: result.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(SAMPLES as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&result, 0, &readback, 0, RESULT_BYTES);
    let submission = queue.submit([encoder.finish()]);
    let bytes = read_gpu(device, &readback, submission);
    bytes
        .chunks_exact(size_of::<u32>())
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

/// Waits for the exact submission and maps its readback with a bounded timeout.
fn read_gpu(
    device: &Device,
    readback: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
) -> Vec<u8> {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |result| tx.send(result).unwrap());
    let timeout = Duration::from_secs(10);
    device
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(timeout),
        })
        .expect("DXR shadow dispatch completed");
    rx.recv_timeout(timeout)
        .expect("DXR readback callback")
        .expect("map DXR readback");
    slice.get_mapped_range().expect("read DXR result").to_vec()
}
