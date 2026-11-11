//! A synthetic native frame verifies the live shader library on software or hardware DX12.
use super::*;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "reports adapter identity and optional DXIL/GPU fixtures"
)]
fn native_dxr_frame_shadows_lights_reflections_and_withdrawal() {
    if DXIL.is_empty() {
        eprintln!("missing fixture: CINNABAR_NATIVE_DXR_DXIL terrain library");
        return;
    }
    let compiler = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.local/dxr/compiler/dxcompiler.dll");
    let Ok(compiler) = compiler.canonicalize() else {
        eprintln!("missing fixture: verified DXC library");
        return;
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::DX12;
    descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
        dxc_path: compiler.to_string_lossy().into_owned(),
    };
    let instance = wgpu::Instance::new(descriptor);
    let Some(adapter) = bevy::tasks::block_on(instance.enumerate_adapters(wgpu::Backends::DX12))
        .into_iter()
        .find(|a| {
            a.features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        })
    else {
        eprintln!("missing fixture: DX12 acceleration structure adapter");
        return;
    };
    eprintln!("Native deferred DXR adapter: {:?}", adapter.get_info());
    let (gpu, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits:
            wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
        // SAFETY: This test explicitly opts into the admitted AS construction capability.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .unwrap();
    let scope = gpu.push_error_scope(wgpu::ErrorFilter::Validation);
    let programs = programs::create(&gpu, wgpu::TextureFormat::Bgra8UnormSrgb, 1).unwrap();
    // The alternate sample count must compile too; the game can change antialiasing at runtime.
    let _msaa = programs::create(&gpu, wgpu::TextureFormat::Bgra8UnormSrgb, 4).unwrap();
    let device = RenderDevice::from(gpu.clone());
    let buffers = Buffers::new(&device, [2, 1]).unwrap();
    let mut build = gpu.create_command_encoder(&Default::default());
    buffers.initialize(&mut build);
    let blas = geometry::build_opaque_mesh(
        &gpu,
        &mut build,
        &[[-1.0, -1.0, 2.0], [0.0, 1.0, 2.0], [1.0, -1.0, 2.0]],
        &[0, 1, 2],
    );
    let mut scene = gpu.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    scene[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        0,
        0xff,
    ));
    build.build_acceleration_structures([], [&scene]);
    queue.submit([build.finish()]);
    let scene_address = super::address(&mut scene);
    // ByteAddressBuffer ABI: float3 world point, ten-bit normal + valid bit; two float4 normal rows.
    let normal = 512u32 | (512u32 << 10) | 0x80000000;
    let receivers = [
        [0f32.to_bits(), 0f32.to_bits(), 3f32.to_bits(), normal],
        [5f32.to_bits(), 0f32.to_bits(), 3f32.to_bits(), normal],
    ];
    queue.write_buffer(&buffers.receivers, 0, bytemuck::cast_slice(&receivers));
    queue.write_buffer(
        &buffers.normals,
        0,
        bytemuck::cast_slice(&[[0f32, 0., -1., 0.1, 0., 0., -1., 1.]; 2]),
    );
    let lights = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0f32, 0., 0., 10., 1., 0.5, 0.2, 1.]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut params = Params {
        frame: Frame {
            world_from_clip: Mat4::IDENTITY.to_cols_array_2d(),
            eye_radius: [0., 0., 0., 32.],
            sun_strength: [0., 0., -1., 1.],
            viewport: [0., 0., 2., 1.],
            ray_length: [40., 0.0087, 1., 0.],
            invalid_instances: [0; 4],
        },
        grid: [2, 1, 1, 1],
        fog: [0.5, 0.6, 0.8, 0.02],
        sky: [0.5, 0.6, 0.8, 0.55],
    };
    let uniform = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    for withdrawn in [false, true] {
        params.frame.invalid_instances[0] = u32::from(withdrawn);
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
        let mut commands = Vec::new();
        for phase in 0..3 {
            commands.extend(
                dispatch::encode(
                    &gpu,
                    &programs,
                    phase,
                    scene_address,
                    &uniform,
                    &lights,
                    &buffers,
                )
                .unwrap(),
            );
        }
        let readback = gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 144,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut copy = gpu.create_command_encoder(&Default::default());
        copy.copy_buffer_to_buffer(&buffers.hits, 0, &readback, 0, 32);
        copy.copy_buffer_to_buffer(&buffers.lighting, 0, &readback, 32, 96);
        copy.copy_buffer_to_buffer(&buffers.fog, 0, &readback, 128, 16);
        commands.push(copy.finish());
        queue.submit(commands);
        let (send, receive) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap();
            });
        gpu.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(20)),
        })
        .unwrap();
        receive.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        assert_eq!(
            words[3],
            u32::from(!withdrawn),
            "primary hit withdrawn={withdrawn}"
        );
        assert_eq!(words[7], 0, "second primary ray misses");
        let shadow = f32::from_bits(words[8]);
        let visibility = f32::from_bits(words[19]);
        let reflection = words[15];
        let light = words[10] & 3;
        eprintln!(
            "withdrawn={withdrawn}: shadow={shadow} sun={visibility} reflection={reflection} local={light}"
        );
        if withdrawn {
            assert_eq!(shadow, 0.);
            assert_eq!(visibility, 1.);
            assert_eq!(reflection, 0);
            assert_eq!(light, 2);
        } else {
            assert!(shadow > 0.5);
            assert_eq!(visibility, 0.);
            assert_eq!(reflection, 1);
            assert_eq!(light, 0);
        }
        assert!(words[32..35].iter().all(|v| f32::from_bits(*v).is_finite()));
        if withdrawn {
            assert!(
                words[32..35].iter().any(|v| f32::from_bits(*v) > 0.0),
                "visible light must scatter in the medium"
            );
        }
        drop(mapped);
        readback.unmap();
    }
    assert!(bevy::tasks::block_on(scope.pop()).is_none());
}
