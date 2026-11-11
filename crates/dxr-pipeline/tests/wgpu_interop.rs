//! Native TraceRay uses the existing wgpu device, submission tracking and acceleration structures.
#![cfg(windows)]
use dxr_pipeline::native::{Exports, HitGroup, Pipeline};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

#[test]
#[expect(
    clippy::print_stderr,
    reason = "records adapter identity and optional GPU fixtures"
)]
fn dxr_native_trace_hits_misses_and_clears_removed_wgpu_instance() {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/dxr/native-trace.dxil");
    let Ok(dxil) = std::fs::read(fixture) else {
        eprintln!("missing fixture: native-trace.dxil");
        return;
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::DX12;
    let compiler = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.local/dxr/compiler/dxcompiler.dll");
    let Ok(compiler) = compiler.canonicalize() else {
        eprintln!(
            "missing fixture: verified DXC library at {}",
            compiler.display()
        );
        return;
    };
    descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
        dxc_path: compiler.to_string_lossy().into_owned(),
    };
    let instance = wgpu::Instance::new(descriptor);
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::DX12));
    for adapter in &adapters {
        eprintln!(
            "DX12 interop candidate: {:?}; AS admission: {}",
            adapter.get_info(),
            adapter
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        );
    }
    let Some(adapter) = adapters.into_iter().find(|adapter| {
        adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    }) else {
        eprintln!("missing fixture: DX12 adapter with acceleration structure support");
        return;
    };
    eprintln!("Native wgpu RT interop adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // This existing feature admits AS construction; the shader itself uses TraceRay, not ray queries.
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits:
            wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .unwrap();
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let vertices = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("RT triangle"),
        contents: bytemuck::cast_slice(&[[-1_f32, -1., 2.], [0., 1., 2.], [1., -1., 2.]]),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    let mut scene = device.create_tlas(&wgpu::CreateTlasDescriptor {
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
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        [&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [&scene],
    );
    queue.submit([encoder.finish()]);
    // SAFETY: The guards borrow this device; all resources and pipeline live through the final poll.
    let pipeline = unsafe {
        let native = device.as_hal::<wgpu::hal::api::Dx12>().unwrap();
        let parameters = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let mut blob = None;
        D3D12SerializeRootSignature(
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: parameters.len() as u32,
                pParameters: parameters.as_ptr(),
                ..Default::default()
            },
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            None,
        )
        .unwrap();
        let blob = blob.unwrap();
        let root: ID3D12RootSignature = native
            .raw_device()
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()),
            )
            .unwrap();
        Pipeline::new(
            native.raw_device(),
            &root,
            &dxil,
            Exports {
                raygen: "RayGen",
                miss: &["Miss"],
                hit: &[HitGroup {
                    name: "TriangleHit",
                    closest_hit: "ClosestHit",
                    any_hit: None,
                }],
            },
            16,
            1,
        )
        .unwrap()
    };
    let table = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("native SBT"),
        contents: pipeline.shader_records(),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Mark native output initialized in wgpu before raw writes; later copies must not lazily clear it.
    let output = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("native RT output"),
        contents: &[0_u8; 8],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for (remove, expected) in [
        (false, [0x484954, 0x4d495353]),
        (true, [0x4d495353, 0x4d495353]),
    ] {
        eprintln!("Native interop round: remove={remove}");
        if remove {
            scene[0] = None;
            let mut update = device.create_command_encoder(&Default::default());
            update.build_acceleration_structures([], [&scene]);
            eprintln!("Submitting removed-instance AS rebuild");
            queue.submit([update.finish()]);
            eprintln!("Removed-instance rebuild submitted");
        }
        let mut transitions = device.create_command_encoder(&Default::default());
        transitions.transition_resources(
            [
                wgpu::BufferTransition {
                    buffer: &table,
                    state: wgpu::BufferUses::STORAGE_READ_ONLY,
                },
                wgpu::BufferTransition {
                    buffer: &output,
                    state: wgpu::BufferUses::STORAGE_READ_WRITE,
                },
            ]
            .into_iter(),
            std::iter::empty(),
        );
        let mut native_encoder = device.create_command_encoder(&Default::default());
        unsafe {
            // Each as_hal guard owns a non-reentrant snatch lock. Keep only scalar addresses.
            // The wgpu owners stay alive until completion of the final submission.
            let table_address = {
                let native = table.as_hal::<wgpu::hal::api::Dx12>().unwrap();
                native.raw_resource().GetGPUVirtualAddress()
            };
            let output_address = {
                let native = output.as_hal::<wgpu::hal::api::Dx12>().unwrap();
                native.raw_resource().GetGPUVirtualAddress()
            };
            let scene_address = {
                let native = scene.as_hal::<wgpu::hal::api::Dx12>().unwrap();
                native.raw_resource().GetGPUVirtualAddress()
            };
            native_encoder.as_hal_mut::<wgpu::hal::api::Dx12, _, _>(|encoder| {
                let list = encoder
                    .unwrap()
                    .raw_command_list()
                    .unwrap()
                    .cast::<ID3D12GraphicsCommandList4>()
                    .unwrap();
                let barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                    ..Default::default()
                };
                list.ResourceBarrier(std::slice::from_ref(&barrier));
                pipeline.bind(&list);
                list.SetComputeRootUnorderedAccessView(0, output_address);
                list.SetComputeRootShaderResourceView(1, scene_address);
                pipeline.dispatch(&list, table_address, [2, 1, 1]).unwrap();
            });
        }
        // Native state is confined to its encoder. Copying uses a fresh tracked wgpu encoder.
        let mut copy = device.create_command_encoder(&Default::default());
        copy.copy_buffer_to_buffer(&output, 0, &readback, 0, 8);
        eprintln!("Submitting native trace and copy");
        queue.submit([transitions.finish(), native_encoder.finish(), copy.finish()]);
        eprintln!("Native trace submitted; polling completion");
        let (send, receive) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap()
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        eprintln!("Native trace poll completed");
        receive.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        eprintln!(
            "Native trace result: {:?}",
            bytemuck::cast_slice::<u8, u32>(&mapped)
        );
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), &expected);
        drop(mapped);
        readback.unmap();
    }
    assert!(pollster::block_on(validation.pop()).is_none());
}
