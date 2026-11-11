//! Pipeline creation runs off the render thread; no public wgpu API is changed.
use super::*;
use dxr_pipeline::native::{Exports, HitGroup};

pub(super) fn create(
    gpu: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> std::result::Result<Programs, String> {
    native(gpu, format, samples).map_err(|error| error.to_string())
}

fn native(
    gpu: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> windows::core::Result<Programs> {
    // SAFETY: Clone the COM reference before releasing the single device HAL guard.
    let device = unsafe {
        gpu.as_hal::<wgpu::hal::api::Dx12>()
            .expect("DX12 device")
            .raw_device()
            .clone()
    };
    let parameters: Vec<_> = (0..8)
        .map(|slot| D3D12_ROOT_PARAMETER {
            ParameterType: match slot {
                0 => D3D12_ROOT_PARAMETER_TYPE_CBV,
                1..=4 => D3D12_ROOT_PARAMETER_TYPE_SRV,
                _ => D3D12_ROOT_PARAMETER_TYPE_UAV,
            },
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: match slot {
                        0 => 0,
                        1..=4 => slot - 1,
                        _ => slot - 5,
                    },
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        })
        .collect();
    // SAFETY: Parameters and bytecode remain live for synchronous native pipeline creation.
    let primary = unsafe {
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
        )?;
        let blob = blob.expect("serialized root signature");
        let root: ID3D12RootSignature = device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()),
        )?;
        Pipeline::new(
            &device,
            &root,
            DXIL,
            Exports {
                raygen: "Primary",
                miss: &["Miss"],
                hit: &[HitGroup {
                    name: "TerrainHit",
                    closest_hit: "ClosestHit",
                    any_hit: Some("FilterWithdrawn"),
                }],
            },
            16,
            1,
        )?
    };
    let rays = [
        primary.for_raygen("Primary")?,
        primary.for_raygen("Secondary")?,
        primary.for_raygen("Beams")?,
    ];
    let tables = std::array::from_fn(|index| {
        gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Native DXR shader table"),
            contents: rays[index].shader_records(),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Native DXR deferred materials"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });
    let compute = |entry| {
        gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let render = |entry, multiply| {
        gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Native DXR deferred composite"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    write_mask: wgpu::ColorWrites::COLOR,
                    blend: Some(if multiply {
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
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        })
    };
    Ok(Programs {
        rays,
        tables,
        receivers: compute(if samples > 1 {
            "receivers_msaa"
        } else {
            "receivers_single"
        }),
        normals: compute("mapped_normals"),
        albedo: render("albedo", true),
        lighting: render("lighting", false),
        fog: render("fog_light", false),
        sampler: gpu.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Native DXR PBR sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }),
    })
}

/// Entry-specific groups keep the deferred passes within eight storage buffers per stage.
/// Reflection tests fail if the shader's resource contract changes without this binding recipe.
pub(super) fn bindings(entry: &str, group: usize) -> Vec<u32> {
    match (entry, group) {
        ("receivers_single", 0) => vec![1, 2],
        ("receivers_msaa", 0) => vec![1, 3],
        ("receivers_single" | "receivers_msaa", 1) => vec![0],
        ("receivers_single" | "receivers_msaa", 2) => vec![0],
        ("mapped_normals" | "albedo", 0) => {
            (1..=21).filter(|id| !matches!(id, 2 | 3 | 8)).collect()
        }
        ("lighting", 0) => (1..=21).filter(|id| !matches!(id, 2 | 3)).collect(),
        ("mapped_normals", 1) => vec![0, 1, 2],
        ("mapped_normals", 2) => vec![1],
        ("albedo" | "lighting", 1) => vec![0, 1, 2, 4],
        ("fog_light", 0) => vec![1],
        ("fog_light", 1) => vec![0, 1, 5],
        _ => vec![],
    }
}
