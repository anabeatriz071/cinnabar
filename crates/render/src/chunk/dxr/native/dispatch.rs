//! Native command lists remain isolated from wgpu encoders and share its queue ordering.
use super::*;

/// Return a scalar address while holding only one non-reentrant HAL guard at a time.
fn address(buffer: &wgpu::Buffer) -> u64 {
    // SAFETY: Owners are retained through submission completion; the short guard cannot overlap another.
    unsafe {
        buffer
            .as_hal::<wgpu::hal::api::Dx12>()
            .expect("DX12 buffer")
            .raw_resource()
            .GetGPUVirtualAddress()
    }
}

pub(super) fn encode(
    gpu: &wgpu::Device,
    programs: &Programs,
    phase: usize,
    scene: &wgpu::Tlas,
    uniform: &wgpu::Buffer,
    lights: &wgpu::Buffer,
    buffers: &Buffers,
) -> windows::core::Result<[wgpu::CommandBuffer; 2]> {
    let mut transitions = gpu.create_command_encoder(&Default::default());
    transitions.transition_resources(
        [
            (uniform, wgpu::BufferUses::UNIFORM),
            (lights, wgpu::BufferUses::STORAGE_READ_ONLY),
            (&programs.tables[phase], wgpu::BufferUses::STORAGE_READ_ONLY),
            (&buffers.receivers, wgpu::BufferUses::STORAGE_READ_ONLY),
            (&buffers.normals, wgpu::BufferUses::STORAGE_READ_ONLY),
            (&buffers.hits, wgpu::BufferUses::STORAGE_READ_WRITE),
            (&buffers.lighting, wgpu::BufferUses::STORAGE_READ_WRITE),
            (&buffers.fog, wgpu::BufferUses::STORAGE_READ_WRITE),
        ]
        .into_iter()
        .map(|(buffer, state)| wgpu::BufferTransition { buffer, state }),
        std::iter::empty(),
    );
    let roots = [
        address(uniform),
        // SAFETY: Same native device and scene owner retained until submitted work completes.
        unsafe {
            scene
                .as_hal::<wgpu::hal::api::Dx12>()
                .expect("DX12 TLAS")
                .raw_resource()
                .GetGPUVirtualAddress()
        },
        address(&buffers.receivers),
        address(&buffers.normals),
        address(lights),
        address(&buffers.hits),
        address(&buffers.lighting),
        address(&buffers.fog),
    ];
    let table = address(&programs.tables[phase]);
    let size = if phase == 2 {
        buffers.size.map(|x| x.div_ceil(2))
    } else {
        buffers.size
    };
    let mut encoder = gpu.create_command_encoder(&Default::default());
    // SAFETY: Dedicated encoder, explicit tracked resource transitions, all owners retained by caller.
    unsafe {
        encoder.as_hal_mut::<wgpu::hal::api::Dx12, _, _>(|native| {
            let list = native
                .expect("DX12 encoder")
                .raw_command_list()
                .expect("open list")
                .cast::<ID3D12GraphicsCommandList4>()?;
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
                ..Default::default()
            };
            list.ResourceBarrier(&[barrier]);
            programs.rays[phase].bind(&list);
            list.SetComputeRootConstantBufferView(0, roots[0]);
            for slot in 1..5 {
                list.SetComputeRootShaderResourceView(slot as u32, roots[slot]);
            }
            for slot in 5..8 {
                list.SetComputeRootUnorderedAccessView(slot as u32, roots[slot]);
            }
            programs.rays[phase].dispatch(&list, table, [size[0], size[1], 1])
        })?;
    }
    Ok([transitions.finish(), encoder.finish()])
}

impl Buffers {
    pub(super) fn new(device: &RenderDevice, size: [u32; 2]) -> Option<Self> {
        let pixels = u64::from(size[0]) * u64::from(size[1]);
        let create = |label: &'static str, bytes| {
            crate::gpu_allocation::checked(device, label, bytes, || {
                device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
        };
        Some(Self {
            size,
            receivers: create("Native DXR receiver positions", pixels * 16)?,
            hits: create("Native DXR primary hits", pixels * 16)?,
            normals: create("Native DXR mapped normals", pixels * 32)?,
            lighting: create("Native DXR lighting and reflections", pixels * 48)?,
            fog: create(
                "Native DXR fog scattering",
                u64::from(size[0].div_ceil(2)) * u64::from(size[1].div_ceil(2)) * 16,
            )?,
        })
    }
    /// Explicit initialization prevents tracked wgpu passes from clearing native shader output later.
    pub(super) fn initialize(&self, encoder: &mut wgpu::CommandEncoder) {
        for buffer in [
            &self.receivers,
            &self.hits,
            &self.normals,
            &self.lighting,
            &self.fog,
        ] {
            encoder.clear_buffer(buffer, 0, None);
        }
    }
}
