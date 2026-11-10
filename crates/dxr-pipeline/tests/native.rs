#![cfg(windows)]
use dxr_pipeline::native::{Exports, HitGroup, Pipeline};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0, Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC,
        },
        System::Threading::{CreateEventW, WaitForSingleObject},
    },
    core::Interface,
};

struct Event(HANDLE);
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// Allocates a buffer with one owner retained through the submitted fence.
unsafe fn buffer(
    device: &ID3D12Device,
    size: u64,
    heap: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
    uav: bool,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut resource = None;
    unsafe {
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    CreationNodeMask: 1,
                    VisibleNodeMask: 1,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut resource,
            )
            .unwrap();
    }
    resource.unwrap()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "names optional hardware and compiled shader fixtures"
)]
fn dxr_native_pipeline_dispatch_writes_and_reads_back_a_marker() {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.local/dxr/native-smoke.dxil");
    let Ok(dxil) = std::fs::read(fixture) else {
        eprintln!(
            "missing fixture: compiled native-smoke.dxil (packaging/windows/compile-rt-smoke.ps1)"
        );
        return;
    };
    // SAFETY: This isolated test owns its native device, queue, descriptors and GPU resources.
    unsafe {
        let mut device: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
            eprintln!("missing fixture: DX12 device for native RT pipeline");
            return;
        }
        let device = device.unwrap();
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                std::ptr::from_mut(&mut options).cast(),
                std::mem::size_of_val(&options) as u32,
            )
            .unwrap();
        if options.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED {
            eprintln!("missing fixture: DXR GPU for native RT pipeline dispatch");
            return;
        }
        let parameter = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let mut blob = None;
        D3D12SerializeRootSignature(
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &parameter,
                ..Default::default()
            },
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            None,
        )
        .unwrap();
        let blob = blob.unwrap();
        let root: ID3D12RootSignature = device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()),
            )
            .unwrap();
        let pipeline = Pipeline::new(
            &device,
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
        .unwrap();
        let table = buffer(
            &device,
            pipeline.layout().bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            false,
        );
        let mut address = std::ptr::null_mut();
        table.Map(0, None, Some(&mut address)).unwrap();
        std::ptr::copy_nonoverlapping(
            pipeline.shader_records().as_ptr(),
            address.cast(),
            pipeline.shader_records().len(),
        );
        table.Unmap(0, None);
        let output = buffer(
            &device,
            4,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            true,
        );
        let readback = buffer(
            &device,
            4,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            false,
        );
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList4 = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        pipeline.bind(&list);
        list.SetComputeRootUnorderedAccessView(0, output.GetGPUVirtualAddress());
        assert!(
            pipeline
                .dispatch(&list, table.GetGPUVirtualAddress() + 1, [1, 1, 1])
                .is_err()
        );
        pipeline
            .dispatch(&list, table.GetGPUVirtualAddress(), [1, 1, 1])
            .unwrap();
        let transition = D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: std::mem::ManuallyDrop::new(Some(output.clone())),
            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
        };
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(transition),
            },
            ..Default::default()
        };
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        // The descriptor union does not automatically release its cloned COM reference.
        std::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
        list.CopyResource(&readback, &output);
        list.Close().unwrap();
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let event = Event(CreateEventW(None, false, false, None).unwrap());
        fence.SetEventOnCompletion(1, event.0).unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&fence, 1).unwrap();
        assert_eq!(
            WaitForSingleObject(event.0, 10000),
            WAIT_OBJECT_0,
            "native dispatch did not complete"
        );
        let mut mapped = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let marker = mapped.cast::<u32>().read_unaligned();
        readback.Unmap(0, None);
        assert_eq!(marker, 0x43585254);
    }
}
