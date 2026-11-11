//! Native state objects use the caller's DX12 device; no wgpu API or global state is changed.
use crate::{IDENTIFIER_BYTES, ShaderTableLayout, TABLE_ALIGNMENT, Table};
use std::{ffi::c_void, mem::ManuallyDrop};
use windows::{
    Win32::Graphics::Direct3D12::*,
    core::{Error, Interface, PCWSTR, Result},
};

/// One triangle hit group. Cutout geometry may supply an any-hit shader.
pub struct HitGroup<'a> {
    pub name: &'a str,
    pub closest_hit: &'a str,
    pub any_hit: Option<&'a str>,
}

/// Exports from one compiled DXIL library, in shader-table order.
pub struct Exports<'a> {
    pub raygen: &'a str,
    pub miss: &'a [&'a str],
    pub hit: &'a [HitGroup<'a>],
}

/// Owns the native pipeline and its identifier bytes. GPU shader-table storage belongs to the caller.
pub struct Pipeline {
    state: ID3D12StateObject,
    root: ID3D12RootSignature,
    records: Box<[u8]>,
    layout: ShaderTableLayout,
}

fn invalid(message: &str) -> Error {
    Error::new(windows::core::HRESULT(0x80070057_u32 as i32), message)
}
fn wide(name: &str) -> Result<Vec<u16>> {
    if name.is_empty() || name.contains('\0') {
        return Err(invalid("empty or NUL-containing DXR export"));
    }
    Ok(name.encode_utf16().chain([0]).collect())
}
fn object<T>(kind: D3D12_STATE_SUBOBJECT_TYPE, value: &T) -> D3D12_STATE_SUBOBJECT {
    D3D12_STATE_SUBOBJECT {
        Type: kind,
        pDesc: std::ptr::from_ref(value).cast::<c_void>(),
    }
}

impl Pipeline {
    /// Builds a state object from DXIL. Native errors propagate; unsupported devices never panic.
    /// # Safety
    /// The root signature must belong to this device and match a valid DXC-compiled library.
    /// The caller must comply with DX12 shader/root-signature validation requirements.
    pub unsafe fn new(
        device: &ID3D12Device,
        root: &ID3D12RootSignature,
        dxil: &[u8],
        exports: Exports<'_>,
        payload_bytes: u32,
        recursion: u32,
    ) -> Result<Self> {
        if dxil.is_empty()
            || !(1..=D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH).contains(&recursion)
        {
            return Err(invalid("invalid DXIL library or recursion depth"));
        }
        let miss_count =
            u32::try_from(exports.miss.len()).map_err(|_| invalid("too many miss exports"))?;
        let hit_count =
            u32::try_from(exports.hit.len()).map_err(|_| invalid("too many hit groups"))?;
        let object_count = hit_count
            .checked_add(4)
            .ok_or_else(|| invalid("too many state subobjects"))?;
        let layout = ShaderTableLayout::new(0, miss_count, hit_count)
            .ok_or_else(|| invalid("shader table overflow"))?;
        let raygen = wide(exports.raygen)?;
        let misses = exports
            .miss
            .iter()
            .map(|name| wide(name))
            .collect::<Result<Vec<_>>>()?;
        let hits = exports
            .hit
            .iter()
            .map(|hit| {
                Ok((
                    wide(hit.name)?,
                    wide(hit.closest_hit)?,
                    hit.any_hit.map(wide).transpose()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let hit_desc: Vec<_> = hits
            .iter()
            .map(|(name, closest, any)| D3D12_HIT_GROUP_DESC {
                HitGroupExport: PCWSTR(name.as_ptr()),
                Type: D3D12_HIT_GROUP_TYPE_TRIANGLES,
                ClosestHitShaderImport: PCWSTR(closest.as_ptr()),
                AnyHitShaderImport: any
                    .as_ref()
                    .map_or(PCWSTR::null(), |value| PCWSTR(value.as_ptr())),
                IntersectionShaderImport: PCWSTR::null(),
            })
            .collect();
        let library = D3D12_DXIL_LIBRARY_DESC {
            DXILLibrary: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            },
            ..Default::default()
        };
        let mut global = D3D12_GLOBAL_ROOT_SIGNATURE {
            pGlobalRootSignature: ManuallyDrop::new(Some(root.clone())),
        };
        let shader = D3D12_RAYTRACING_SHADER_CONFIG {
            MaxPayloadSizeInBytes: payload_bytes,
            MaxAttributeSizeInBytes: 8,
        };
        let config = D3D12_RAYTRACING_PIPELINE_CONFIG {
            MaxTraceRecursionDepth: recursion,
        };
        let mut objects = vec![
            object(D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library),
            object(D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global),
            object(D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader),
            object(
                D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
                &config,
            ),
        ];
        objects.extend(
            hit_desc
                .iter()
                .map(|hit| object(D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, hit)),
        );
        let descriptor = D3D12_STATE_OBJECT_DESC {
            Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
            NumSubobjects: object_count,
            pSubobjects: objects.as_ptr(),
        };
        let result = device.cast::<ID3D12Device5>().and_then(|device| {
            // SAFETY: All subobject and DXIL allocations remain live for this synchronous call.
            unsafe { device.CreateStateObject::<ID3D12StateObject>(&descriptor) }
        });
        // SAFETY: This local COM reference was initialized exactly once and has no automatic drop.
        unsafe {
            ManuallyDrop::drop(&mut global.pGlobalRootSignature);
        }
        let state = result?;
        let properties = state.cast::<ID3D12StateObjectProperties>()?;
        let mut records =
            vec![0; usize::try_from(layout.bytes).map_err(|_| invalid("shader table too large"))?];
        let names = std::iter::once((layout.raygen, vec![raygen])).chain([
            (layout.miss, misses),
            (
                layout.hit,
                hits.into_iter().map(|(name, _, _)| name).collect(),
            ),
        ]);
        for (table, names) in names {
            for (index, name) in names.iter().enumerate() {
                // SAFETY: The UTF-16 name is terminated and the pipeline owns its identifier memory.
                let identifier = unsafe { properties.GetShaderIdentifier(PCWSTR(name.as_ptr())) };
                if identifier.is_null() {
                    return Err(invalid("DXR export has no shader identifier"));
                }
                let offset = (table.offset + index as u64 * table.stride) as usize;
                // SAFETY: Native shader identifiers are exactly IDENTIFIER_BYTES bytes; bounds are checked by the layout.
                let bytes = unsafe {
                    std::slice::from_raw_parts(identifier.cast::<u8>(), IDENTIFIER_BYTES as usize)
                };
                records[offset..offset + bytes.len()].copy_from_slice(bytes);
            }
        }
        Ok(Self {
            state,
            root: root.clone(),
            records: records.into(),
            layout,
        })
    }

    /// Selects another ray-generation export from this state object without recompilation.
    pub fn for_raygen(&self, name: &str) -> Result<Self> {
        let name = wide(name)?;
        let properties = self.state.cast::<ID3D12StateObjectProperties>()?;
        // SAFETY: The terminated name and state object remain live for this identifier lookup.
        let identifier = unsafe { properties.GetShaderIdentifier(PCWSTR(name.as_ptr())) };
        if identifier.is_null() {
            return Err(invalid("unknown ray-generation export"));
        }
        let mut records = self.records.clone();
        // SAFETY: DXR identifiers have the SDK-checked size; records always hold one raygen entry.
        records[..IDENTIFIER_BYTES as usize].copy_from_slice(unsafe {
            std::slice::from_raw_parts(identifier.cast(), IDENTIFIER_BYTES as usize)
        });
        Ok(Self {
            state: self.state.clone(),
            root: self.root.clone(),
            records,
            layout: self.layout,
        })
    }

    pub fn layout(&self) -> ShaderTableLayout {
        self.layout
    }
    pub fn shader_records(&self) -> &[u8] {
        &self.records
    }

    /// Selects this pipeline before the caller binds its root arguments.
    /// # Safety
    /// The list must be open on this pipeline's device. Keep this pipeline alive until GPU completion.
    pub unsafe fn bind(&self, list: &ID3D12GraphicsCommandList4) {
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState1(&self.state);
        }
    }

    /// Records DispatchRays after bind/root arguments and resource transitions.
    /// # Safety
    /// `table_address` must name GPU-readable storage containing shader_records(), on the same device.
    /// Retain it, the pipeline and every referenced resource until the submitted fence completes.
    /// The caller owns resource-state transitions and must not interleave untracked wgpu commands.
    pub unsafe fn dispatch(
        &self,
        list: &ID3D12GraphicsCommandList4,
        table_address: u64,
        extent: [u32; 3],
    ) -> Result<()> {
        if table_address == 0
            || table_address % TABLE_ALIGNMENT != 0
            || !ShaderTableLayout::valid_dispatch(extent)
            || table_address.checked_add(self.layout.bytes).is_none()
        {
            return Err(invalid("invalid shader table address or dispatch extent"));
        }
        let table = |value: Table| D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
            StartAddress: if value.size == 0 {
                0
            } else {
                table_address + value.offset
            },
            SizeInBytes: value.size,
            StrideInBytes: if value.size == 0 { 0 } else { value.stride },
        };
        let desc = D3D12_DISPATCH_RAYS_DESC {
            RayGenerationShaderRecord: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
                StartAddress: table_address,
                SizeInBytes: self.layout.raygen.size,
            },
            MissShaderTable: table(self.layout.miss),
            HitGroupTable: table(self.layout.hit),
            CallableShaderTable: Default::default(),
            Width: extent[0],
            Height: extent[1],
            Depth: extent[2],
        };
        // SAFETY: The caller supplies the native command/resource lifetime and state guarantees above.
        unsafe {
            list.DispatchRays(&desc);
        }
        Ok(())
    }
}

const _: () = assert!(IDENTIFIER_BYTES == D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES as u64);
const _: () =
    assert!(crate::RECORD_ALIGNMENT == D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT as u64);
const _: () = assert!(TABLE_ALIGNMENT == D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT as u64);
