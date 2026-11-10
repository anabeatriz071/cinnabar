use wgpu::util::DeviceExt;

/// Uploads one opaque triangle mesh and records its BLAS build in the scene encoder.
pub(super) fn build_opaque_mesh(
    gpu: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    positions: &[[f32; 3]],
    indices: &[u32],
) -> wgpu::Blas {
    let upload = |label, contents| {
        gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::BLAS_INPUT,
        })
    };
    let vertex = upload("DXR chunk vertices", bytemuck::cast_slice(positions));
    let index = upload("DXR chunk indices", bytemuck::cast_slice(indices));
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: positions.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = gpu.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("DXR opaque chunk"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    encoder.build_acceleration_structures(
        [&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vertex,
                first_vertex: 0,
                vertex_stride: size_of::<[f32; 3]>() as u64,
                index_buffer: Some(&index),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [],
    );
    blas
}
