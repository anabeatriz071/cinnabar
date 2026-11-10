/// Uses the scene's existing color and sample count, with a read-only depth input.
pub(super) fn create_pipeline(
    gpu: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let shader = gpu.create_shader_module(wgpu::include_wgsl!("../dxr.wgsl"));
    gpu.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("DXR terrain sunlight"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(if samples > 1 {
                "multisampled"
            } else {
                "single"
            }),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::COLOR,
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
}
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct Frame {
    pub(super) world_from_clip: [[f32; 4]; 4],
    pub(super) eye_radius: [f32; 4],
    pub(super) sun_strength: [f32; 4],
    pub(super) viewport: [f32; 4],
    pub(super) ray_length: [f32; 4],
}
