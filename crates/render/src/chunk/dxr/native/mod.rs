//! Native DXR intersection passes with tracked wgpu material evaluation and compositing.
use super::*;
use dxr_pipeline::native::Pipeline;
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

mod dispatch;
mod programs;
pub(super) mod record;
#[cfg(test)]
mod tests;

#[cfg(native_dxr_shader)]
const DXIL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native-terrain.dxil"));
#[cfg(not(native_dxr_shader))]
const DXIL: &[u8] = &[];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    frame: Frame,
    grid: [u32; 4],
    fog: [f32; 4],
    sky: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<Params>() == 192);

struct Programs {
    rays: [Pipeline; 3],
    tables: [wgpu::Buffer; 3],
    receivers: wgpu::ComputePipeline,
    normals: wgpu::ComputePipeline,
    albedo: wgpu::RenderPipeline,
    lighting: wgpu::RenderPipeline,
    fog: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

struct Buffers {
    size: [u32; 2],
    receivers: wgpu::Buffer,
    hits: wgpu::Buffer,
    normals: wgpu::Buffer,
    lighting: wgpu::Buffer,
    fog: wgpu::Buffer,
}

struct KeepAlive {
    _programs: Arc<Programs>,
    _scene: Arc<wgpu::Tlas>,
    _buffers: Arc<Buffers>,
    _uniform: wgpu::Buffer,
    _lights: wgpu::Buffer,
}

type ProgramResult = std::result::Result<Arc<Programs>, String>;
#[derive(Default)]
enum Loading {
    #[default]
    Idle,
    Pending(std::sync::mpsc::Receiver<ProgramResult>),
    Ready(Arc<Programs>),
    Failed,
}

#[derive(Resource, Default)]
pub(super) struct NativeTerrain {
    programs: Mutex<HashMap<(wgpu::TextureFormat, u32), Loading>>,
    buffers: Mutex<Option<Arc<Buffers>>>,
    failed_size: Mutex<Option<[u32; 2]>>,
    encoded: Mutex<Option<Entity>>,
    pending: Mutex<Vec<KeepAlive>>,
    reported: AtomicBool,
}

/// Capture the immutable TLAS address while its new owner is still mutable.
/// Existing public wgpu signatures stay unchanged; graph readers retain its Arc owner.
pub(super) fn address(scene: &mut wgpu::Tlas) -> u64 {
    // SAFETY: The short HAL guard borrows this live DX12 TLAS and only copies its address.
    unsafe {
        scene
            .as_hal::<wgpu::hal::api::Dx12>()
            .map_or(0, |value| value.raw_resource().GetGPUVirtualAddress())
    }
}

pub(super) fn encoded(world: &World, entity: Entity) -> bool {
    world.get_resource::<NativeTerrain>().is_some_and(|state| {
        *state
            .encoded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            == Some(entity)
    })
}

/// Reset view publication before this frame's graph runs.
pub(super) fn reset(
    state: Res<NativeTerrain>,
    preference: Option<Res<render_scene::terrain_shadows::DxrTerrainShadows>>,
) {
    if preference.is_none_or(|value| !value.0) {
        *state
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        *state
            .failed_size
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
    *state
        .encoded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Register retention after render_system has submitted all graph command buffers.
pub(super) fn retain(state: Res<NativeTerrain>, queue: Res<RenderQueue>) {
    let values = std::mem::take(
        &mut *state
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    if !values.is_empty() {
        queue.on_submitted_work_done(move || drop(values));
    }
}

impl NativeTerrain {
    fn programs(
        &self,
        device: &RenderDevice,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> Option<Arc<Programs>> {
        if DXIL.is_empty() {
            return None;
        }
        let mut cache = self
            .programs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let state = cache.entry((format, samples)).or_default();
        if matches!(state, Loading::Idle) {
            let gpu = device.wgpu_device().clone();
            let (send, receive) = std::sync::mpsc::channel();
            *state = Loading::Pending(receive);
            std::thread::spawn(move || {
                let _ = send.send(programs::create(&gpu, format, samples).map(Arc::new));
            });
        }
        if let Loading::Pending(receive) = state {
            match receive.try_recv() {
                Ok(Ok(programs)) => *state = Loading::Ready(programs),
                Ok(Err(error)) => {
                    bevy::log::error!(%error,"Native DXR unavailable; retaining compatibility lighting");
                    *state = Loading::Failed;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    bevy::log::error!("Native DXR pipeline worker failed");
                    *state = Loading::Failed;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Loading::Ready(programs) = state {
            Some(programs.clone())
        } else {
            None
        }
    }
}

fn shader_source() -> String {
    surfaces::material_source(include_str!("deferred.wgsl"))
}

/// Shrink expensive intermediate buffers before exceeding the device's binding limit.
fn grid(mut size: [u32; 2], limit: u64) -> Option<[u32; 2]> {
    if size.contains(&0) || limit < 48 {
        return None;
    }
    while u64::from(size[0])
        .checked_mul(u64::from(size[1]))?
        .checked_mul(48)?
        > limit
    {
        size = size.map(|axis| axis.div_ceil(2));
    }
    Some(size)
}

#[cfg(test)]
mod shader_tests {
    use super::*;
    #[test]
    fn large_views_stay_within_storage_binding_budget() {
        assert_eq!(grid([1920, 1080], 128 * 1024 * 1024), Some([1920, 1080]));
        assert_eq!(grid([3840, 2160], 128 * 1024 * 1024), Some([1920, 1080]));
        assert_eq!(grid([0, 1080], 128 * 1024 * 1024), None);
        assert_eq!(grid([1, 1], 47), None);
    }
    #[test]
    fn deferred_material_shader_validates_without_ray_queries() {
        let module = naga::front::wgsl::parse_str(&shader_source()).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        for (index, entry) in module
            .entry_points
            .iter()
            .enumerate()
            .filter(|(_, e)| e.name != "vertex")
        {
            let usage = info.get_entry_point(index);
            for group in 0..3 {
                let mut actual: Vec<_> = module
                    .global_variables
                    .iter()
                    .filter_map(|(handle, global)| {
                        global
                            .binding
                            .as_ref()
                            .filter(|b| b.group == group as u32 && !usage[handle].is_empty())
                            .map(|b| b.binding)
                    })
                    .collect();
                actual.sort_unstable();
                assert_eq!(
                    actual,
                    programs::bindings(&entry.name, group),
                    "{} group {group}",
                    entry.name
                );
            }
        }
    }
}
