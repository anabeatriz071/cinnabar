//! Bounded direct-light approximation from active-pack face-average MER emission.
use super::*;
use assets::{BlockFace, ResolvedBlock, RuntimeAssets, TextureArray};
use render_scene::EnhancedTextureAssets;

/// Face emitters sit outside their own cube, avoiding self-occlusion by the opaque BLAS.
/// This uses the final mip's average, not per-pixel path transport or indirect bounce light.
pub(super) fn templates(
    assets: &RuntimeAssets,
    maps: &EnhancedTextureAssets,
    block: ResolvedBlock,
) -> Vec<DxrLocalLight> {
    BlockFace::ALL
        .into_iter()
        .filter_map(|face| {
            let material = block.face(face).material_id();
            let reference = maps.material_reference(material, assets.material(material).texture);
            if reference == u32::MAX || reference & assets::texture::PBR_REF_MATERIAL == 0 {
                return None;
            }
            let page = (reference >> 31) as usize;
            let layer = (reference & (assets::texture::PBR_REF_COLOR - 1)) as usize;
            let mer = tail(&maps.mer_pages[page], layer)?;
            let emission = f32::from(mer[1]) / 255.0;
            if emission <= 1.0 / 255.0 {
                return None;
            }
            let colour = if reference & assets::texture::PBR_REF_COLOR != 0 {
                let pixel = tail(&maps.color_pages[page], layer)?;
                std::array::from_fn(|axis| srgb(pixel[axis]))
            } else {
                let texture = assets.material(material).texture;
                let carrier = assets.texture_shards().nth(texture.page() as usize)?;
                let pixel = tail(carrier, texture.layer() as usize)?;
                std::array::from_fn(|axis| srgb(pixel[axis]))
            };
            let mut position = [0.0; 3];
            // Bedrock face discriminants encode negative/positive pairs in X,Y,Z order.
            let axis = face as usize / 2;
            position[axis] = if face as usize % 2 == 0 { -0.51 } else { 0.51 };
            Some(DxrLocalLight {
                position,
                radius: f32::from(block.light_properties().emission().max(8)),
                colour,
                intensity: emission * 0.8,
            })
        })
        .collect()
}

fn tail(page: &TextureArray, layer: usize) -> Option<[u8; 4]> {
    let mip = page.mips.last()?;
    if mip.size != 1 {
        return None;
    }
    mip.rgba8
        .get(layer.checked_mul(4)?..layer.checked_add(1)?.checked_mul(4)?)?
        .try_into()
        .ok()
}
fn srgb(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Appends face emitters for changed palettes. Resolution work is shared across blocks and frames.
pub(super) fn scan(
    key: world::SubChunkKey,
    data: &world::SubChunk,
    mut resolve: impl FnMut(u32) -> Vec<DxrLocalLight>,
) -> Vec<DxrLocalLight> {
    let mut properties = HashMap::new();
    let mut lights = Vec::new();
    let side = world::SUB_CHUNK_SIDE as u8;
    for y in 0..side {
        for z in 0..side {
            for x in 0..side {
                let Some(id) = data.runtime_id(0, x, y, z) else {
                    continue;
                };
                let values = properties.entry(id).or_insert_with(|| resolve(id));
                let center = [
                    key.x as f32 * f32::from(side) + f32::from(x) + 0.5,
                    key.y as f32 * f32::from(side) + f32::from(y) + 0.5,
                    key.z as f32 * f32::from(side) + f32::from(z) + 0.5,
                ];
                lights.extend(values.iter().map(|light| DxrLocalLight {
                    position: std::array::from_fn(|axis| center[axis] + light.position[axis]),
                    ..*light
                }));
            }
        }
    }
    // Retain a bounded representative set for dense emissive packs. This provisional
    // direct-light budget is not Bedrock's per-pixel path-transport behavior.
    if lights.len() > MAX_DXR_LOCAL_LIGHTS {
        let center = Vec3::new(
            (key.x as f32 + 0.5) * f32::from(side),
            (key.y as f32 + 0.5) * f32::from(side),
            (key.z as f32 + 0.5) * f32::from(side),
        );
        lights.select_nth_unstable_by(MAX_DXR_LOCAL_LIGHTS, |a, b| {
            Vec3::from(a.position)
                .distance_squared(center)
                .total_cmp(&Vec3::from(b.position).distance_squared(center))
        });
        lights.truncate(MAX_DXR_LOCAL_LIGHTS);
    }
    lights
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dxr_authored_faces_emit_outside_the_block_and_clear_on_removal() {
        let key = world::SubChunkKey::new(0, 0, 0, 0);
        let mut blocks = [0; world::SUB_CHUNK_SIDE.pow(3)];
        blocks[0] = 1;
        let resolve = |id| {
            if id == 1 {
                vec![DxrLocalLight {
                    position: [0.51, 0., 0.],
                    radius: 8.,
                    colour: [1., 0., 0.],
                    intensity: 0.8,
                }]
            } else {
                vec![]
            }
        };
        let lights = scan(key, &world::SubChunk::from_runtime_ids(&blocks), resolve);
        assert_eq!(lights.len(), 1);
        assert!(lights[0].position[0] > 1.0);
        blocks[0] = 0;
        assert!(scan(key, &world::SubChunk::from_runtime_ids(&blocks), resolve).is_empty());
    }
    #[test]
    fn dxr_material_average_does_not_read_outside_the_layer() {
        let page = TextureArray {
            layers: 1,
            mips: vec![assets::TextureMip {
                size: 1,
                rgba8: vec![0, 128, 255, 255].into_boxed_slice(),
            }]
            .into(),
        };
        assert_eq!(tail(&page, 0), Some([0, 128, 255, 255]));
        assert_eq!(tail(&page, 1), None);
        assert_eq!(srgb(0), 0.);
        assert_eq!(srgb(255), 1.);
    }
}
