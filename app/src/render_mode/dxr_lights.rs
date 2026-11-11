//! Cached resident emitter discovery; this never changes authoritative block light.
use crate::runtime::world::ClientWorld;
use bevy::prelude::*;
use client_presentation::camera::FlyCamera;
use client_shell::ui_resources::UiRuntimeResource;
use render_scene::terrain_shadows::{
    DxrLocalLight, DxrLocalLights, DxrTerrainShadows, MAX_DXR_LOCAL_LIGHTS,
};
use std::{collections::HashMap, sync::Arc};
mod emissive;

#[derive(Default)]
pub(super) struct EmitterCache {
    source: Option<(u64, u64)>,
    names_revision: u64,
    radius: u8,
    authored: Option<Arc<render_scene::EnhancedTextureAssets>>,
    templates: HashMap<u32, Vec<DxrLocalLight>>,
    emitters: HashMap<u32, [f32; 3]>,
    assets: Option<Arc<assets::RuntimeAssets>>,
    sections: HashMap<world::SubChunkKey, (Arc<world::SubChunk>, Vec<DxrLocalLight>)>,
}

fn light(position: [f32; 3], emission: u8, colour: [f32; 3]) -> DxrLocalLight {
    DxrLocalLight {
        position,
        radius: f32::from(emission),
        colour,
        intensity: 1.2,
    }
}

/// Source identifiers and provisional linear colours for the first local-light set.
const EMITTERS: [(&str, [f32; 3]); 6] = [
    ("minecraft:torch", [1.0, 0.65, 0.32]),
    ("minecraft:lantern", [1.0, 0.65, 0.32]),
    ("minecraft:soul_torch", [0.25, 0.75, 1.0]),
    ("minecraft:soul_lantern", [0.25, 0.75, 1.0]),
    ("minecraft:redstone_torch", [1.0, 0.08, 0.03]),
    ("minecraft:end_rod", [0.85, 0.9, 1.0]),
];

/// Reads only changed palettes, bounded to two section scans per frame; held lights update every frame.
pub(super) fn publish(
    preference: Res<DxrTerrainShadows>,
    world: Option<Res<ClientWorld>>,
    player: Res<client_state::PlayerRuntime>,
    ui: Res<UiRuntimeResource>,
    cameras: Query<&Transform, With<FlyCamera>>,
    textures: Option<Res<render_scene::ChunkTextureAssets>>,
    mut output: ResMut<DxrLocalLights>,
    mut cache: Local<EmitterCache>,
) {
    let Some((stream, camera)) = world
        .as_deref()
        .and_then(|world| world.stream.as_ref())
        .zip(cameras.single().ok())
        .filter(|_| preference.0)
    else {
        output.0.clear();
        *cache = EmitterCache::default();
        return;
    };
    let assets = stream.runtime_assets();
    let source = stream.terrain_publication_source().0;
    if cache.source != Some(source)
        || cache
            .assets
            .as_ref()
            .is_none_or(|old| !Arc::ptr_eq(old, assets))
    {
        *cache = EmitterCache {
            source: Some(source),
            assets: Some(assets.clone()),
            ..default()
        };
    }
    let maps = textures
        .as_ref()
        .and_then(|textures| textures.enhanced.clone());
    if match (&cache.authored, &maps) {
        (Some(old), Some(new)) => !Arc::ptr_eq(old, new),
        (None, None) => false,
        _ => true,
    } {
        cache.authored = maps.clone();
        cache.sections.clear();
        cache.templates.clear();
    }
    if cache.names_revision != stream.authority().block_name_revision() {
        cache.names_revision = stream.authority().block_name_revision();
        cache.sections.clear();
        cache.templates.clear();
        cache.emitters = EMITTERS
            .iter()
            .flat_map(|&(name, colour)| {
                stream
                    .authority()
                    .block_ids_for_name(name)
                    .iter()
                    .map(move |&id| (id, colour))
            })
            .collect();
    }
    cache.radius = cache
        .emitters
        .keys()
        .filter_map(|&id| stream.block_network_id(id))
        .filter_map(|wire| assets.resolve_known(stream.network_id_mode(), wire))
        .map(|block| block.light_properties().emission())
        .max()
        .unwrap_or(0);
    if maps.is_some() {
        cache.radius = cache.radius.max(15);
    }
    let horizon = render_scene::terrain_shadows::DXR_RECEIVER_RADIUS;
    let span = ((horizon + f32::from(cache.radius)) / world::SUB_CHUNK_SIDE as f32).ceil() as i32;
    let eye = camera.translation;
    let section = eye
        .floor()
        .as_ivec3()
        .to_array()
        .map(|axis| axis.div_euclid(world::SUB_CHUNK_SIDE as i32));
    let mut keys = Vec::new();
    for y in -span..=span {
        for z in -span..=span {
            for x in -span..=span {
                keys.push(world::SubChunkKey::new(
                    stream.current_dimension(),
                    section[0] + x,
                    section[1] + y,
                    section[2] + z,
                ));
            }
        }
    }
    keys.sort_by_key(|key| {
        (key.x - section[0]).pow(2) + (key.y - section[1]).pow(2) + (key.z - section[2]).pow(2)
    });
    cache.sections.retain(|key, _| keys.contains(key));
    let mut scans = 0;
    for key in keys {
        let Some(data) = stream.collision_store().sub_chunk(key) else {
            cache.sections.remove(&key);
            continue;
        };
        if cache
            .sections
            .get(&key)
            .is_some_and(|(old, _)| Arc::ptr_eq(old, &data))
        {
            continue;
        }
        // Withdraw old lights immediately on a block edit; discovering replacements is budgeted.
        cache.sections.remove(&key);
        if scans == 2 {
            continue;
        }
        scans += 1;
        let mut emitters = scan_section(key, &data, |id| {
            let &colour = cache.emitters.get(&id)?;
            let emission = stream
                .block_network_id(id)
                .and_then(|wire| assets.resolve_known(stream.network_id_mode(), wire))
                .map_or(0, |block| block.light_properties().emission());
            (emission != 0).then_some((emission, colour))
        });
        if let Some(maps) = maps.as_ref() {
            emitters.extend(emissive::scan(key, &data, |id| {
                if cache.emitters.contains_key(&id) {
                    return Vec::new();
                }
                cache
                    .templates
                    .entry(id)
                    .or_insert_with(|| {
                        stream
                            .block_network_id(id)
                            .and_then(|wire| assets.resolve_known(stream.network_id_mode(), wire))
                            .map_or_else(Vec::new, |block| emissive::templates(assets, maps, block))
                    })
                    .clone()
            }));
        }
        cache.sections.insert(key, (data, emitters));
    }
    let mut lights = Vec::new();
    for (stack, side) in [
        (player.selected_stack(), 1.0),
        (ui.local_offhand_stack(&player), -1.0),
    ] {
        let Some(stack) = stack else {
            continue;
        };
        let Some(name) = stream.authority().item_identifier(stack.network_id) else {
            continue;
        };
        let Some(colour) = EMITTERS
            .iter()
            .find(|(id, _)| *id == name.as_ref())
            .map(|(_, colour)| *colour)
        else {
            continue;
        };
        let emission =
            client_interaction::block_use::held_block_store_id(stream, stack.block_runtime_id)
                .and_then(|id| stream.block_network_id(id))
                .and_then(|wire| assets.resolve_known(stream.network_id_mode(), wire))
                .map(|block| block.light_properties().emission())
                .or_else(|| {
                    stream
                        .authority()
                        .block_ids_for_name(&name)
                        .iter()
                        .filter_map(|&id| stream.block_network_id(id))
                        .filter_map(|wire| assets.resolve_known(stream.network_id_mode(), wire))
                        .map(|block| block.light_properties().emission())
                        .max()
                })
                .unwrap_or(0);
        if emission != 0 {
            let position =
                eye + *camera.right() * side * 0.25 + *camera.forward() * 0.35 - Vec3::Y * 0.25;
            lights.push(light(position.to_array(), emission, colour));
        }
    }
    let placed = cache
        .sections
        .values()
        .flat_map(|(_, lights)| lights.iter().copied())
        .filter(|light| {
            Vec3::from(light.position).distance_squared(eye) < (light.radius + horizon).powi(2)
        });
    output.0 = select_lights(eye, lights, placed);
}

fn select_lights(
    eye: Vec3,
    mut lights: Vec<DxrLocalLight>,
    placed: impl Iterator<Item = DxrLocalLight>,
) -> Vec<DxrLocalLight> {
    let mut placed: Vec<_> = placed.collect();

    placed.sort_by(|a, b| {
        Vec3::from(a.position)
            .distance_squared(eye)
            .total_cmp(&Vec3::from(b.position).distance_squared(eye))
            .then_with(|| a.position[0].total_cmp(&b.position[0]))
            .then_with(|| a.position[1].total_cmp(&b.position[1]))
            .then_with(|| a.position[2].total_cmp(&b.position[2]))
    });
    lights.extend(
        placed
            .into_iter()
            .take(MAX_DXR_LOCAL_LIGHTS.saturating_sub(lights.len())),
    );
    lights
}

/// Scans immutable block data; a replacement palette naturally withdraws removed emitters.
fn scan_section(
    key: world::SubChunkKey,
    data: &world::SubChunk,
    mut resolve: impl FnMut(u32) -> Option<(u8, [f32; 3])>,
) -> Vec<DxrLocalLight> {
    let mut emitters = Vec::new();
    let mut properties = HashMap::new();
    let side = world::SUB_CHUNK_SIDE;
    for y in 0..side as u8 {
        for z in 0..side as u8 {
            for x in 0..side as u8 {
                for layer in 0..2 {
                    let Some(id) = data.runtime_id(layer, x, y, z) else {
                        continue;
                    };
                    let Some((emission, colour)) =
                        *properties.entry(id).or_insert_with(|| resolve(id))
                    else {
                        continue;
                    };
                    emitters.push(light(
                        [
                            key.x as f32 * side as f32 + f32::from(x) + 0.5,
                            key.y as f32 * side as f32 + f32::from(y) + 0.5,
                            key.z as f32 * side as f32 + f32::from(z) + 0.5,
                        ],
                        emission,
                        colour,
                    ));
                    break;
                }
            }
        }
    }
    emitters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxr_emitter_names_are_pinned_items() {
        for (name, colour) in EMITTERS {
            assert!(
                assets::item::names::identifiers().any(|id| id == name),
                "{name}"
            );
            assert!(
                colour
                    .into_iter()
                    .all(|channel| (0.0..=1.0).contains(&channel))
            );
        }
    }

    #[test]
    fn dxr_placed_emitters_follow_palette_replacements() {
        let mut blocks = [0; world::SUB_CHUNK_SIDE.pow(3)];
        blocks[0] = 1;
        let key = world::SubChunkKey::new(0, -1, 4, 2);
        let resolve = |id| (id == 1).then_some((14, EMITTERS[0].1));
        let source = world::SubChunk::from_runtime_ids(&blocks);
        let lights = scan_section(key, &source, resolve);
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].position, [-15.5, 64.5, 32.5]);
        assert_eq!(lights[0].radius, 14.0);
        blocks[0] = 0;
        assert!(scan_section(key, &world::SubChunk::from_runtime_ids(&blocks), resolve).is_empty());
    }

    #[test]
    fn dxr_both_hands_survive_placed_light_budget_and_clear_on_switch() {
        let main = light([0.2, 0.0, 0.0], 14, EMITTERS[0].1);
        let off = light([-0.2, 0.0, 0.0], 15, EMITTERS[1].1);
        let placed = || {
            (0..MAX_DXR_LOCAL_LIGHTS * 2)
                .map(|index| light([index as f32 + 1.0, 0.0, 0.0], 14, EMITTERS[0].1))
        };
        let selected = select_lights(Vec3::ZERO, vec![main, off], placed());
        assert_eq!(selected.len(), MAX_DXR_LOCAL_LIGHTS);
        assert_eq!(&selected[..2], &[main, off]);
        assert!(select_lights(Vec3::ZERO, Vec::new(), std::iter::empty()).is_empty());
        let replaced = select_lights(Vec3::ZERO, vec![off], placed());
        assert!(!replaced.contains(&main));
    }
}
