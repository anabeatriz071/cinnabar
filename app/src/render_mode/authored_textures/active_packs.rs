//! Bridges the in-game admitted pack selection to the optional material decoder.
use super::{Arc, AuthoredMaps, MaterialKeys, RuntimeAssets, enhanced_textures};
use bevy::prelude::{info, warn};
use pack_compiler::pbr::PbrPack;
use resource_pack::{LayeredPackView, ValidatedPackStack};
use std::{
    collections::BTreeMap,
    path::{Component, Path},
};

/// Decodes admitted textures off-thread; temporary files live only for this decode.
pub(super) fn load(
    base: &RuntimeAssets,
    runtime: &RuntimeAssets,
    keys: &MaterialKeys,
    selection: Option<Arc<ValidatedPackStack>>,
) -> AuthoredMaps {
    let Some(selection) = selection.filter(|stack| !stack.packs().is_empty()) else {
        return enhanced_textures::load_optional_enhanced_textures(runtime, keys);
    };
    let directory = tempfile::tempdir().ok()?;
    let view = LayeredPackView::new(selection);
    for name in view.list_with_suffixes("textures/", &[".png", ".tga", ".jpg", ".jpeg", ".json"]) {
        let relative = Path::new(name);
        if !relative
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        {
            continue;
        }
        let Some(bytes) = view.read_capped(name, 64 * 1024 * 1024) else {
            continue;
        };
        let path = directory.path().join(relative);
        std::fs::create_dir_all(path.parent()?).ok()?;
        std::fs::write(path, bytes).ok()?;
    }
    let maps = enhanced_textures::load_selected(
        base,
        runtime,
        keys,
        vec![PbrPack::new(directory.path(), None)],
        128,
    );
    if maps.is_none() {
        warn!("Active resource packs contain no usable terrain PBR maps");
    } else {
        info!("DXR materials loaded from the active in-game resource packs");
    }
    maps
}

/// Pack overlays allocate new material IDs. Bind their face textures to the same source key.
pub(super) fn remap_faces(
    base: &RuntimeAssets,
    runtime: &RuntimeAssets,
    keys: &MaterialKeys,
    groups: &mut BTreeMap<(String, String), Vec<enhanced_textures::Source>>,
) {
    for source in keys
        .block_sources()
        .iter()
        .filter(|source| source.cube_binding_eligible)
    {
        let original = base.resolve(assets::NetworkIdMode::Sequential, source.visual);
        let current = runtime.resolve(assets::NetworkIdMode::Sequential, source.visual);
        for face in assets::BlockFace::ALL {
            let old = original.face(face).material_id();
            let new = current.face(face).material_id();
            if old == new {
                continue;
            }
            let Some(material) = runtime.materials().get(new as usize) else {
                continue;
            };
            for sources in groups.values_mut() {
                let Some(mut mapped) = sources
                    .iter()
                    .find(|source| source.material == old)
                    .copied()
                else {
                    continue;
                };
                if sources.iter().any(|source| source.material == new) {
                    continue;
                }
                mapped.material = new;
                mapped.texture = material.texture.raw();
                sources.push(mapped);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dxr_documented_bedrock_format_is_admitted() {
        assert!(pack_compiler::pbr::PbrFormat::parse("bedrock").is_some());
    }
}
