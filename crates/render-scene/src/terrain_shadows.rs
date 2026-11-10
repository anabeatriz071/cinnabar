//! Live preference for the experimental Windows sunlight pass.

use bevy::{prelude::Resource, render::extract_resource::ExtractResource};

/// The menu can stop shadow preparation and drawing without recreating the device.
#[derive(Resource, ExtractResource, Clone, Copy, Default, Debug, PartialEq, Eq)]
#[extract_app(bevy::render::RenderApp)]
pub struct DxrTerrainShadows(pub bool);

/// Runtime admission from the selected device; never a persisted user preference.
#[derive(Resource, Default, Clone, Copy)]
pub struct DxrTerrainShadowsSupported(pub bool);
