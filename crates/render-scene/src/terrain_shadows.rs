//! Live preference for the experimental Windows sunlight pass.

use bevy::{prelude::Resource, render::extract_resource::ExtractResource};

/// The menu can stop shadow preparation and drawing without recreating the device.
#[derive(Resource, ExtractResource, Clone, Copy, Default, Debug, PartialEq, Eq)]
#[extract_app(bevy::render::RenderApp)]
pub struct DxrTerrainShadows(pub bool);

/// Runtime admission from the selected device; never a persisted user preference.
#[derive(Resource, Default, Clone, Copy)]
pub struct DxrTerrainShadowsSupported(pub bool);

/// One optional client-side local light. Linear colour and radius do not affect server light state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DxrLocalLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub colour: [f32; 3],
    pub intensity: f32,
}

/// Nearest placed emitters and both hands, published independently of shader packs.
#[derive(Resource, ExtractResource, Clone, Default)]
#[extract_app(bevy::render::RenderApp)]
pub struct DxrLocalLights(pub Vec<DxrLocalLight>);

/// Bounded local-light work for the experimental renderer.
pub const MAX_DXR_LOCAL_LIGHTS: usize = 32;

/// Receiver distance shared by caster preparation and local-light discovery.
pub const DXR_RECEIVER_RADIUS: f32 = 32.0;
