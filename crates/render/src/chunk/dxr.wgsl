enable wgpu_ray_query;

struct Frame {
    world_from_clip: mat4x4<f32>,
    eye_radius: vec4<f32>,
    sun_strength: vec4<f32>,
    viewport: vec4<f32>,
    ray_length: vec4<f32>,
}

@group(0) @binding(0) var terrain: acceleration_structure;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var scene_msaa_depth: texture_depth_multisampled_2d;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2(f32(index & 1u), f32(index >> 1u));
    return vec4(corner * 4.0 - vec2(1.0), 0.0, 1.0);
}

fn shadow(pixel: vec2<f32>, depth: f32) -> vec4<f32> {
    if depth <= 0.0 {
        return vec4(0.0);
    }
    let uv = (pixel - frame.viewport.xy) / frame.viewport.zw;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let homogeneous = frame.world_from_clip * vec4(ndc, depth, 1.0);
    let world = homogeneous.xyz / homogeneous.w;
    let camera_offset = world - frame.eye_radius.xyz;
    let distance = length(camera_offset);
    if distance >= frame.eye_radius.w || distance <= 0.01 {
        return vec4(0.0);
    }
    // Nudge the ray toward the viewer to keep the visible surface out of its own shadow.
    let origin = world - camera_offset * (0.015 / distance);
    var query: ray_query;
    rayQueryInitialize(
        &query, terrain,
        RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xffu, 0.005, frame.ray_length.x,
            origin, frame.sun_strength.xyz),
    );
    // All admitted casters are opaque triangles.
    _ = rayQueryProceed(&query);
    let hit = rayQueryGetCommittedIntersection(&query);
    let fade = 1.0 - smoothstep(frame.eye_radius.w * 0.75, frame.eye_radius.w, distance);
    let alpha = select(0.0, frame.sun_strength.w * fade, hit.kind != RAY_QUERY_INTERSECTION_NONE);
    return vec4(0.0, 0.0, 0.0, alpha);
}

@fragment
fn single(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    return shadow(pixel.xy, textureLoad(scene_depth, vec2<i32>(pixel.xy), 0));
}

@fragment
fn multisampled(
    @builtin(position) pixel: vec4<f32>,
    @builtin(sample_index) sample: u32,
) -> @location(0) vec4<f32> {
    return shadow(pixel.xy, textureLoad(scene_msaa_depth, vec2<i32>(pixel.xy), i32(sample)));
}
