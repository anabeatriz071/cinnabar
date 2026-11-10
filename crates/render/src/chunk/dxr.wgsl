enable wgpu_ray_query;

struct Frame {
    world_from_clip: mat4x4<f32>,
    eye_radius: vec4<f32>,
    sun_strength: vec4<f32>,
    viewport: vec4<f32>,
    ray_length: vec4<f32>,
    invalid_instances: vec4<u32>,
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

// Disk samples use a fixed world-space basis, so camera motion does not rotate the pattern.
fn occlusion(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let filtering = any(frame.invalid_instances != vec4<u32>(0u));
    let flags = RAY_FLAG_TERMINATE_ON_FIRST_HIT | select(0u, RAY_FLAG_FORCE_NO_OPAQUE, filtering);
    var query: ray_query;
    rayQueryInitialize(&query, terrain,
        RayDesc(flags, 0xffu, 0.005, frame.ray_length.x, origin, direction));
    while rayQueryProceed(&query) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let instance = candidate.instance_custom_data;
        if (frame.invalid_instances[instance / 32u] & (1u << (instance % 32u))) == 0u {
            rayQueryConfirmIntersection(&query);
        }
    }
    return select(0.0, 1.0,
        rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE);
}

// Two opposing probes cover fully lit/shaded regions; mixed edges use the other diagonal too.
fn sun_occlusion(origin: vec3<f32>, sun: vec3<f32>) -> f32 {
    let axis = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(sun.y) > 0.99);
    let tangent = normalize(cross(sun, axis));
    let bitangent = cross(sun, tangent);
    let diagonal = (tangent + bitangent) * (frame.ray_length.y * 0.5);
    let a = occlusion(origin, normalize(sun + diagonal));
    let b = occlusion(origin, normalize(sun - diagonal));
    if a == b {
        return a;
    }
    let other = (tangent - bitangent) * (frame.ray_length.y * 0.5);
    return (a + b + occlusion(origin, normalize(sun + other))
        + occlusion(origin, normalize(sun - other))) * 0.25;
}

fn shadow(pixel: vec2<f32>, depth: f32) -> vec4<f32> {
    let uv = (pixel - frame.viewport.xy) / frame.viewport.zw;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let homogeneous = frame.world_from_clip * vec4(ndc, depth, 1.0);
    let denominator = select(-1.0, 1.0, homogeneous.w >= 0.0) * max(abs(homogeneous.w), 0.000001);
    let world = homogeneous.xyz / denominator;
    let camera_offset = world - frame.eye_radius.xyz;
    let distance = length(camera_offset);
    // Screen derivatives recover the surface plane before divergent early returns.
    let plane = cross(dpdx(world), dpdy(world));
    let plane_length = length(plane);
    let facing = select(-1.0, 1.0, dot(plane, -camera_offset) >= 0.0);
    let normal = select(-camera_offset / max(distance, 0.01),
        plane * facing / max(plane_length, 0.000001), plane_length > 0.000001);
    if depth <= 0.0 || distance >= frame.eye_radius.w || distance <= 0.01 {
        return vec4(0.0);
    }
    // A surface-normal offset stays on the same side when the camera changes direction.
    let origin = world + normal * 0.015;
    let sun = normalize(frame.sun_strength.xyz);
    // Vanilla's lightmap already shades back-facing surfaces; avoid darkening them again.
    let incidence = max(dot(normal, sun), 0.0);
    if incidence <= 0.0 {
        return vec4(0.0);
    }
    let coverage = sun_occlusion(origin, sun);
    let fade = 1.0 - smoothstep(frame.eye_radius.w * 0.75, frame.eye_radius.w, distance);
    let alpha = frame.sun_strength.w * incidence * coverage * fade;
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
