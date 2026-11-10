enable wgpu_ray_query;

@group(0) @binding(0) var terrain: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> occluded: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ground_x = array(0.15, 1.15, 8.15);
    let origin = vec3(ground_x[id.x], 0.0, 0.35);
    let sun_direction = normalize(vec3(0.25, 1.0, 0.0));
    var query: ray_query;
    rayQueryInitialize(
        &query, terrain,
        RayDesc(0u, 0xffu, 0.001, 32.0, origin, sun_direction),
    );
    // All fixture geometry is opaque, so a single advance resolves the committed hit.
    _ = rayQueryProceed(&query);
    let hit = rayQueryGetCommittedIntersection(&query);
    occluded[id.x] = u32(hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
