enable wgpu_ray_query;
struct Frame {
    world_from_clip: mat4x4<f32>, eye_radius: vec4<f32>, sun_strength: vec4<f32>,
    viewport: vec4<f32>, ray_length: vec4<f32>, invalid_instances: vec4<u32>,
}
struct Instance { origin: vec3<f32>, quad_start: u32 }
struct Material { texture: u32, flags: u32, animation: u32, start: u32, count: u32, weight: u32 }
struct Light { position_radius: vec4<f32>, colour_intensity: vec4<f32> }
@group(0) @binding(0) var terrain: acceleration_structure;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var depth_single: texture_depth_2d;
@group(0) @binding(3) var depth_msaa: texture_depth_multisampled_2d;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> quads: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
@group(0) @binding(7) var<storage, read> authored: array<u32>;
@group(0) @binding(8) var<storage, read> lights: array<Light>;
@group(0) @binding(9) var tile_sampler: sampler;
@group(0) @binding(10) var carrier0: texture_2d_array<f32>;
@group(0) @binding(11) var carrier1: texture_2d_array<f32>;
@group(0) @binding(12) var carrier2: texture_2d_array<f32>;
@group(0) @binding(13) var carrier3: texture_2d_array<f32>;
@group(0) @binding(14) var carrier4: texture_2d_array<f32>;
@group(0) @binding(15) var colour0: texture_2d_array<f32>;
@group(0) @binding(16) var colour1: texture_2d_array<f32>;
@group(0) @binding(17) var normal0: texture_2d_array<f32>;
@group(0) @binding(18) var normal1: texture_2d_array<f32>;
@group(0) @binding(19) var mer0: texture_2d_array<f32>;
@group(0) @binding(20) var mer1: texture_2d_array<f32>;
@group(0) @binding(21) var base_sampler: sampler;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2(f32(index & 1u), f32(index >> 1u));
    return vec4(corner * 4.0 - vec2(1.0), 0.0, 1.0);
}

fn trace_with_flags(origin: vec3<f32>, direction: vec3<f32>, maximum: f32, flags: u32) -> RayIntersection {
    let filtering = any(frame.invalid_instances != vec4<u32>(0u));
    var query: ray_query;
    rayQueryInitialize(&query, terrain, RayDesc(flags | select(0u, RAY_FLAG_FORCE_NO_OPAQUE, filtering),
        0xffu, 0.015, maximum, origin, direction));
    while rayQueryProceed(&query) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let id = candidate.instance_custom_data;
        if (frame.invalid_instances[id / 32u] & (1u << (id % 32u))) == 0u {
            rayQueryConfirmIntersection(&query);
        }
    }
    return rayQueryGetCommittedIntersection(&query);
}

fn trace(origin: vec3<f32>, direction: vec3<f32>, maximum: f32) -> RayIntersection {
    return trace_with_flags(origin,direction,maximum,0u);
}
fn blocked(origin: vec3<f32>, direction: vec3<f32>, maximum: f32) -> bool {
    return trace_with_flags(origin,direction,maximum,RAY_FLAG_TERMINATE_ON_FIRST_HIT).kind!=RAY_QUERY_INTERSECTION_NONE;
}

// DXR_MATERIAL_FUNCTIONS

fn shade(pixel: vec2<f32>, depth: f32) -> vec4<f32> {
    if depth <= 0.0 { return vec4(0.0); }
    let uv=(pixel-frame.viewport.xy)/frame.viewport.zw;
    let h=frame.world_from_clip*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    let point=h.xyz/h.w;
    let offset=point-frame.eye_radius.xyz;
    let distance=length(offset);
    if distance<=0.02 || distance>=frame.eye_radius.w { return vec4(0.0); }
    let direction=offset/distance;
    let primary=trace(frame.eye_radius.xyz,direction,distance+0.08);
    if primary.kind==RAY_QUERY_INTERSECTION_NONE || abs(primary.t-distance)>0.08 { return vec4(0.0); }
    let receiver=surface(primary,point,footprint(pixel,depth,point));
    let origin=point+receiver.geometric*0.025;
    var added=receiver.colour*receiver.emission*0.8;
    var direct=vec3(0.0);
    var evaluated=0u;
    for (var index=0u; index<u32(frame.ray_length.z); index++) {
        let light=lights[index];
        let delta=light.position_radius.xyz-point;
        let range=length(delta);
        let cosine=max(dot(receiver.normal,delta/max(range,0.001)),0.0);
        if range>=light.position_radius.w || range<0.03 || cosine==0.0 { continue; }
        if !blocked(origin,delta/range,max(range-0.08,0.02)) {
            let falloff=pow(1.0-range/light.position_radius.w,2.0);
            let brdf=direct_brdf(receiver,-direction,delta/range);
            direct+=(brdf.diffuse+brdf.specular)*light.colour_intensity.rgb*light.colour_intensity.w*falloff*3.14159265;
        }
        evaluated++;
        if evaluated==4u { break; }
    }
    added+=min(direct,vec3(1.5));
    // Normal-map relief contributes only a restrained directional term to the lightmapped base.
    let sun=frame.sun_strength.xyz/max(length(frame.sun_strength.xyz),0.0001);
    let relief=(max(dot(receiver.normal,sun),0.0)-max(dot(receiver.geometric,sun),0.0))*frame.sun_strength.w;
    added+=receiver.colour*max(relief,0.0)*0.25;
    let relief_shadow=max(-relief,0.0)*0.25;
    if receiver.roughness<0.95 && dot(receiver.normal,sun)>0.0 && frame.sun_strength.w>0.001 {
        if !blocked(origin,sun,frame.ray_length.x) {
            added+=direct_brdf(receiver,-direction,sun).specular*frame.sun_strength.w*2.0;
        }
    }
    let reflectivity=receiver.metal*pow(1.0-receiver.roughness,2.0)*0.85;
    var reflected=vec3(0.0);
    if reflectivity>0.025 {
        let reflected_ray=reflect(direction,receiver.normal);
        let ray=normalize(reflected_ray+receiver.geometric*max(0.02-dot(reflected_ray,receiver.geometric),0.0));
        let hit=trace(origin,ray,frame.ray_length.x);
        reflected=environment(ray);
        if hit.kind!=RAY_QUERY_INTERSECTION_NONE {
            let reflected_surface=surface(hit,origin+ray*hit.t,max(receiver.roughness*0.08,footprint(pixel,depth,point)));
            let illumination=0.15+0.65*frame.sun_strength.w*max(dot(reflected_surface.normal,frame.sun_strength.xyz),0.0);
            reflected=reflected_surface.colour*(illumination+reflected_surface.emission);
        }
    }
    let fade=1.0-smoothstep(frame.eye_radius.w*0.75,frame.eye_radius.w,distance);
    return vec4((added+reflected*receiver.colour*reflectivity)*fade,(reflectivity+(1.0-reflectivity)*relief_shadow)*fade);
}
@fragment fn single(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    return shade(pixel.xy,textureLoad(depth_single,vec2<i32>(pixel.xy),0));
}
@fragment fn multisampled(@builtin(position) pixel: vec4<f32>, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    return shade(pixel.xy,textureLoad(depth_msaa,vec2<i32>(pixel.xy),i32(sample)));
}

// Hardware multiplicative blending replaces authored albedo while retaining the scene's
// native lightmap, fog, biome tint and sunlight shadowing, without sampling the render target.
fn albedo(pixel: vec2<f32>, depth: f32) -> vec4<f32> {
    if depth<=0.0 { return vec4(1.0); }
    let uv=(pixel-frame.viewport.xy)/frame.viewport.zw;
    let h=frame.world_from_clip*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    let point=h.xyz/h.w;
    let offset=point-frame.eye_radius.xyz;
    let distance=length(offset);
    if distance<=0.02 || distance>=frame.eye_radius.w { return vec4(1.0); }
    let hit=trace(frame.eye_radius.xyz,offset/distance,distance+0.08);
    if hit.kind==RAY_QUERY_INTERSECTION_NONE || abs(hit.t-distance)>0.08 { return vec4(1.0); }
    let receiver=surface(hit,point,footprint(pixel,depth,point));
    let fade=1.0-smoothstep(frame.eye_radius.w*0.75,frame.eye_radius.w,distance);
    let ratio=clamp(receiver.colour/max(receiver.base,vec3(0.01)),vec3(0.0),vec3(8.0));
    return vec4(mix(vec3(1.0),ratio,fade),1.0);
}
@fragment fn albedo_single(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    return albedo(pixel.xy,textureLoad(depth_single,vec2<i32>(pixel.xy),0));
}
@fragment fn albedo_multisampled(@builtin(position) pixel: vec4<f32>, @builtin(sample_index) sample: u32) -> @location(0) vec4<f32> {
    return albedo(pixel.xy,textureLoad(depth_msaa,vec2<i32>(pixel.xy),i32(sample)));
}
