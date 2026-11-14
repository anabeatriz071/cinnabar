// Enhanced post chain: half-resolution light shafts and
// the HDR composite with grading and a filmic shoulder.
import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import render::enhanced::common::{
    EnhancedFrame, FEATURE_SHAFTS, interleaved_gradient_noise,
};

@group(0) @binding(0) var<uniform> frame: EnhancedFrame;
@group(0) @binding(1) var source_texture: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var shaft_texture: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_depth_2d;
@group(0) @binding(6) var shadow_map: texture_depth_2d_array;
@group(0) @binding(7) var shadow_sampler: sampler_comparison;

const SHAFT_STEPS: u32 = 24u;
const SHAFT_ANISOTROPY: f32 = 0.72;
// Share of the forward (sunward) lobe; the rest scatters evenly so lit haze and
// shadowed gaps stay visible when looking away from the sun.
const SHAFT_FORWARD_WEIGHT: f32 = 0.6;
const SHAFT_DENSITY: f32 = 0.012;

// Linear Rec.709 luminance.
fn luminance(colour: vec3<f32>) -> f32 {
    return dot(colour, vec3(0.2126, 0.7152, 0.0722));
}

// Henyey-Greenstein forward lobe blended with isotropic scattering.
fn shaft_phase(cos_theta: f32) -> f32 {
    let g = SHAFT_ANISOTROPY;
    let denominator = 1.0 + g * g - 2.0 * g * cos_theta;
    let forward = (1.0 - g * g) / (12.566371 * denominator * sqrt(denominator));
    return mix(0.07957747, forward, SHAFT_FORWARD_WEIGHT);
}

// Sun visibility from the finest cascade containing `world`; points outside
// every cascade are treated as lit, so rays leaving the map do not darken.
fn sun_visibility(world: vec3<f32>) -> f32 {
    for (var cascade = 0u; cascade < frame.flags.y; cascade += 1u) {
        let clip = frame.cascade_clip_from_world[cascade] * vec4(world, 1.0);
        let ndc = clip.xyz / clip.w;
        let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (all(uv > vec2(0.0)) && all(uv < vec2(1.0)) && ndc.z > 0.0 && ndc.z < 1.0) {
            let bias = frame.cascade_texel[cascade] * frame.cascade_depth_scale[cascade];
            return textureSampleCompareLevel(shadow_map, shadow_sampler, uv, i32(cascade), ndc.z - bias);
        }
    }
    return 1.0;
}

// Single-scattered sunlight along the view ray, up to the surface or the shadow
// distance, attenuated by the fog in front of each sample.
@fragment
fn light_shafts(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    if ((frame.flags.x & FEATURE_SHAFTS) == 0u) {
        return vec4(0.0);
    }
    let size = vec2<i32>(textureDimensions(depth_texture));
    let coord = clamp(vec2<i32>(in.uv * vec2<f32>(size)), vec2(0), size - vec2(1));
    let depth = max(textureLoad(depth_texture, coord, 0), 1.0e-6);
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let world_h = frame.world_from_clip * vec4(ndc, depth, 1.0);
    let camera = frame.camera_time.xyz;
    let ray = world_h.xyz / world_h.w - camera;
    let ray_length = length(ray);
    let direction = ray / max(ray_length, 1.0e-4);
    let march = min(ray_length, frame.cascade_texel.w);
    let step_length = march / f32(SHAFT_STEPS);
    let noise = interleaved_gradient_noise(in.position.xy);
    var inscatter = 0.0;
    for (var step_index = 0u; step_index < SHAFT_STEPS; step_index += 1u) {
        let travel = (f32(step_index) + noise) * step_length;
        inscatter += sun_visibility(camera + direction * travel) * exp(-SHAFT_DENSITY * travel);
    }
    // Riemann sum of the in-scattering integral; fully lit rays give 1 - exp(-density * march).
    inscatter *= SHAFT_DENSITY * step_length;
    let scatter = inscatter * shaft_phase(dot(direction, frame.light_direction.xyz));
    let strength = frame.light_direction.w * frame.grade.w * (1.0 - 0.6 * frame.ambient_colour.w);
    return vec4(frame.light_colour.rgb * scatter * strength, 1.0);
}

// View distance of the scene at `uv` from reverse-Z depth; the sky reads as very far.
fn scene_distance(uv: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth_texture));
    let coord = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - vec2(1));
    return frame.projection.x / max(textureLoad(depth_texture, coord, 0), 1.0e-6);
}

// Depth-aware 3x3 tent upsample of the half-resolution shafts. It removes the
// per-pixel march jitter without bleeding light across silhouettes.
fn filtered_shafts(uv: vec2<f32>) -> vec3<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(shaft_texture));
    let centre = scene_distance(uv);
    var sum = vec3(0.0);
    var total = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let tap = uv + vec2(f32(x), f32(y)) * texel;
            let difference = abs(scene_distance(tap) - centre) / (0.1 * centre + 0.5);
            let weight = (2.0 - abs(f32(x))) * (2.0 - abs(f32(y))) / (1.0 + difference * difference);
            sum += weight * textureSampleLevel(shaft_texture, linear_sampler, tap, 0.0).rgb;
            total += weight;
        }
    }
    return sum / total;
}

// Narkowicz's published ACES fit, evaluated in linear RGB (see README.md).
fn filmic_shoulder(colour: vec3<f32>) -> vec3<f32> {
    let positive = max(colour, vec3(0.0));
    let numerator = 2.51 * positive * positive + 0.03 * positive;
    let denominator = 2.43 * positive * positive + 0.59 * positive + vec3(0.14);
    return clamp(numerator / denominator, vec3(0.0), vec3(1.0));
}

// Warm daylight and cool night balance with mild contrast.
fn grade(colour: vec3<f32>, warmth: f32) -> vec3<f32> {
    let warm = vec3(1.06, 1.0, 0.9);
    let cool = vec3(0.9, 0.97, 1.08);
    let balance = mix(vec3(1.0), select(cool, warm, warmth > 0.0), min(abs(warmth), 1.0));
    let balanced = colour * balance;
    let saturated = mix(vec3(luminance(balanced)), balanced, 1.08);
    // Gentle contrast pivoting on mid-grey.
    return 0.18 * pow(max(saturated, vec3(0.0)) / 0.18, vec3(1.04));
}

@fragment
fn composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var colour = textureSampleLevel(source_texture, linear_sampler, in.uv, 0.0).rgb;
    if ((frame.flags.x & FEATURE_SHAFTS) != 0u) {
        colour += filtered_shafts(in.uv);
    }
    colour = filmic_shoulder(grade(colour * frame.grade.y, frame.grade.x));
    let vignette = mix(1.0, (1.0 - smoothstep(0.3, 0.95, length(in.uv - vec2(0.5)))), 0.22);
    let dither = (interleaved_gradient_noise(in.position.xy) - 0.5) / 255.0;
    return vec4(max(colour * vignette + dither, vec3(0.0)), 1.0);
}
