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

fn authored_reference(id: u32, carrier: u32) -> u32 {
    let raw = (carrier & DXR_LAYER_MASK) | (carrier & (7u << DXR_PAGE_SHIFT));
    var low = DXR_REFERENCE_WORDS;
    var high = arrayLength(&authored);
    // Binary search material-owned overrides after the sparse carrier table.
    var left = 0u;
    var right = (high - min(high, low)) / 3u;
    while left < right {
        let middle = (left + right) / 2u;
        let offset = low + middle * 3u;
        let material = authored[offset];
        let texture = authored[offset+1u];
        if material < id || (material == id && texture < raw) { left = middle+1u; }
        else { right = middle; }
    }
    let offset = low + left*3u;
    if offset+2u < high && authored[offset] == id && authored[offset+1u] == raw { return authored[offset+2u]; }
    let slot = ((carrier >> DXR_PAGE_SHIFT) & 7u) * (DXR_LAYER_MASK+1u) + (carrier & DXR_LAYER_MASK);
    return authored[min(slot, arrayLength(&authored)-1u)];
}

fn sample_base(image: texture_2d_array<f32>, reference: u32, uv: vec2<f32>, footprint: f32) -> vec3<f32> {
    let scale=exp2(-f32((reference >> DXR_GRID_SHIFT) & 31u));
    let encoded=vec2((reference >> DXR_WIDTH_SHIFT)&15u,(reference >> DXR_HEIGHT_SHIFT)&15u);
    let size=select(vec2(1u) << (max(encoded,vec2(1u))-vec2(1u)),textureDimensions(image,0),encoded==vec2(0u));
    let lod=max(log2(max(footprint,0.00001)*f32(max(size.x,size.y))*scale),0.0);
    return textureSampleLevel(image,base_sampler,fract(uv)*scale,i32(reference & DXR_LAYER_MASK),lod).rgb;
}
fn base_colour(reference: u32, uv: vec2<f32>, footprint: f32) -> vec3<f32> {
    switch (reference >> DXR_PAGE_SHIFT) & 7u {
        case 0u: { return sample_base(carrier0,reference,uv,footprint); }
        case 1u: { return sample_base(carrier1,reference,uv,footprint); }
        case 2u: { return sample_base(carrier2,reference,uv,footprint); }
        case 3u: { return sample_base(carrier3,reference,uv,footprint); }
        default: { return sample_base(carrier4,reference,uv,footprint); }
    }
}
struct Surface { colour: vec3<f32>, metal: f32, normal: vec3<f32>, roughness: f32, emission: f32, geometric: vec3<f32>, base: vec3<f32> }

fn surface(hit: RayIntersection, point: vec3<f32>, footprint: f32) -> Surface {
    let instance = instances[hit.instance_custom_data];
    let quad = quads[instance.quad_start + hit.primitive_index / 2u];
    let geometry = quad.x;
    let face = (geometry >> 15u) & 7u;
    let origin = vec3(f32(geometry & 31u), f32((geometry >> 5u) & 31u), f32((geometry >> 10u) & 31u));
    let local = point-instance.origin-origin;
    // Native axes: West +Z/-Y, East -Z/-Y, Down +X/-Z, Up +X/+Z, North -X/-Y, South +X/-Y.
    var tangent = vec3(1.0,0.0,0.0);
    var bitangent = vec3(0.0,-1.0,0.0);
    var normal = vec3(0.0,0.0,1.0);
    switch face {
        case 0u: { tangent=vec3(0.0,0.0,1.0); normal=vec3(-1.0,0.0,0.0); }
        case 1u: { tangent=vec3(0.0,0.0,-1.0); normal=vec3(1.0,0.0,0.0); }
        case 2u: { bitangent=vec3(0.0,0.0,-1.0); normal=vec3(0.0,-1.0,0.0); }
        case 3u: { bitangent=vec3(0.0,0.0,1.0); normal=vec3(0.0,1.0,0.0); }
        case 4u: { tangent=vec3(-1.0,0.0,0.0); normal=vec3(0.0,0.0,-1.0); }
        default: {}
    }
    var uv = vec2(dot(local,tangent), dot(local,bitangent));
    let material = materials[quad.y];
    // Apply pack UV turns to both sampling and the tangent frame.
    switch material.flags & 3u {
        case 1u: { uv=vec2(uv.y,-uv.x); let old=tangent; tangent=bitangent; bitangent=-old; }
        case 2u: { uv=-uv; tangent=-tangent; bitangent=-bitangent; }
        case 3u: { uv=vec2(-uv.y,uv.x); let old=tangent; tangent=-bitangent; bitangent=old; }
        default: {}
    }
    if (material.flags & 4u)!=0u { uv.x=-uv.x; tangent=-tangent; }
    if (material.flags & 8u)!=0u { uv.y=-uv.y; bitangent=-bitangent; }
    let reference = authored_reference(quad.y, material.texture);
    let base=base_colour(material.texture,uv,footprint);
    var result = Surface(base,0.0,normal,1.0,0.0,normal,base);
    if reference == 0xffffffffu { return result; }
    let layer = i32(reference & DXR_AUTHORED_MASK);
    let coordinate = fract(uv);
    let page = reference >> 31u;
    let dimensions=select(textureDimensions(colour0,0),textureDimensions(colour1,0),page!=0u);
    let lod=max(log2(max(footprint,0.00001)*f32(max(dimensions.x,dimensions.y))),0.0);
    if (reference & (1u << 11u)) != 0u {
        if page == 0u { result.colour=textureSampleLevel(colour0,tile_sampler,coordinate,layer,lod).rgb; }
        else { result.colour=textureSampleLevel(colour1,tile_sampler,coordinate,layer,lod).rgb; }
    }
    if (reference & (1u << 12u)) != 0u {
        var map = vec4(0.5,0.5,1.0,0.0);
        if page == 0u { map=textureSampleLevel(normal0,tile_sampler,coordinate,layer,lod); }
        else { map=textureSampleLevel(normal1,tile_sampler,coordinate,layer,lod); }
        let xy=map.xy*2.0-vec2(1.0);
        result.normal=normalize(tangent*xy.x+bitangent*xy.y+normal*sqrt(max(1.0-dot(xy,xy),0.0)));
    }
    if (reference & (1u << 14u)) != 0u {
        var mer=vec4(0.0,0.0,1.0,0.0);
        if page == 0u { mer=textureSampleLevel(mer0,tile_sampler,coordinate,layer,lod); }
        else { mer=textureSampleLevel(mer1,tile_sampler,coordinate,layer,lod); }
        result.metal=mer.x; result.emission=mer.y; result.roughness=clamp(mer.z,0.04,1.0);
        // LabPBR stores categorical metal/F0 values, normalized emission and perceptual roughness.
        if (reference & (1u << 15u)) != 0u { result.metal=select(0.0,1.0,mer.x>=230.0/255.0); }
    }
    return result;
}

fn environment(direction: vec3<f32>) -> vec3<f32> {
    let packed=u32(frame.ray_length.y);
    let sky=vec3(f32(packed & 255u),f32((packed>>8u)&255u),f32((packed>>16u)&255u))/255.0;
    return sky*mix(0.55,1.0,max(direction.y,0.0));
}

// GGX distribution and Smith visibility with Schlick Fresnel; direct terms stay linear RGB.
struct Brdf { diffuse: vec3<f32>, specular: vec3<f32> }
fn direct_brdf(material: Surface, view: vec3<f32>, light: vec3<f32>) -> Brdf {
    let nl=max(dot(material.normal,light),0.0);
    let nv=max(dot(material.normal,view),0.0001);
    let half=(view+light)/max(length(view+light),0.0001);
    let nh=max(dot(material.normal,half),0.0);
    let vh=max(dot(view,half),0.0);
    let alpha=material.roughness*material.roughness;
    let alpha2=alpha*alpha;
    let divisor=nh*nh*(alpha2-1.0)+1.0;
    let distribution=alpha2/(3.14159265*max(divisor*divisor,0.000000000001));
    let k=pow(material.roughness+1.0,2.0)*0.125;
    let visibility=(nv/(nv*(1.0-k)+k))*(nl/max(nl*(1.0-k)+k,0.0001));
    let f0=mix(vec3(0.04),material.colour,material.metal);
    let fresnel=f0+(vec3(1.0)-f0)*pow(1.0-vh,5.0);
    let specular=distribution*visibility*fresnel/max(4.0*nv*nl,0.0001);
    let diffuse=(vec3(1.0)-fresnel)*(1.0-material.metal)*material.colour/3.14159265;
    return Brdf(diffuse*nl,specular*nl);
}
fn footprint(pixel: vec2<f32>, depth: f32, point: vec3<f32>) -> f32 {
    let uv=(pixel-frame.viewport.xy)/frame.viewport.zw;
    let adjacent=frame.world_from_clip*vec4((uv.x+1.0/frame.viewport.z)*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    return length(adjacent.xyz/adjacent.w-point);
}

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
