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

