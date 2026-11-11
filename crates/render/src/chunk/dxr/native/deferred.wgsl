struct Frame {
    world_from_clip: mat4x4<f32>, eye_radius: vec4<f32>, sun_strength: vec4<f32>,
    viewport: vec4<f32>, ray_length: vec4<f32>, invalid_instances: vec4<u32>,
}
struct Instance { origin: vec3<f32>, quad_start: u32 }
struct Material { texture: u32, flags: u32, animation: u32, start: u32, count: u32, weight: u32 }
struct Light { position_radius: vec4<f32>, colour_intensity: vec4<f32> }
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

struct RayIntersection { instance_custom_data: u32, primitive_index: u32, t: f32, kind: u32 }
struct Params { base: Frame, grid: vec4<u32>, fog: vec4<f32>, sky: vec4<f32> }
struct Receiver { point: vec3<f32>, normal: u32 }
struct Normal { mapped: vec4<f32>, geometric: vec4<f32> }
struct Lighting { visibility: vec4<u32>, reflection: vec4<u32>, direction_sun: vec4<f32> }
@group(1) @binding(0) var<uniform> config: Params;
@group(1) @binding(1) var<storage, read> receivers: array<Receiver>;
@group(1) @binding(2) var<storage, read> hits: array<vec4<u32>>;
@group(2) @binding(0) var<storage, read_write> receiver_output: array<Receiver>;
@group(2) @binding(1) var<storage, read_write> normal_output: array<Normal>;
@group(1) @binding(4) var<storage, read> traced: array<Lighting>;
@group(1) @binding(5) var<storage, read> beams: array<vec4<f32>>;
// DXR_MATERIAL_FUNCTIONS
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner=vec2(f32(index & 1u),f32(index >> 1u));
    return vec4(corner*4.0-vec2(1.0),0.0,1.0);
}
fn point_at(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv=(pixel-frame.viewport.xy)/frame.viewport.zw;
    let h=frame.world_from_clip*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
    return h.xyz/h.w;
}
fn pixel_at(cell: vec2<u32>) -> vec2<f32> { return frame.viewport.xy+(vec2<f32>(cell)+vec2(0.5))/vec2<f32>(config.grid.xy)*frame.viewport.zw; }
fn index_at(cell: vec2<u32>) -> u32 { return cell.y*config.grid.x+cell.x; }
fn packed(normal:vec3<f32>) -> u32 {
    let n=vec3<u32>(round(clamp(normal*0.5+vec3(0.5),vec3(0.0),vec3(1.0))*1023.0));
    return n.x | (n.y<<10u) | (n.z<<20u) | 0x80000000u;
}
fn unpacked(value:u32) -> vec3<f32> {return normalize(vec3<f32>(f32(value&1023u),f32((value>>10u)&1023u),f32((value>>20u)&1023u))*(2.0/1023.0)-vec3(1.0));}
fn intersection(value:vec4<u32>) -> RayIntersection {return RayIntersection(value.x,value.y,bitcast<f32>(value.z),value.w);}
fn matched(index:u32) -> bool {return hits[index].w!=0u && abs(bitcast<f32>(hits[index].z)-length(receivers[index].point-frame.eye_radius.xyz))<0.08;}
fn receive(cell:vec2<u32>,depth:f32,dx:f32,dy:f32) {
    let index=index_at(cell);
    receiver_output[index]=Receiver(vec3(0.0),0u);
    if depth<=0.0 {return;}
    let pixel=pixel_at(cell);
    let point=point_at(pixel,depth);
    if distance(point,frame.eye_radius.xyz)>frame.eye_radius.w {return;}
    var n=cross(point_at(pixel+vec2(1.0,0.0),dx)-point,point_at(pixel+vec2(0.0,1.0),dy)-point);
    if length(n)<0.00001 {return;}
    n=normalize(n);
    if dot(n,point-frame.eye_radius.xyz)>0.0 {n=-n;}
    receiver_output[index]=Receiver(point,packed(n));
}
@compute @workgroup_size(8,8) fn receivers_single(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=config.grid.xy) {return;}
    let pixel=vec2<i32>(pixel_at(id.xy));
    let maximum=vec2<i32>(textureDimensions(depth_single))-vec2(1);
    receive(id.xy,textureLoad(depth_single,pixel,0),textureLoad(depth_single,min(pixel+vec2(1,0),maximum),0),textureLoad(depth_single,min(pixel+vec2(0,1),maximum),0));
}
@compute @workgroup_size(8,8) fn receivers_msaa(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=config.grid.xy) {return;}
    let pixel=vec2<i32>(pixel_at(id.xy));
    let maximum=vec2<i32>(textureDimensions(depth_msaa))-vec2(1);
    // Select nearest covered sample on reverse-Z, preserving silhouettes without tracing each MSAA sample.
    var depth=0.0; var sample=0;
    for(var i=0;i<i32(config.grid.z);i++) {let d=textureLoad(depth_msaa,pixel,i);if d>depth {depth=d;sample=i;}}
    receive(id.xy,depth,textureLoad(depth_msaa,min(pixel+vec2(1,0),maximum),sample),textureLoad(depth_msaa,min(pixel+vec2(0,1),maximum),sample));
}
@compute @workgroup_size(8,8) fn mapped_normals(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=config.grid.xy) {return;}
    let index=index_at(id.xy);
    var normal=vec3(0.0,1.0,0.0);
    if receivers[index].normal!=0u {normal=unpacked(receivers[index].normal);}
    var result=Normal(vec4(normal,1.0),vec4(normal,0.0));
    if matched(index) {
        let point=receivers[index].point;
        let material=surface(intersection(hits[index]),point,length(point-frame.eye_radius.xyz)/frame.viewport.w);
        result=Normal(vec4(material.normal,material.roughness),vec4(material.geometric,material.metal));
    }
    normal_output[index]=result;
}
fn cell_at(pixel:vec2<f32>) -> vec2<u32> {return min(vec2<u32>((pixel-frame.viewport.xy)/frame.viewport.zw*vec2<f32>(config.grid.xy)),config.grid.xy-vec2(1u));}
fn material_at(index:u32) -> Surface {
    let point=receivers[index].point;
    return surface(intersection(hits[index]),point,length(point-frame.eye_radius.xyz)/frame.viewport.w);
}
@fragment fn albedo(@builtin(position) pixel:vec4<f32>) -> @location(0) vec4<f32> {
    let index=index_at(cell_at(pixel.xy));
    if receivers[index].normal==0u {return vec4(1.0);}
    let visibility=traced[index].visibility;
    let shadow=1.0-bitcast<f32>(visibility.x);
    let ao=mix(0.65,1.0,bitcast<f32>(visibility.y));
    var ratio=vec3(1.0);
    if matched(index) {
        let material=material_at(index);
        let fade=1.0-smoothstep(frame.eye_radius.w*0.75,frame.eye_radius.w,distance(receivers[index].point,frame.eye_radius.xyz));
        ratio=mix(vec3(1.0),clamp(material.colour/max(material.base,vec3(0.01)),vec3(0.0),vec3(8.0)),fade);
    }
    return vec4(ratio*shadow*ao,1.0);
}
fn fog_at(cell:vec2<u32>,distance:f32) -> vec3<f32> {
    let size=(config.grid.xy+vec2(1u))/2u;
    let coordinate=(vec2<f32>(cell)+vec2(0.5))*0.5-vec2(0.5);
    let base=vec2<i32>(floor(coordinate));
    var colour=vec3(0.0);var total=0.0;
    for(var y=0;y<2;y++) {for(var x=0;x<2;x++) {
        let p=vec2<u32>(clamp(base+vec2(x,y),vec2(0),vec2<i32>(size)-vec2(1)));
        let value=beams[p.y*size.x+p.x];
        let offset=abs(coordinate-vec2<f32>(base+vec2(x,y)));
        let weight=(1.0-offset.x)*(1.0-offset.y)*exp(-abs(value.w-distance)*2.0);
        colour+=value.rgb*weight;total+=weight;
    }}
    return colour/max(total,0.00001);
}
@fragment fn lighting(@builtin(position) pixel:vec4<f32>) -> @location(0) vec4<f32> {
    let cell=cell_at(pixel.xy);let index=index_at(cell);
    let point=receivers[index].point;
    let distance=select(frame.eye_radius.w,length(point-frame.eye_radius.xyz),receivers[index].normal!=0u);
    if !matched(index) {return vec4(0.0);}
    let material=material_at(index);
    let direction=normalize(point-frame.eye_radius.xyz);
    let result=traced[index];
    var added=material.colour*material.emission*0.8;
    for(var light=0u;light<u32(frame.ray_length.z);light++) {
        let mask=select(result.visibility.z,result.visibility.w,light>=16u);
        let visible=f32((mask>>((light%16u)*2u))&3u)*0.5;
        if visible==0.0 {continue;}
        let source=lights[light];let delta=source.position_radius.xyz-point;let range=length(delta);
        let brdf=direct_brdf(material,-direction,delta/max(range,0.001));
        added+=(brdf.diffuse+brdf.specular)*source.colour_intensity.rgb*source.colour_intensity.w*pow(max(1.0-range/source.position_radius.w,0.0),2.0)*3.14159265*visible;
    }
    let sun=frame.sun_strength.xyz;
    let relief=(max(dot(material.normal,sun),0.0)-max(dot(material.geometric,sun),0.0))*frame.sun_strength.w*result.direction_sun.w;
    added+=material.colour*max(relief,0.0)*0.25;
    if config.grid.w!=0u {added+=direct_brdf(material,-direction,sun).specular*frame.sun_strength.w*2.0*result.direction_sun.w;}
    let reflectivity=material.metal*pow(1.0-material.roughness,2.0)*0.85;
    var reflected=config.sky.rgb*mix(0.55,1.0,max(result.direction_sun.y,0.0));
    if result.reflection.w!=0u {
        let hit=intersection(result.reflection);
        let surface=surface(hit,point+material.geometric*0.025+result.direction_sun.xyz*hit.t,max(material.roughness*0.08,distance/frame.viewport.w));
        reflected=surface.colour*(0.15+0.65*frame.sun_strength.w*max(dot(surface.normal,sun),0.0)+surface.emission);
    }
    let fade=1.0-smoothstep(frame.eye_radius.w*0.75,frame.eye_radius.w,distance);
    return vec4((min(added,vec3(2.0))+reflected*material.colour*reflectivity)*fade,(reflectivity+(1.0-reflectivity)*max(-relief,0.0)*0.25)*fade);
}

@fragment fn fog_light(@builtin(position) pixel:vec4<f32>) -> @location(0) vec4<f32> {
    let cell=cell_at(pixel.xy);let index=index_at(cell);
    let distance=select(frame.eye_radius.w,length(receivers[index].point-frame.eye_radius.xyz),receivers[index].normal!=0u);
    return vec4(fog_at(cell,min(distance,frame.eye_radius.w)),0.0);
}
