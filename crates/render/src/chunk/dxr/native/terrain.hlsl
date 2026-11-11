// Native DXR intersections. Material textures are evaluated in the adjacent deferred passes.
cbuffer Frame : register(b0) {
    float4x4 WorldFromClip;
    float4 EyeRadius;
    float4 SunStrength;
    float4 Viewport;
    float4 RayLength;
    uint4 InvalidInstances;
    uint4 Grid;
    float4 Fog;
    float4 Sky;
};
RaytracingAccelerationStructure Scene : register(t0);
ByteAddressBuffer Receivers : register(t1);
ByteAddressBuffer MaterialNormals : register(t2);
ByteAddressBuffer Lights : register(t3);
RWByteAddressBuffer PrimaryHits : register(u0);
RWByteAddressBuffer Lighting : register(u1);
RWByteAddressBuffer FogLight : register(u2);
struct Hit { uint instance; uint primitive; float distance; uint kind; };
uint indexOf(uint2 pixel) { return pixel.y * Grid.x + pixel.x; }
float3 normalOf(uint packed) {
    return normalize(float3(packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023) * (2.0 / 1023.0) - 1.0);
}
Hit trace(float3 origin, float3 direction, float maximum, bool first) {
    Hit result = (Hit)0;
    uint flags = first ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE;
    flags |= any(InvalidInstances != 0) ? RAY_FLAG_FORCE_NON_OPAQUE : RAY_FLAG_FORCE_OPAQUE;
    RayDesc ray;
    ray.Origin = origin; ray.Direction = direction; ray.TMin = 0.005; ray.TMax = max(maximum, 0.006);
    TraceRay(Scene, flags, 0xff, 0, 1, 0, ray, result);
    return result;
}
float visibility(float3 positionWorld, float3 direction, float maximum) { return trace(positionWorld, direction, maximum, true).kind == 0 ? 1.0 : 0.0; }
float sunVisibility(float3 positionWorld) {
    float3 sun = normalize(SunStrength.xyz);
    float3 axis = abs(sun.y) > 0.99 ? float3(1,0,0) : float3(0,1,0);
    float3 tangent = normalize(cross(sun, axis));
    float3 bitangent = cross(sun, tangent);
    float3 diagonal = (tangent + bitangent) * (RayLength.y * 0.5);
    float a = visibility(positionWorld, normalize(sun + diagonal), RayLength.x);
    float b = visibility(positionWorld, normalize(sun - diagonal), RayLength.x);
    if (a == b) return a;
    float3 other = (tangent - bitangent) * (RayLength.y * 0.5);
    return (a + b + visibility(positionWorld, normalize(sun + other), RayLength.x) + visibility(positionWorld, normalize(sun - other), RayLength.x)) * 0.25;
}
[shader("miss")] void Miss(inout Hit hit) { hit.kind = 0; }
[shader("closesthit")] void ClosestHit(inout Hit hit, BuiltInTriangleIntersectionAttributes attributes) {
    hit.instance = InstanceID(); hit.primitive = PrimitiveIndex(); hit.distance = RayTCurrent(); hit.kind = 1;
}
[shader("anyhit")] void FilterWithdrawn(inout Hit hit, BuiltInTriangleIntersectionAttributes attributes) {
    uint id = InstanceID();
    if ((InvalidInstances[id / 32] & (1u << (id % 32))) != 0) IgnoreHit();
}
[shader("raygeneration")] void Primary() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint index = indexOf(pixel);
    uint4 receiver = Receivers.Load4(index * 16);
    Hit hit = (Hit)0;
    if ((receiver.w & 0x80000000u) != 0) {
        float3 positionWorld = asfloat(receiver.xyz);
        float3 delta = positionWorld - EyeRadius.xyz;
        float distance = length(delta);
        if (distance > 0.02 && distance < EyeRadius.w) hit = trace(EyeRadius.xyz, delta / distance, distance + 0.08, false);
    }
    PrimaryHits.Store4(index * 16, uint4(hit.instance, hit.primitive, asuint(hit.distance), hit.kind));
}
[shader("raygeneration")] void Secondary() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint index = indexOf(pixel);
    uint4 receiver = Receivers.Load4(index * 16);
    float shadow = 0, ao = 1, sun = 1;
    uint low = 0, high = 0;
    Hit reflection = (Hit)0;
    float3 reflectionDirection = 0;
    if ((receiver.w & 0x80000000u) != 0) {
        float3 positionWorld = asfloat(receiver.xyz);
        float4 mapped = asfloat(MaterialNormals.Load4(index * 32));
        float4 detail = asfloat(MaterialNormals.Load4(index * 32 + 16));
        float3 geometric = detail.xyz;
        float3 normal = normalize(mapped.xyz);
        float3 origin = positionWorld + geometric * 0.025;
        float distance = length(positionWorld - EyeRadius.xyz);
        float fade = 1 - smoothstep(EyeRadius.w * 0.75, EyeRadius.w, distance);
        if (Grid.w != 0 && dot(geometric, SunStrength.xyz) > 0) {
            sun = sunVisibility(origin);
            shadow = (1 - sun) * SunStrength.w * Sky.w * smoothstep(0, 0.2, dot(geometric, SunStrength.xyz)) * fade;
        }
        float3 axis = abs(geometric.y) > 0.99 ? float3(1,0,0) : float3(0,1,0);
        float3 tangent = normalize(cross(geometric, axis));
        float3 bitangent = cross(geometric, tangent);
        float occlusion = 0;
        [unroll] for (uint probe = 0; probe < 4; probe++) {
            float3 ray = normalize(geometric * 0.8 + tangent * ((probe & 1) != 0 ? 0.6 : -0.6) + bitangent * ((probe & 2) != 0 ? 0.6 : -0.6));
            Hit hit = trace(origin, ray, 2, false);
            if (hit.kind != 0) occlusion += 1 - saturate(hit.distance * 0.5);
        }
        ao = 1 - occlusion * 0.25 * fade;
        uint evaluated = 0;
        [loop] for (uint light = 0; light < uint(RayLength.z); light++) {
            float4 position = asfloat(Lights.Load4(light * 32));
            float3 delta = position.xyz - positionWorld;
            float range = length(delta);
            if (range < 0.03 || range >= position.w || dot(normal, delta / range) <= 0) continue;
            float3 side = normalize(cross(delta / range, abs(delta.y / range) > 0.99 ? float3(1,0,0) : float3(0,1,0))) * 0.035;
            float3 a = position.xyz + side - origin, b = position.xyz - side - origin;
            uint visible = uint(visibility(origin, normalize(a), length(a) - 0.08)) + uint(visibility(origin, normalize(b), length(b) - 0.08));
            if (light < 16) low |= visible << (light * 2); else high |= visible << ((light - 16) * 2);
            if (++evaluated == 4) break;
        }
        if (detail.w * pow(1 - mapped.w, 2) > 0.025) {
            float3 incoming = normalize(positionWorld - EyeRadius.xyz);
            float3 reflected = reflect(incoming, normal);
            reflectionDirection = normalize(reflected + geometric * max(0.02 - dot(reflected, geometric), 0));
            reflection = trace(origin, reflectionDirection, RayLength.x, false);
        }
    }
    Lighting.Store4(index * 48, uint4(asuint(shadow), asuint(ao), low, high));
    Lighting.Store4(index * 48 + 16, uint4(reflection.instance, reflection.primitive, asuint(reflection.distance), reflection.kind));
    Lighting.Store4(index * 48 + 32, uint4(asuint(reflectionDirection), asuint(sun)));
}
[shader("raygeneration")] void Beams() {
    uint2 cell = DispatchRaysIndex().xy;
    uint2 size = (Grid.xy + 1) / 2;
    uint2 pixel = min(cell * 2 + 1, Grid.xy - 1);
    uint4 receiver = Receivers.Load4(indexOf(pixel) * 16);
    float2 uv = (float2(pixel) + 0.5) / Grid.xy;
    float4 nearPoint = mul(WorldFromClip, float4(uv.x * 2 - 1, 1 - uv.y * 2, 1, 1));
    float3 ray = normalize(nearPoint.xyz / nearPoint.w - EyeRadius.xyz);
    float lengthLimit = EyeRadius.w;
    if ((receiver.w & 0x80000000u) != 0) lengthLimit = min(lengthLimit, length(asfloat(receiver.xyz) - EyeRadius.xyz));
    float3 scattering = 0;
    float stepLength = lengthLimit / 6;
    [unroll] for (uint step = 0; step < 6; step++) {
        float distance = (float(step) + 0.5) * stepLength;
        float3 positionWorld = EyeRadius.xyz + ray * distance;
        float3 incident = 0;
        if (Grid.w != 0) {
            float cosine = dot(ray, normalize(SunStrength.xyz));
            float phase = 0.75 / pow(max(1.25 - cosine, 0.08), 1.5);
            incident += Sky.rgb * SunStrength.w * phase * visibility(positionWorld, normalize(SunStrength.xyz), RayLength.x);
        }
        uint evaluated = 0;
        [loop] for (uint light = 0; light < uint(RayLength.z); light++) {
            float4 position = asfloat(Lights.Load4(light * 32));
            float4 colour = asfloat(Lights.Load4(light * 32 + 16));
            float3 delta = position.xyz - positionWorld;
            float range = length(delta);
            if (range < 0.08 || range >= position.w) continue;
            incident += colour.rgb * colour.w * pow(1 - range / position.w, 2) * visibility(positionWorld, delta / range, range - 0.08);
            if (++evaluated == 2) break;
        }
        scattering += incident * Fog.rgb * exp(-Fog.w * distance) * Fog.w * stepLength * 0.5;
    }
    FogLight.Store4((cell.y * size.x + cell.x) * 16, uint4(asuint(min(scattering, 1.5)), asuint(lengthLimit)));
}
