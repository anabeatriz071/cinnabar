RaytracingAccelerationStructure Scene : register(t0);
RWByteAddressBuffer Output : register(u0);
struct Payload { float4 colour; };
[shader("raygeneration")]
void RayGen() {
    uint index = DispatchRaysIndex().x;
    RayDesc ray;
    ray.Origin = float3(index * 5.0, 0, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.01;
    ray.TMax = 10;
    Payload payload;
    payload.colour = 0;
    TraceRay(Scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    Output.Store(index * 4, payload.colour.x > 0.5 ? 0x484954 : 0x4d495353);
}
[shader("miss")]
void Miss(inout Payload payload) { payload.colour = float4(0, 0, 0, 1); }
[shader("closesthit")]
void ClosestHit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.colour = float4(1, attributes.barycentrics, 1);
}
