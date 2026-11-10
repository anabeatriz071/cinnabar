RWByteAddressBuffer Output : register(u0);
struct Payload { float4 colour; };
[shader("raygeneration")]
void RayGen() { Output.Store(0, 0x43585254); }
[shader("miss")]
void Miss(inout Payload payload) { payload.colour = float4(0, 0, 0, 1); }
[shader("closesthit")]
void ClosestHit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.colour = float4(attributes.barycentrics, 0, 1);
}
