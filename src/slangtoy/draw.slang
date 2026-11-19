
float4 render(int2 pos, int2 resolution)
{
    float2 uv = (float2(pos) + 0.5) / float2(resolution);
    return float4(uv, sin(uv.x * 60 + (pos.y * 9182739812731) % 30), 1.0);
}