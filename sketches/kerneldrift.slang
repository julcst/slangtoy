import "slangpy";

void convolve(int2 pos, ITensor<float4, 2> curr, IRWTensor<float4, 2> old, float4x4[3][3] kernel)
{
    let size = int2(curr.shape[1], curr.shape[0]);

    float4 accel = 0.0;
    [unroll] for (int kx = 0; kx < 3; ++kx)
    [unroll] for (int ky = 0; ky < 3; ++ky)
    {
        int2 neighbor = pos + int2(kx, ky) - 1;
        //neighbor = clamp(neighbor, 0, size - 1);
        neighbor = (neighbor + size) % size;
        accel += mul(kernel[kx][ky], curr[neighbor]);
    }

    var velocity = (curr[pos] - old[pos]);
    velocity *= 0.99; // damping
    velocity += accel; // acceleration

    old[pos] = normalize(curr[pos] + velocity);
}

float4 shade(float4 field, float3[4] palette)
{
    float3 color = 0.0;
    [unroll]
    for (int i = 0; i < 4; ++i)
        color += field[i] * field[i] * palette[i];
    return float4(color, 1.0);
}
