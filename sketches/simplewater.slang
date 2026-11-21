import "slangpy";

static const float KERNEL[3][3] = {
    {1./6., 4./6., 1./6.},
    {4./6., -20./6., 4./6.},
    {1./6., 4./6., 1./6.},
};

void waveprop(int2 pos, ITensor<float, 2> curr, IRWTensor<float, 2> old, float dt)
{
    let size = int2(curr.shape[1], curr.shape[0]);

    float result = 0.0;
    [unroll] for (int x = 0; x < 3; ++x)
    [unroll] for (int y = 0; y < 3; ++y)
    {
        int2 neighbor = pos + int2(x, y) - 1;
        neighbor = clamp(neighbor, 0, size - 1);
        //neighbor = (neighbor + size) % size;
        result += KERNEL[x][y] * curr[neighbor];
    }

    float velocity = (curr[pos] - old[pos]);

    velocity *= 0.99; // damping
    velocity += result * 0.2; // acceleration

    old[pos] = curr[pos] + velocity;
}

float4 shade(int2 pos, ITensor<float, 2> field)
{
    //float value = 0.5 + 0.5 * field[pos];
    //return float4(value, value, value, 1.0);

    let max = int2(field.shape[1], field.shape[0]) - 1;
    let dhdx = field[clamp(pos + int2(1, 0), 0, max)] - field[clamp(pos - int2(1, 0), 0, max)];
    let dhdy = field[clamp(pos + int2(0, 1), 0, max)] - field[clamp(pos - int2(0, 1), 0, max)];
    let normal = normalize(float3(-dhdx, -dhdy, 1.0));
    let light = normalize(float3(-0.5, -0.5, 1.0));
    let diffuse = dot(normal, light);
    let specular = pow(dot(reflect(-light, normal), float3(0.0, 0.0, 1.0)), 16.0);

    let color = float3(0.0, 0.2, 0.4) * diffuse + float3(1.0, 1.0, 1.0) * specular;
    return float4(color, 1.0);
}
