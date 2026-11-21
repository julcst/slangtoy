import "slangpy";

typealias Field = ITensor<float, 3>;

static const int2 neighbors[8] = {
    int2(-1, -1), int2(0, -1), int2(1, -1), int2(-1, 0),
    int2(1, 0), int2(-1, 1), int2(0, 1), int2(1, 1),
};

// Jarzynski and Olano, Hash Functions for GPU Rendering (JCGT 2020).
uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;
    [unroll]
    for (int i = 0; i < 2; ++i)
    {
        v.x += v.y * 1664525u;
        v.y += v.x * 1664525u;
        v ^= v >> 16u;
    }
    return v;
}

float2 random2(uint2 key)
{
    return float2(pcg2d(key) >> 8u) / 16777216.0;
}

float advance(int3 id, Field field, Field kernel, float rate)
{
    int3 pos = id.zyx;
    int2 size = int2(field.shape[0], field.shape[1]);
    float momentum = 0.0;
    for (int n = 0; n < kernel.shape[0]; ++n)
    {
        int2 p = (pos.xy + neighbors[n] + size) % size;
        for (int c = 0; c < field.shape[2]; ++c)
        {
            float weight = kernel[n, pos.z, c] + (c == pos.z ? 0.08 : 0.0);
            momentum += weight * (field[p.x, p.y, c] - field[pos.x, pos.y, c]);
        }
    }
    return field[pos.x, pos.y, pos.z] + rate * tanh(momentum);
}

float sumBlock(int3 id, Field field, bool square)
{
    int3 start = id.zyx * int3(8, 8, 1);
    int2 end = min(start.xy + 8, int2(field.shape[0], field.shape[1]));
    float sum = 0.0;
    for (int y = start.x; y < end.x; ++y)
        for (int x = start.y; x < end.y; ++x)
        {
            float value = field[y, x, start.z];
            sum += square ? value * value : value;
        }
    return sum;
}

float energyScale(Field energy, float initialEnergy)
{
    float total = 0.0;
    for (int c = 0; c < energy.shape[2]; ++c)
        total += energy[0, 0, c];
    return total > 0.0 ? sqrt(initialEnergy / total) : 1.0;
}

float normalize(int3 id, Field field, Field energy, float initialEnergy)
{
    return field[id.z, id.y, id.x] * energyScale(energy, initialEnergy);
}

float cellPeak(Field field, int2 pos)
{
    float peak = field[pos.x, pos.y, 0];
    for (int c = 1; c < field.shape[2]; ++c)
        peak = max(peak, field[pos.x, pos.y, c]);
    return peak;
}

float probabilityWeight(float value, bool root, float peak)
{
    return root ? value * value : exp(value - peak);
}

float collapse(int3 id, Field field, Field energy, float initialEnergy,
               bool normalized, bool root, uint seed, float rate)
{
    int3 pos = id.zyx;
    uint2 frameKey = pcg2d(uint2(seed, 0u));
    float2 random = random2(uint2(pos.xy) + frameKey);
    float scale = normalized ? energyScale(energy, initialEnergy) : 1.0;
    float value = field[pos.x, pos.y, pos.z] * scale;
    if (random.x >= rate)
        return value;

    int channels = int(field.shape[2]);
    float peak = root ? 0.0 : cellPeak(field, pos.xy) * scale;
    float total = 0.0, lower = 0.0, upper = 0.0;
    for (int c = 0; c < channels; ++c)
    {
        if (c == pos.z)
            lower = total;
        total += probabilityWeight(field[pos.x, pos.y, c] * scale, root, peak);
        if (c == pos.z)
            upper = total;
    }
    if (total == 0.0)
    {
        total = float(channels);
        lower = float(pos.z);
        upper = lower + 1.0;
    }
    float sample = random.y * total;
    // The final nonempty interval also handles a sample rounded up to total.
    bool winner = upper > lower && sample >= lower && (sample < upper || upper == total);
    if (!root)
    {
        float gap = log(99.0 * max(channels - 1, 1));
        return (float(winner) - 1.0 / channels) * gap;
    }
    float probability = winner ? (channels == 1 ? 1.0 : 0.99) : 0.01 / max(channels - 1, 1);
    float magnitude = sqrt(channels * probability);
    return value < 0.0 ? -magnitude : magnitude;
}

float4 render(int2 pos, int2 resolution, Field field, ITensor<float, 2> palette, bool root)
{
    float2 size = float2(field.shape[1], field.shape[0]);
    float2 fit = float2(resolution) / size;
    float2 uv = (float2(pos) + 0.5 - float2(resolution) * 0.5)
                / (size * min(fit.x, fit.y)) + 0.5;
    if (any(uv < 0.0) || any(uv >= 1.0))
        return float4(0.008, 0.009, 0.013, 1.0);

    int2 p = int2(uv * size).yx;
    float peak = root ? 0.0 : cellPeak(field, p);
    float total = 0.0;
    float3 color = 0.0;
    for (int c = 0; c < field.shape[2]; ++c)
    {
        float weight = probabilityWeight(field[p.x, p.y, c], root, peak);
        color += weight * float3(palette[c, 0], palette[c, 1], palette[c, 2]);
        total += weight;
    }
    if (total == 0.0)
    {
        for (int c = 0; c < field.shape[2]; ++c)
            color += float3(palette[c, 0], palette[c, 1], palette[c, 2]);
        total = float(field.shape[2]);
    }
    return float4(color / total, 1.0);
}
