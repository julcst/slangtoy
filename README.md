# Logit World

Run with `uv run slangtoy`. Press **Space** to pause or **R** to reset the field.

The **Simulation** panel controls energy normalization, the collapse probability
limit, and the advance rate. **Steps per frame** divides the advance rate and
collapse probability across smaller updates. More steps increase GPU work.
**Reset field** draws independent standard-normal logits. **Reseed kernel**
changes how the current field evolves. **Randomize palette** draws new display
colors.

A random kernel mixes channels across a wrapping grid. With normalization on,
each step restores the channels' sum of squared logits before and after
collapse. PCG2D selects cells and draws a dominant channel from their softmax.
The display blends the palette using the same softmax weights.
