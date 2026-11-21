# Square World

Run with `uv run slangtoy`. Press **Space** to pause or **R** to reset the field.

The **Simulation** panel sets the advance rate and collapse chance per second.
**Steps per frame** divides the elapsed frame time into smaller updates. More
steps increase GPU work while keeping the advance budget and cumulative
collapse chance for the same elapsed time. **Reset field** draws independent
standard-normal values. **Reseed kernel** changes how the current field evolves.
**Randomize palette** draws new display colors.

A random kernel mixes signed values across a wrapping grid. **Probability**
switches the current field between **Root**, which uses normalized squared
amplitudes, and **Logit**, which uses softmax. An all-zero cell uses equal
probabilities in both modes. These probabilities drive collapse sampling and
palette blending. Collapse gives the sampled channel 99% before the next
normalization pass and divides the rest equally. Root mode also preserves the
signs of nonzero amplitudes during collapse.

**Normalize energy** applies one scale factor to the entire field before
collapse sampling and after each step. The target is the sum of squares across
all cells and channels in the most recent reset field.
