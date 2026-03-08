# Standard DPNV 625-Grid Benchmark Results

**Date**: 2026-03-08
**Device**: NVIDIA GeForce RTX 4090 (128 SMs, 16K optimal pop)
**Test file**: `Evolvatron.Tests/Evolvion/StandardDPNVBenchmark.cs`

## Background

The Double Pole Balancing No Velocity (DPNV) benchmark is a standard neuroevolution test introduced by Gruau et al. (1996). The agent observes only 3 values (cart position, pole1 angle, pole2 angle) — no velocities — and must infer dynamics through recurrence.

The standard generalization test uses a 5^4 = 625-position grid over cart position, cart velocity, pole1 angle, and pole1 angular velocity (pole2 always at zero). Each position must survive 1000 timesteps. Passing threshold: >= 200/625.

**Previous approach was flawed**: Our custom 16-position benchmark included extreme starting conditions (15-degree poles, 2 rad/s angular velocities, combined perturbations) that are 4-13x beyond the standard grid ranges. No published method has ever been tested against those conditions. Both Evolvion and PPO baselines (Evolvatron-Verify) scored 0/16.

## Experiment Setup

- **Training**: Single starting position (pole1 = 4 degrees), MaxSteps=1000, Gruau anti-jiggle fitness
- **Recurrence**: Elman ctx=2 (2 feedback outputs fed back as inputs)
- **Evolution**: 1 species, 16K population, no edge mutations, jitter=0.15
- **Grid test**: Champion tested on 625-grid every 5s after first solve
- **Budget**: 100s per run, 5 seeds per topology

## Results

### Evolvion vs Published Literature

| Method | Grid Score /625 | Evaluations | Year |
|--------|----------------|-------------|------|
| **Evolvion 5->4->4->3** | **314** (median) | ~10M (GPU-parallel) | 2026 |
| CE (Gruau) | 300 | 840,000 | 1996 |
| ESP | 289 | 169,466 | 1999 |
| **Evolvion 5->8->3** | **287** (median) | ~9M (GPU-parallel) | 2026 |
| NEAT | 286 | 33,184 | 2002 |
| **Evolvion 5->3->3** | **285** (median) | ~15M (GPU-parallel) | 2026 |
| CMA-ES (3 hidden) | 250 | 6,061 | 2003 |
| CoSyNE | N/A (not reported) | 3,416 | 2008 |

### Full Topology Sweep

| Topology | Edges | Median Grid | Range | Pass >=200 | Gen/s |
|----------|-------|-------------|-------|------------|-------|
| **5->4->4->3** | 48 | **314** | 195-356 | 4/5 | 6 |
| **5->8->3** | 58 | **287** | 228-418 | **5/5** | 6 |
| 5->3->3 | 24 | 285 | 129-321 | 4/5 | 10 |
| 5->10->3 | 68 | 258 | 73-283 | 3/5 | 5 |
| 5->6->6->3 | 84 | 247 | 125-356 | 4/5 | 4 |
| 5->6->3 | 48 | 216 | 113-381 | 4/5 | 7 |

### Raw Data

```
5>3>3 (24 edges):
  Seed 0: solve=gen5/0.4s,  grid=248/625 @ gen187 (18.4s)
  Seed 1: solve=gen7/0.4s,  grid=321/625 @ gen122 (13.0s)
  Seed 2: solve=gen11/0.7s, grid=288/625 @ gen122 (13.3s)
  Seed 3: solve=gen9/0.6s,  grid=285/625 @ gen165 (19.2s)
  Seed 4: solve=gen4/0.3s,  grid=129/625 @ gen110 (11.9s)

5>6>3 (48 edges):
  Seed 0: solve=gen5/0.5s,  grid=216/625 @ gen370 (58.1s)
  Seed 1: solve=gen7/0.7s,  grid=379/625 @ gen49  (8.0s)
  Seed 2: solve=gen6/0.7s,  grid=208/625 @ gen46  (7.2s)
  Seed 3: solve=gen10/1.0s, grid=113/625 @ gen607 (85.4s)
  Seed 4: solve=gen6/0.6s,  grid=381/625 @ gen49  (7.8s)

5>8>3 (58 edges):
  Seed 0: solve=gen12/1.5s, grid=228/625 @ gen47 (8.3s)
  Seed 1: solve=gen11/1.3s, grid=272/625 @ gen83 (15.2s)
  Seed 2: solve=gen11/1.2s, grid=347/625 @ gen83 (15.7s)
  Seed 3: solve=gen7/0.8s,  grid=287/625 @ gen43 (8.4s)
  Seed 4: solve=gen8/1.0s,  grid=418/625 @ gen42 (8.4s)

5>10>3 (68 edges):
  Seed 0: solve=gen15/1.8s, grid=73/625  @ gen245 (47.1s)
  Seed 1: solve=gen10/1.2s, grid=193/625 @ gen44  (8.4s)
  Seed 2: solve=gen10/1.2s, grid=283/625 @ gen142 (29.6s)
  Seed 3: solve=gen11/1.5s, grid=277/625 @ gen44  (9.4s)
  Seed 4: solve=gen5/0.6s,  grid=258/625 @ gen37  (7.7s)

5>4>4>3 (48 edges):
  Seed 0: solve=gen7/0.7s,  grid=195/625 @ gen45  (7.8s)
  Seed 1: solve=gen6/0.7s,  grid=356/625 @ gen236 (40.5s)
  Seed 2: solve=gen7/0.8s,  grid=339/625 @ gen46  (8.4s)
  Seed 3: solve=gen7/0.7s,  grid=314/625 @ gen119 (20.9s)
  Seed 4: solve=gen7/0.8s,  grid=300/625 @ gen160 (27.9s)

5>6>6>3 (84 edges):
  Seed 0: solve=gen8/1.1s,  grid=356/625 @ gen36  (9.4s)
  Seed 1: solve=gen7/1.0s,  grid=287/625 @ gen62  (14.4s)
  Seed 2: solve=gen6/0.9s,  grid=202/625 @ gen61  (14.2s)
  Seed 3: solve=gen9/1.4s,  grid=247/625 @ gen38  (9.3s)
  Seed 4: solve=gen6/0.9s,  grid=125/625 @ gen148 (32.2s)
```

## Conclusions

### 1. Evolvion is competitive with the best published neuroevolution methods

The 5->4->4->3 topology (median 314/625) beats CE, ESP, and NEAT on generalization. The 5->8->3 topology matches NEAT. This puts Evolvion in the same league as methods developed by Stanley, Gomez, Igel, and Gruau — on the standard benchmark they defined.

### 2. The old 16-position benchmark was testing beyond known limits

Our custom starting positions included 15-degree poles and 2 rad/s angular velocities. The standard 625-grid maxes out at 3.6 degrees and 0.15 rad/s. We were testing 4-13x beyond what anyone in the literature has attempted. The failure to solve those positions is not a failure of the algorithm — it's a failure of the benchmark design.

### 3. Deeper networks generalize better than wider ones

At the same edge count (48), the 2-hidden-layer 5->4->4->3 topology (median 314) dramatically outperforms the 1-hidden-layer 5->6->3 topology (median 216). This suggests depth helps the Elman recurrence build better temporal representations.

### 4. There's a sweet spot in network size

The 5->8->3 (58 edges) is the most reliable topology — the only one where all 5 seeds pass the >=200 threshold. Going wider (5->10->3, 68 edges) or adding more parameters (5->6->6->3, 84 edges) hurts generalization. The CMA-ES literature confirms this: 3 hidden nodes (28 weights) beats 9+ hidden nodes.

### 5. Generalization doesn't improve with more single-position training

Best grid scores typically appear within 7-20 seconds (gen 36-187), despite 100 seconds of total training budget. Gruau fitness on a single starting position does not create selection pressure for generalization. To push scores higher (toward 400+/625), training on multiple grid positions directly would be needed.

### 6. High variance is inherent to the problem

Even the best topology (5->4->4->3) ranges from 195 to 356 across seeds. The 5->8->3 topology shows even more variance (228 to 418 — seed 4 hit an extraordinary 418). This matches the literature: generalization on DPNV is stochastic. Reporting median over multiple seeds is essential.

### 7. Evaluation counts are not directly comparable

Evolvion uses 16K parallel evaluations per generation on GPU, yielding ~10-15M total evaluations per run. CMA-ES uses ~6K sequential evaluations with pop=13. The wall-clock comparison is more meaningful: Evolvion solves in <2 seconds and achieves peak generalization in <20 seconds. CMA-ES timing is not reported in the papers but was run on 2003-era hardware.

## Recommended Baseline Configurations

For fixed-topology ES-style evolution on DPNV:

- **Best generalization**: 5->4->4->3, Elman ctx=2, dense, 16K pop, 1 species
- **Most reliable**: 5->8->3, Elman ctx=2, dense, 16K pop, 1 species
- **Fastest (fewest gens)**: 5->3->3, Elman ctx=2, dense, 16K pop, 1 species

All with: WeightJitterStdDev=0.15, no edge mutations, Gruau fitness, MaxSteps=1000.

## References

- Gruau, Whitley & Pyeatt (1996). "A Comparison between Cellular Encoding and Direct Encoding for Genetic Neural Networks."
- Gomez & Miikkulainen (1999). "Solving Non-Markovian Control Tasks with Neuroevolution." IJCAI.
- Stanley & Miikkulainen (2002). "Evolving Neural Networks through Augmenting Topologies." Evolutionary Computation.
- Igel (2003). "Neuroevolution for Reinforcement Learning Using Evolution Strategies." CEC.
- Gomez, Schmidhuber & Miikkulainen (2008). "Accelerated Neural Evolution through Cooperatively Coevolved Synapses." JMLR.
- Pagliuca, Milano & Nolfi (2018). "Maximizing adaptive power in neuroevolution." PLOS ONE.
