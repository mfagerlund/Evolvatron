# Algorithm Selection Results

## Winner: CEM (Cross-Entropy Method)

**Flagship algorithm for Evolvatron.** Diagonal Gaussian search distribution updated from elites.

### Definitive Benchmark (10 seeds, 120s budget, 5->4->4->3 topology, 59 params)

| Algorithm | Median | Range | Pass>=200 | Median Gens |
|-----------|--------|-------|-----------|-------------|
| **CEM multi-pos(25)** | **338/625** | **293-353** | **10/10** | 667 |
| ES multi-pos(25) | 316/625 | 216-395 | 10/10 | 565 |
| CEM multi-pos(10) | 307/625 | 175-362 | 9/10 | 1089 |
| GA 3sp multi-pos(10) | 250/625 | 184-318 | 8/10 | 1163 |

CEM multi-pos(25) wins on median, range tightness, and reliability.

### Why CEM Wins

1. **Highest median AND tightest range** — most consistent algorithm
2. **Final champion directly deployable** — multi-pos prevents sigma collapse
3. **Simple to tune** — InitSigma, EliteFraction, smoothing (all well-understood)
4. **Deterministic convergence** — same inputs, same trajectory
5. **25 positions is the sweet spot** — more signal, still fast enough for 2x more generations than ES

### CEM Config (Optimal)

```
InitSigma=0.25, MinSigma=0.08, MaxSigma=2.0
EliteFraction=0.01, SigmaSmoothing=0.3, MuSmoothing=0.2
MultiPos=25, 1 island, topology 5->4->4->3 (59 params)
```

## Fallback: ES (OpenAI Evolution Strategies)

Antithetic sampling around mu, Adam gradient update. Competitive but more variable.

- **Config**: ESSigma=0.05, LR=0.05, Beta1=0.80, Beta2=0.999
- Adam momentum is critical (Beta1=0 completely fails)
- Needs 25 training positions (same as CEM sweet spot)
- Use case: harder problems where gradient signal might help

## Eliminated Algorithms

### GA (Tournament + Jitter)
- Multi-pos **hurts** GA — jitter already provides implicit regularization
- Same-topology species add no value (3sp=271/625, 160sp=258/625)
- NEAT's species worked because of topology diversity, not weight diversity

### CMA-ES
- O(n^2) memory per species, O(n^3) eigendecomposition
- Would need ~1024 species to fill GPU — impractical
- For n=59 params, per-parameter sigma is sufficient

## Eliminated Techniques

| Technique | CEM Effect | ES Effect |
|-----------|-----------|-----------|
| Pre-activation scaling (1/sqrt(fan_in)) | -21% (321->252) | Dead (0/625) |
| Real layer norm | -7% (324->300) | Not tested |
| L2 weight decay | Hurts (all WD>0 worse) | Hurts |

Evolution uses the full scale/distribution of activations. Normalizing removes information.

## Multi-Position Training (Key Insight)

| Algorithm | Single-pos | Multi-pos | Effect |
|-----------|-----------|-----------|--------|
| CEM | 112/625 | 303-338/625 | Massive win — prevents sigma collapse |
| ES | 21/625 | 282-316/625 | Massive win — gradient needs diverse signal |
| GA | 314/625 | 250-279/625 | Hurts — slows gens without proportional benefit |

Multi-pos is **mandatory** for distribution-based methods (CEM/ES). Always >= 5 positions.

## GPU Population Context

Running 16K population per generation is extremely large vs. literature norms (NES: 200-1000, CMA-ES: 50-500, CEM: 100-2000). This massive parallelism makes simple algorithms competitive with sophisticated ones — likely why CEM already performs extremely well.

## SNES Evaluation (2026-03-10)

SNES (Separable Natural Evolution Strategies) was tested as a potential CEM replacement. Uses natural gradient with log-space sigma adaptation and rank-based fitness shaping.

### SNES Eta Sweep (3 seeds, 500 gens)

Best config: **etaMu=2.0, etaSigma=0.1** (avg 302/625). Pattern: higher etaMu better (large population makes gradient estimate accurate), lower etaSigma better (conservative sigma adaptation).

### Generation-Matched Head-to-Head (10 seeds, 700 gens, multi-pos(25))

| Algorithm | Median | Mean | Range | Pass>=200 |
|-----------|--------|------|-------|-----------|
| **CEM** | **340/625** | **334/625** | **294-355** | **10/10** |
| SNES | 327/625 | 321/625 | 301-347 | 10/10 |
| SNES+mirrored | 326/625 | 324/625 | 296-343 | 10/10 |

### Why CEM Beats SNES at 16K Population

CEM's "fit to top 1% elites" is extremely effective with 16K samples — 164 elite individuals give a precise estimate of the optimal direction. SNES uses the entire population with rank-based weights, diluting the signal from the best solutions. At small populations (100-1000), SNES's approach is more stable. But at 16K, CEM's elite focus is a feature, not a limitation.

### Why Mirrored Sampling Didn't Help

With 16K population, each generation already has enormous sample diversity. Mirrored sampling's variance reduction is most valuable at small populations. At 16K, the gradient estimate is already very accurate.

### Verdict

SNES is competitive (10/10 reliability) but does not beat CEM. CEM remains flagship. SNES is available as a third option alongside ES but offers no advantage on our hardware.

## Tested Improvements Summary

| Improvement | Status | Result |
|-------------|--------|--------|
| SNES (natural gradient) | Tested | CEM still wins at 16K pop |
| Mirrored sampling | Tested | No effect at 16K pop |
| Rank-based utilities | Built into SNES | Good but CEM's elite focus is better at scale |
| Weighted elites for CEM | Not tested | Low priority given CEM already wins |
| Restart strategy | Already have | Island stagnation replacement |
| Evolution path sigma | Not tested | Low priority |
| Fitness shaping for CEM | Not tested | Low priority |

## Remaining Potential Improvements

These have not been tested but are lower priority given CEM's strong performance:

- **Weighted elites for CEM** — replace uniform elite averaging with rank-based weights (free, zero cost)
- **Evolution path sigma control** — track smoothed search direction for sigma adaptation (O(n))
- **Quality-Diversity / MAP-Elites** — maintain archive of diverse behaviors (overkill for current problem)

## Theoretical Notes

CEM, ES, and NES all approximate the same objective. The differences are how the gradient is estimated:

| Algorithm | Gradient Estimation |
|-----------|-------------------|
| CEM | Elite maximum likelihood |
| ES | Score function estimator |
| NES/SNES | Natural gradient |

For ~50-100 parameter networks, diagonal methods (CEM, SNES) match or beat full-covariance methods (CMA-ES). With 59 parameters, per-parameter sigma captures the essential degrees of freedom without the estimation noise of a 59x59 covariance matrix.

At very large populations (16K), simple elite-based methods (CEM) outperform gradient-based methods (ES, SNES) because the elite sample is large enough for precise distribution fitting. The theoretical advantages of natural gradients are most relevant at small population sizes where sample efficiency matters.
