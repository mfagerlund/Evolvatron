# Phase 7 Recommended Defaults (CORRECTED - Post-Bug-Fix)

**Date**: 2025-11-01 (Updated after density bug fix)
**Based on**: Comprehensive hyperparameter sweep + corrected sparse/density validation

---

## ⚠️ CRITICAL CORRECTION

**Original Phase 7 claimed "dense initialization required" - THIS WAS WRONG!**

A bug in `InitializeDense()` caused densities 0.85, 0.95, and 1.0 to produce IDENTICAL networks.
After fixing the bug, **moderately sparse (0.7-0.85) actually BEATS fully dense by 25-37%!**

See: `CRITICAL-BUGS-TO-FIX.md` for full details.

---

## ⚠️ ADDITIONAL CRITICAL UPDATE (Re-validation Required)

**ALL RECOMMENDATIONS BELOW MAY NEED RE-VALIDATION!**

After the density bug fix, **THREE additional critical bugs** were discovered and fixed (commit 641033a):

1. **Zero bias initialization** - All biases were initialized to 0 (no diversity)
2. **Order-dependent edge sampling** - Sequential iteration bias in InitializeDense
3. **Shared Random instance** - All species shared same Random (reduced diversity)

**Impact**: These fixes will change ALL evolutionary dynamics. The architecture and mutation rate recommendations below were derived with buggy code and may no longer be optimal.

**Status**: Phase 7 defaults are **PROVISIONAL** pending Phase 8 full re-validation.

See `CRITICAL-BUGS-TO-FIX.md` Bug #3 for details.

---

## Recommended Default Configuration

### Architecture: Ultra-Deep Narrow (15 layers × 2 nodes)

```csharp
// WINNER: 2→(2 nodes × 15 layers)→1
// Provides +63.4% improvement over 6×3 baseline
var topology = new SpeciesBuilder()
    .AddInputRow(2)
    .AddHiddenRow(2, ActivationType.Tanh, count: 15)  // 15 layers with count parameter!
    .AddOutputRow(1, ActivationType.Tanh)
    .WithMaxInDegree(6)
    .InitializeDense(random, density: 0.85f)  // CORRECTED: Moderately sparse wins!
    .Build();
```

**Key insights**: 
- DEPTH >> WIDTH for this architecture
- **Moderately sparse (0.7-0.85) beats fully dense** (bug fix discovery)

---

## Corrected Sparse vs Dense Results

**Critical question**: Does moderately sparse initialization work?
**Answer**: YES - moderately sparse (0.85) is BEST!

| Density | Final Fitness | vs Dense (1.0) | Status |
|---------|---------------|----------------|--------|
| **0.85** | -0.6898 | **137%** 🥇 | BEST |
| 0.7 | -0.7173 | 125% | Excellent |
| 0.5 | -0.7149 | 125% | Excellent |
| **1.0 (Dense)** | -0.7669 | 100% | Baseline |
| 0.95 | -0.7669 | 100% | Same as 1.0 |
| 0.3 | -0.7841 | 93% | Worse |
| 0.2 | -0.7841 | 91% | Worse |
| 0.1 | -0.8763 | 61% | Much worse |

**REVISED Key findings**:
- **Sweet spot: 0.5-0.85 density (all beat fully dense!)**
- Overparameterization hurts evolution
- Sparser networks have better gradient landscape
- Very sparse (<0.3) still fails (insufficient connectivity)

**RECOMMENDATION**: Use `density: 0.85f` for best results.

---

## See Also

- `scratch/CRITICAL-BUGS-TO-FIX.md` - Full bug analysis and validation
- `scratch/COMPREHENSIVE-SWEEP-RESULTS.md` - Full sweep results (partially invalidated)
