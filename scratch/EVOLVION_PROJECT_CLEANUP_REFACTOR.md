# Evolvion Project Cleanup Refactor

## Problem

`Evolvatron.Evolvion` currently ships test environments and benchmarks that should not be part of the core library:
- `Environments/` (SpiralEnvironment, CartPoleEnvironment, etc.)
- `Benchmarks/` (LandscapeNavigationTask, OptimizationLandscapes)

This violates separation of concerns and bloats the production library with testing code.

## Goal

Move all test/benchmark code to `Evolvatron.Tests` while keeping `Evolvatron.Evolvion` as a pure algorithm library.

## What Stays in Evolvatron.Evolvion (Core Library)

✅ Keep:
- `IEnvironment.cs` - Interface contract
- `SpeciesSpec.cs`, `Individual.cs`, `Population.cs` - Data structures
- `Evolver.cs` - Main evolution loop
- `EvolutionConfig.cs` - Configuration
- `Mutation/` - All mutation operators
- `Selection/` - Selection logic
- `Culling/` - Culling logic
- `Evaluation/SimpleFitnessEvaluator.cs` - Fitness evaluation orchestration
- `Evaluation/CPUEvaluator.cs` - Network forward pass
- `SpeciesBuilder.cs` - Topology construction API

## What Moves to Evolvatron.Tests

❌ Move to `Evolvatron.Tests/Evolvion/`:
- `Environments/SpiralEnvironment.cs`
- `Environments/CartPoleEnvironment.cs` (if exists)
- `Environments/FollowTheCorridorEnvironment.cs` (if exists)
- `Benchmarks/LandscapeNavigationTask.cs`
- `Benchmarks/OptimizationLandscapes.cs`

## Step-by-Step Refactor

### Step 1: Create Target Directories
```bash
cd C:\Dev\Evolvatron\Evolvatron.Tests\Evolvion
mkdir Environments
mkdir Benchmarks
```

### Step 2: Move Files Using Git
```bash
cd C:\Dev\Evolvatron

# Move environments
git mv Evolvatron.Evolvion/Environments/SpiralEnvironment.cs \
        Evolvatron.Tests/Evolvion/Environments/

git mv Evolvatron.Evolvion/Environments/CartPoleEnvironment.cs \
        Evolvatron.Tests/Evolvion/Environments/ || true

git mv Evolvatron.Evolvion/Environments/FollowTheCorridorEnvironment.cs \
        Evolvatron.Tests/Evolvion/Environments/ || true

# Move benchmarks
git mv Evolvatron.Evolvion/Benchmarks/LandscapeNavigationTask.cs \
        Evolvatron.Tests/Evolvion/Benchmarks/

git mv Evolvatron.Evolvion/Benchmarks/OptimizationLandscapes.cs \
        Evolvatron.Tests/Evolvion/Benchmarks/

# Remove empty directories
rmdir Evolvatron.Evolvion/Environments
rmdir Evolvatron.Evolvion/Benchmarks
```

### Step 3: Update Namespaces

**In moved files**, change:
```csharp
// OLD
namespace Evolvatron.Evolvion.Environments;
namespace Evolvatron.Evolvion.Benchmarks;

// NEW
namespace Evolvatron.Tests.Evolvion.Environments;
namespace Evolvatron.Tests.Evolvion.Benchmarks;
```

**Files to update:**
- `Evolvatron.Tests/Evolvion/Environments/SpiralEnvironment.cs`
- `Evolvatron.Tests/Evolvion/Environments/CartPoleEnvironment.cs` (if exists)
- `Evolvatron.Tests/Evolvion/Environments/FollowTheCorridorEnvironment.cs` (if exists)
- `Evolvatron.Tests/Evolvion/Benchmarks/LandscapeNavigationTask.cs`
- `Evolvatron.Tests/Evolvion/Benchmarks/OptimizationLandscapes.cs`

### Step 4: Update Using Statements in Tests

**In all test files** (`Evolvatron.Tests/Evolvion/*Test.cs`), update imports:
```csharp
// OLD
using Evolvatron.Evolvion.Environments;
using Evolvatron.Evolvion.Benchmarks;

// NEW
using Evolvatron.Tests.Evolvion.Environments;
using Evolvatron.Tests.Evolvion.Benchmarks;
```

**Files to check/update:**
- `LongRunConvergenceTest.cs`
- `DeterminismVerificationTest.cs`
- `LandscapeNavigationBenchmarks.cs`
- `SpiralEvolutionTest.cs`
- Any other test files that import environments/benchmarks

### Step 5: Update Documentation

**Files to update:**
- `Evolvatron.Evolvion/README.md` - Remove references to Environments/ and Benchmarks/
- `README.MD` - Update project structure diagram
- `OPTUNA_RESULTS.md` - Update file paths if mentioned

**Update project structure in README.MD:**
```
OLD:
Evolvatron.Evolvion/
├── Environments/          # Test environments
├── Benchmarks/           # Benchmark tasks
└── ...

NEW:
Evolvatron.Evolvion/       # Pure algorithm library
├── Evolver.cs
├── Population.cs
└── ...

Evolvatron.Tests/
└── Evolvion/
    ├── Environments/      # Test environments (not shipped with library)
    ├── Benchmarks/        # Benchmark tasks (not shipped with library)
    └── *Tests.cs
```

### Step 6: Build and Test

```bash
# Clean build
dotnet clean
dotnet build Evolvatron.sln

# Run tests to verify everything still works
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj
```

**Expected**: All tests pass, no compilation errors

### Step 7: Verify Evolvatron.Evolvion is Clean

Check that core library no longer contains test code:
```bash
ls Evolvatron.Evolvion/Environments  # Should not exist
ls Evolvatron.Evolvion/Benchmarks    # Should not exist
```

### Step 8: Commit

```bash
git add -A
git commit -m "Refactor: Move test environments and benchmarks out of core Evolvion library

Moved Environments/ and Benchmarks/ from Evolvatron.Evolvion to Evolvatron.Tests/Evolvion/.
The core library should be pure algorithm code - test environments don't belong in production DLL.

Changes:
- Moved SpiralEnvironment.cs to Tests
- Moved CartPoleEnvironment.cs to Tests (if exists)
- Moved LandscapeNavigationTask.cs to Tests
- Moved OptimizationLandscapes.cs to Tests
- Updated all namespaces to Evolvatron.Tests.Evolvion.*
- Updated all using statements in test files
- Updated documentation to reflect new structure

Evolvatron.Evolvion is now a clean library with just the evolution engine.
Test/benchmark code is properly isolated in the test project.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Verification Checklist

After refactor, verify:

- [ ] `dotnet build` succeeds
- [ ] `dotnet test` passes all tests
- [ ] No `Environments/` folder in Evolvatron.Evolvion
- [ ] No `Benchmarks/` folder in Evolvatron.Evolvion
- [ ] `IEnvironment.cs` still exists in Evolvatron.Evolvion (interface stays)
- [ ] All moved files have correct namespace `Evolvatron.Tests.Evolvion.*`
- [ ] All test files import from correct namespace
- [ ] Documentation updated
- [ ] Git history shows moves (not delete+add)

## Benefits After Refactor

1. **Cleaner library boundary**: Core library is pure algorithm
2. **Smaller deployment**: Production builds don't include test code
3. **Better organization**: Test infrastructure clearly separated
4. **Easier to understand**: New developers see clean separation
5. **Follows .NET conventions**: Test code in test projects

## Estimated Time

- File moves: 5 minutes
- Namespace updates: 10 minutes
- Using statement updates: 10 minutes
- Documentation updates: 10 minutes
- Build/test verification: 5 minutes

**Total: ~40 minutes**

## Risks

**Low risk** - This is a pure refactor:
- No algorithm changes
- No logic changes
- Just moving files and updating namespaces
- Git preserves history

**Mitigation**: Run full test suite after refactor to ensure nothing broke.

## Future Enhancement (Optional)

After this refactor works, **consider** creating `Evolvatron.Evolvion.Benchmarks` as a separate published library:
- Others could use standard benchmarks
- Can version independently
- Can be referenced by multiple test projects
- Useful for comparisons with other evolutionary algorithms

But for now, keeping in Tests project is simpler and sufficient.
