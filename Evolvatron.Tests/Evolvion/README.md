# Evolvion Test Suite

This directory contains comprehensive tests for the Evolvion evolutionary neural controller framework.

## Test Structure

```
Evolvion/
├── ActivationFunctionTests.cs      (17 tests) - All 11 activation functions
├── CoreDataStructureTests.cs       (19 tests) - Individual, SpeciesSpec, RowPlan
├── MutationOperatorTests.cs        (26 tests) - 5 mutation operators + Glorot init
└── CPUEvaluatorTests.cs            (29 tests) - Neural network forward pass
```

## Running Tests

```bash
# Run all Evolvion tests
dotnet test --filter "FullyQualifiedName~Evolvion"

# Run specific test class
dotnet test --filter "FullyQualifiedName~ActivationFunctionTests"

# Run specific test
dotnet test --filter "FullyQualifiedName~CPUEvaluator_TwoLayerNetwork_ComputesCorrectly"

# Run with verbose output
dotnet test --filter "FullyQualifiedName~Evolvion" --logger "console;verbosity=detailed"
```

## Test Coverage Summary

| Component | Tests | Coverage |
|-----------|-------|----------|
| Activation Functions | 17 | All 11 types + validation |
| Core Data Structures | 19 | Individual, SpeciesSpec, RowPlan |
| Mutation Operators | 26 | 5 operators + initialization |
| CPU Evaluator | 29 | Forward pass, multi-layer networks |
| **Total** | **91** | **100% pass rate** |

## Key Test Categories

### 1. Activation Functions
- ✅ Correctness of all 11 activation types
- ✅ Parameter requirements (LeakyReLU, ELU)
- ✅ Output validation (Linear/Tanh for output layers)
- ✅ Numerical stability (no NaN/Inf)

### 2. Data Structures
- ✅ Individual: construction, copying, parameter access
- ✅ SpeciesSpec: validation, topology rules, acyclic constraints
- ✅ RowPlan: metadata storage, edge sorting

### 3. Mutation Operators
- ✅ WeightJitter (Gaussian noise proportional to magnitude)
- ✅ WeightReset (uniform random replacement)
- ✅ WeightL1Shrink (regularization)
- ✅ ActivationSwap (respects allowed activations)
- ✅ NodeParamMutate (parameter tuning)
- ✅ GlorotUniform (proper weight initialization)

### 4. CPU Evaluator
- ✅ Single/multi-layer networks
- ✅ Bias handling
- ✅ Weighted sum accumulation
- ✅ Activation application
- ✅ Determinism
- ✅ Complex network execution

## Test Design Principles

1. **Isolation:** Each test is independent and can run in any order
2. **Clarity:** Test names follow `Component_Scenario_ExpectedResult` pattern
3. **Coverage:** Both happy path and edge cases tested
4. **Speed:** Fast execution (~100ms for 91 tests)
5. **Determinism:** Tests use fixed random seeds for reproducibility

## Example Test Patterns

### Basic Correctness Test
```csharp
[Fact]
public void Linear_ReturnsInputUnchanged()
{
    var activation = ActivationType.Linear;
    var result = activation.Evaluate(2.5f, Array.Empty<float>());
    Assert.Equal(2.5f, result, precision: 6);
}
```

### Parameterized Theory Test
```csharp
[Theory]
[InlineData(ActivationType.Linear)]
[InlineData(ActivationType.Tanh)]
public void OnlyLinearAndTanh_ValidForOutput(ActivationType activation)
{
    Assert.True(activation.IsValidForOutput());
}
```

### Integration Test
```csharp
[Fact]
public void CPUEvaluator_TwoLayerNetwork_ComputesCorrectly()
{
    // Setup network with known weights
    // Evaluate with test inputs
    // Verify outputs match hand-calculated values
}
```

## Adding New Tests

When adding new functionality:

1. Create tests first (TDD approach)
2. Follow existing naming conventions
3. Use `[Fact]` for single tests, `[Theory]` for parameterized tests
4. Group related tests with `#region` comments
5. Add to appropriate test file or create new one
6. Update this README with new test counts

## Test Dependencies

- xUnit 2.5.3
- .NET 8.0
- Evolvatron.Evolvion project reference

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (<200ms)
- No external dependencies
- Deterministic results
- Clear pass/fail signals

## Related Documentation

- [Evolvion Specification](../../Evolvion.md)
- [Test Report](../../EVOLVION_TEST_REPORT.md)
- [CLAUDE.md](../../CLAUDE.md)
