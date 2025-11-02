using ILGPU;
using ILGPU.Algorithms;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU kernels for neural network forward pass evaluation.
/// Implements parallel row-by-row evaluation for batches of individuals.
/// </summary>
public static class GPUEvolvionKernels
{
    private const float Epsilon = 1e-9f;

    /// <summary>
    /// Evaluates a single row for all individuals in parallel.
    /// Each thread handles one individual's computation for the specified row.
    /// </summary>
    public static void EvaluateRowKernel(
        Index1D index,
        ArrayView<float> nodeValues,
        ArrayView<GPUEdge> edges,
        ArrayView<float> weights,
        ArrayView<float> biases,
        ArrayView<byte> activations,
        ArrayView<float> nodeParams,
        ArrayView<GPURowPlan> allRowPlans,
        int rowIdx,
        int individualCount,
        int nodesPerIndividual,
        int weightsPerIndividual)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        var rowPlan = allRowPlans[rowIdx];
        int nodeOffset = individualIdx * nodesPerIndividual;
        int weightOffset = individualIdx * weightsPerIndividual;

        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            nodeValues[nodeOffset + rowPlan.NodeStart + i] = 0.0f;
        }

        for (int edgeIdx = rowPlan.EdgeStart; edgeIdx < rowPlan.EdgeStart + rowPlan.EdgeCount; edgeIdx++)
        {
            var edge = edges[edgeIdx];
            float weight = weights[weightOffset + edgeIdx];
            float sourceValue = nodeValues[nodeOffset + edge.Source];

            nodeValues[nodeOffset + edge.Dest] += weight * sourceValue;
        }

        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            int nodeIdx = rowPlan.NodeStart + i;
            nodeValues[nodeOffset + nodeIdx] += biases[nodeOffset + nodeIdx];
        }

        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            int nodeIdx = rowPlan.NodeStart + i;
            int globalNodeIdx = nodeOffset + nodeIdx;
            float preActivation = nodeValues[globalNodeIdx];
            byte activationType = activations[nodeOffset + nodeIdx];

            int paramOffset = (nodeOffset + nodeIdx) * 4;
            float param0 = nodeParams[paramOffset];
            float param1 = nodeParams[paramOffset + 1];
            float param2 = nodeParams[paramOffset + 2];
            float param3 = nodeParams[paramOffset + 3];

            nodeValues[globalNodeIdx] = EvaluateActivation(
                activationType, preActivation, param0, param1, param2, param3);
        }
    }

    /// <summary>
    /// Evaluates a single row for all episodes in parallel.
    /// Each thread handles one episode. Maps episodeIdx to individualIdx for weight/bias lookup.
    /// </summary>
    public static void EvaluateRowForEpisodesKernel(
        Index1D index,
        ArrayView<float> nodeValues,
        ArrayView<GPUEdge> edges,
        ArrayView<float> weights,
        ArrayView<float> biases,
        ArrayView<byte> activations,
        ArrayView<float> nodeParams,
        ArrayView<GPURowPlan> allRowPlans,
        int rowIdx,
        int totalEpisodes,
        int episodesPerIndividual,
        int nodesPerIndividual,
        int weightsPerIndividual)
    {
        int episodeIdx = index;
        if (episodeIdx >= totalEpisodes) return;

        var rowPlan = allRowPlans[rowIdx];
        int individualIdx = episodeIdx / episodesPerIndividual;

        int nodeOffset = episodeIdx * nodesPerIndividual;
        int weightOffset = individualIdx * weightsPerIndividual;
        int biasOffset = individualIdx * nodesPerIndividual;

        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            nodeValues[nodeOffset + rowPlan.NodeStart + i] = 0.0f;
        }

        for (int edgeIdx = rowPlan.EdgeStart; edgeIdx < rowPlan.EdgeStart + rowPlan.EdgeCount; edgeIdx++)
        {
            var edge = edges[edgeIdx];
            float weight = weights[weightOffset + edgeIdx];
            float sourceValue = nodeValues[nodeOffset + edge.Source];

            nodeValues[nodeOffset + edge.Dest] += weight * sourceValue;
        }

        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            int nodeIdx = rowPlan.NodeStart + i;
            nodeValues[nodeOffset + nodeIdx] += biases[biasOffset + nodeIdx];
        }

        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            int nodeIdx = rowPlan.NodeStart + i;
            int globalNodeIdx = nodeOffset + nodeIdx;
            float preActivation = nodeValues[globalNodeIdx];
            byte activationType = activations[biasOffset + nodeIdx];

            int paramOffset = (biasOffset + nodeIdx) * 4;
            float param0 = nodeParams[paramOffset];
            float param1 = nodeParams[paramOffset + 1];
            float param2 = nodeParams[paramOffset + 2];
            float param3 = nodeParams[paramOffset + 3];

            nodeValues[globalNodeIdx] = EvaluateActivation(
                activationType, preActivation, param0, param1, param2, param3);
        }
    }

    /// <summary>
    /// Evaluates activation function on GPU.
    /// Handles all 11 activation types using ILGPU.XMath for GPU compatibility.
    /// </summary>
    private static float EvaluateActivation(
        byte activationType,
        float x,
        float param0,
        float param1,
        float param2,
        float param3)
    {
        switch (activationType)
        {
            case 0:
                return x;

            case 1:
                return XMath.Tanh(x);

            case 2:
                return 1.0f / (1.0f + XMath.Exp(-x));

            case 3:
                return XMath.Max(0.0f, x);

            case 4:
                return x > 0 ? x : param0 * x;

            case 5:
                return x > 0 ? x : param0 * (XMath.Exp(x) - 1.0f);

            case 6:
                return x / (1.0f + XMath.Abs(x));

            case 7:
                return XMath.Log(1.0f + XMath.Exp(x));

            case 8:
                return XMath.Sin(x);

            case 9:
                return XMath.Exp(-x * x);

            case 10:
            {
                float sqrtTerm = XMath.Sqrt(2.0f / XMath.PI);
                float cubicTerm = x + 0.044715f * x * x * x;
                float tanhTerm = XMath.Tanh(sqrtTerm * cubicTerm);
                return x * 0.5f * (1.0f + tanhTerm);
            }

            default:
                return x;
        }
    }

    /// <summary>
    /// Copies input values to the input layer (row 0) for all individuals.
    /// Each thread handles one individual.
    /// </summary>
    public static void SetInputsKernel(
        Index1D index,
        ArrayView<float> nodeValues,
        ArrayView<float> inputs,
        int individualCount,
        int nodesPerIndividual,
        int inputSize)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        int nodeOffset = individualIdx * nodesPerIndividual;
        int inputOffset = individualIdx * inputSize;

        for (int i = 0; i < inputSize; i++)
        {
            nodeValues[nodeOffset + i] = inputs[inputOffset + i];
        }
    }

    /// <summary>
    /// Copies input values to the input layer for all episodes.
    /// Each thread handles one episode.
    /// Maps episodeIdx to individualIdx for node value storage.
    /// </summary>
    public static void SetInputsForEpisodesKernel(
        Index1D index,
        ArrayView<float> nodeValues,
        ArrayView<float> inputs,
        int totalEpisodes,
        int episodesPerIndividual,
        int nodesPerIndividual,
        int inputSize)
    {
        int episodeIdx = index;
        if (episodeIdx >= totalEpisodes) return;

        int nodeOffset = episodeIdx * nodesPerIndividual;
        int inputOffset = episodeIdx * inputSize;

        for (int i = 0; i < inputSize; i++)
        {
            nodeValues[nodeOffset + i] = inputs[inputOffset + i];
        }
    }

    /// <summary>
    /// Extracts output values from the output layer for all individuals.
    /// Each thread handles one individual.
    /// </summary>
    public static void GetOutputsKernel(
        Index1D index,
        ArrayView<float> nodeValues,
        ArrayView<float> outputs,
        ArrayView<GPURowPlan> allRowPlans,
        int outputRowIdx,
        int individualCount,
        int nodesPerIndividual,
        int outputSize)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        var outputRowPlan = allRowPlans[outputRowIdx];
        int nodeOffset = individualIdx * nodesPerIndividual;
        int outputOffset = individualIdx * outputSize;

        for (int i = 0; i < outputSize; i++)
        {
            outputs[outputOffset + i] = nodeValues[nodeOffset + outputRowPlan.NodeStart + i];
        }
    }

    /// <summary>
    /// Extracts output values from the output layer for all episodes.
    /// Each thread handles one episode.
    /// </summary>
    public static void GetOutputsForEpisodesKernel(
        Index1D index,
        ArrayView<float> nodeValues,
        ArrayView<float> outputs,
        ArrayView<GPURowPlan> allRowPlans,
        int outputRowIdx,
        int totalEpisodes,
        int nodesPerIndividual,
        int outputSize)
    {
        int episodeIdx = index;
        if (episodeIdx >= totalEpisodes) return;

        var outputRowPlan = allRowPlans[outputRowIdx];
        int nodeOffset = episodeIdx * nodesPerIndividual;
        int outputOffset = episodeIdx * outputSize;

        for (int i = 0; i < outputSize; i++)
        {
            outputs[outputOffset + i] = nodeValues[nodeOffset + outputRowPlan.NodeStart + i];
        }
    }

    /// <summary>
    /// Initializes all episodes with random starting positions.
    /// Each thread handles one episode.
    /// Uses simple LCG for deterministic random number generation.
    /// </summary>
    public static void InitializeEpisodesKernel(
        Index1D index,
        ArrayView<float> positions,
        ArrayView<int> steps,
        ArrayView<byte> isTerminal,
        GPULandscapeConfig config,
        int episodeCount,
        int baseSeed)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        int seed = baseSeed + episodeIdx;
        int posOffset = episodeIdx * config.Dimensions;

        for (int i = 0; i < config.Dimensions; i++)
        {
            seed = LCGNext(seed);
            float randValue = (float)(seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
            positions[posOffset + i] = config.MinBound + randValue * (config.MaxBound - config.MinBound);
        }

        steps[episodeIdx] = 0;
        isTerminal[episodeIdx] = 0;
    }

    /// <summary>
    /// Computes observations from current environment state.
    /// For FullPosition observation type: just copy positions to observations.
    /// Each thread handles one episode.
    /// </summary>
    public static void ComputeObservationsKernel(
        Index1D index,
        ArrayView<float> observations,
        ArrayView<float> positions,
        GPULandscapeConfig config,
        int episodeCount)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        int posOffset = episodeIdx * config.Dimensions;
        int obsOffset = episodeIdx * config.Dimensions;

        for (int i = 0; i < config.Dimensions; i++)
        {
            observations[obsOffset + i] = positions[posOffset + i];
        }
    }

    /// <summary>
    /// Steps environment by applying actions to positions.
    /// Each thread handles one episode.
    /// </summary>
    public static void StepEnvironmentKernel(
        Index1D index,
        ArrayView<float> positions,
        ArrayView<float> actions,
        ArrayView<int> steps,
        ArrayView<byte> isTerminal,
        GPULandscapeConfig config,
        int episodeCount)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        if (isTerminal[episodeIdx] != 0) return;

        int posOffset = episodeIdx * config.Dimensions;
        int actionOffset = episodeIdx * config.Dimensions;

        for (int i = 0; i < config.Dimensions; i++)
        {
            positions[posOffset + i] += actions[actionOffset + i] * config.StepSize;
            positions[posOffset + i] = XMath.Clamp(positions[posOffset + i], config.MinBound, config.MaxBound);
        }

        steps[episodeIdx]++;

        if (steps[episodeIdx] >= config.MaxSteps)
        {
            isTerminal[episodeIdx] = 1;
        }
    }

    /// <summary>
    /// Finalizes fitness for all individuals by averaging landscape values across episodes.
    /// Each thread handles one individual.
    /// </summary>
    public static void FinalizeFitnessKernel(
        Index1D index,
        ArrayView<float> fitnessValues,
        ArrayView<float> positions,
        ArrayView<float> finalLandscapeValues,
        GPULandscapeConfig config,
        int individualCount,
        int episodesPerIndividual)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        int episodeStart = individualIdx * episodesPerIndividual;
        float totalFitness = 0.0f;

        for (int ep = 0; ep < episodesPerIndividual; ep++)
        {
            int episodeIdx = episodeStart + ep;
            int posOffset = episodeIdx * config.Dimensions;

            float landscapeValue = EvaluateLandscape(positions, posOffset, config);
            finalLandscapeValues[episodeIdx] = landscapeValue;
            totalFitness += -landscapeValue;
        }

        fitnessValues[individualIdx] = totalFitness / episodesPerIndividual;
    }

    /// <summary>
    /// Simple linear congruential generator for deterministic random numbers on GPU.
    /// </summary>
    private static int LCGNext(int seed)
    {
        return (1103515245 * seed + 12345) & 0x7FFFFFFF;
    }

    /// <summary>
    /// Evaluates landscape function at given position.
    /// </summary>
    private static float EvaluateLandscape(
        ArrayView<float> positions,
        int offset,
        GPULandscapeConfig config)
    {
        switch (config.LandscapeType)
        {
            case 0:
                return EvaluateSphere(positions, offset, config.Dimensions);
            case 1:
                return EvaluateRosenbrock(positions, offset, config.Dimensions);
            default:
                return EvaluateSphere(positions, offset, config.Dimensions);
        }
    }

    /// <summary>
    /// Sphere function: sum of squares.
    /// Global minimum at origin with value 0.
    /// </summary>
    private static float EvaluateSphere(ArrayView<float> positions, int offset, int dimensions)
    {
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++)
        {
            float val = positions[offset + i];
            sum += val * val;
        }
        return sum;
    }

    /// <summary>
    /// Rosenbrock function: classic optimization benchmark.
    /// Global minimum at (1, 1, ..., 1) with value 0.
    /// </summary>
    private static float EvaluateRosenbrock(ArrayView<float> positions, int offset, int dimensions)
    {
        float sum = 0.0f;
        for (int i = 0; i < dimensions - 1; i++)
        {
            float xi = positions[offset + i];
            float xi1 = positions[offset + i + 1];
            float term1 = xi1 - xi * xi;
            float term2 = 1.0f - xi;
            sum += 100.0f * term1 * term1 + term2 * term2;
        }
        return sum;
    }

    public static void InitializeXORKernel(
        Index1D index,
        ArrayView<int> currentCases,
        ArrayView<float> totalErrors,
        ArrayView<byte> isTerminal,
        int episodeCount)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        currentCases[episodeIdx] = 0;
        totalErrors[episodeIdx] = 0.0f;
        isTerminal[episodeIdx] = 0;
    }

    public static void GetXORObservationsKernel(
        Index1D index,
        ArrayView<float> observations,
        ArrayView<int> currentCases,
        int episodeCount)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        int caseIdx = currentCases[episodeIdx];
        int obsOffset = episodeIdx * 2;

        if (caseIdx >= 4)
        {
            observations[obsOffset] = 0.0f;
            observations[obsOffset + 1] = 0.0f;
            return;
        }

        switch (caseIdx)
        {
            case 0:
                observations[obsOffset] = 0.0f;
                observations[obsOffset + 1] = 0.0f;
                break;
            case 1:
                observations[obsOffset] = 0.0f;
                observations[obsOffset + 1] = 1.0f;
                break;
            case 2:
                observations[obsOffset] = 1.0f;
                observations[obsOffset + 1] = 0.0f;
                break;
            case 3:
                observations[obsOffset] = 1.0f;
                observations[obsOffset + 1] = 1.0f;
                break;
        }
    }

    public static void StepXORKernel(
        Index1D index,
        ArrayView<float> actions,
        ArrayView<int> currentCases,
        ArrayView<float> totalErrors,
        ArrayView<byte> isTerminal,
        int episodeCount)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        if (isTerminal[episodeIdx] != 0) return;

        int caseIdx = currentCases[episodeIdx];
        if (caseIdx >= 4) return;

        float expected = 0.0f;
        switch (caseIdx)
        {
            case 0: expected = 0.0f; break;
            case 1: expected = 1.0f; break;
            case 2: expected = 1.0f; break;
            case 3: expected = 0.0f; break;
        }

        float output = actions[episodeIdx];
        float error = (output - expected) * (output - expected);
        totalErrors[episodeIdx] += error;

        currentCases[episodeIdx]++;

        if (currentCases[episodeIdx] >= 4)
        {
            isTerminal[episodeIdx] = 1;
        }
    }

    public static void FinalizeXORFitnessKernel(
        Index1D index,
        ArrayView<float> fitnessValues,
        ArrayView<float> totalErrors,
        int individualCount,
        int episodesPerIndividual)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        int episodeStart = individualIdx * episodesPerIndividual;
        float totalFitness = 0.0f;

        for (int ep = 0; ep < episodesPerIndividual; ep++)
        {
            int episodeIdx = episodeStart + ep;
            float avgError = totalErrors[episodeIdx] / 4.0f;
            totalFitness += -avgError;
        }

        fitnessValues[individualIdx] = totalFitness / episodesPerIndividual;
    }

    public static void InitializeSpiralPointsKernel(
        Index1D index,
        ArrayView<float> spiralPoints,
        int pointsPerSpiral,
        float noise)
    {
        int pointIdx = index;
        int totalPoints = pointsPerSpiral * 2;
        if (pointIdx >= totalPoints) return;

        int spiralIdx = pointIdx / pointsPerSpiral;
        int i = pointIdx % pointsPerSpiral;

        float t = i * 4.0f * XMath.PI / pointsPerSpiral;
        float r = t / (4.0f * XMath.PI);

        int seed = 42 + pointIdx;
        seed = LCGNext(seed);
        float randX = noise > 0 ? ((float)(seed & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * noise : 0.0f;
        seed = LCGNext(seed);
        float randY = noise > 0 ? ((float)(seed & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * noise : 0.0f;

        float angle = spiralIdx == 0 ? t : t + XMath.PI;
        float x = r * XMath.Cos(angle) + randX;
        float y = r * XMath.Sin(angle) + randY;
        float label = spiralIdx == 0 ? -1.0f : 1.0f;

        int offset = pointIdx * 3;
        spiralPoints[offset] = x;
        spiralPoints[offset + 1] = y;
        spiralPoints[offset + 2] = label;
    }

    public static void InitializeSpiralEpisodesKernel(
        Index1D index,
        ArrayView<int> currentCases,
        ArrayView<float> totalErrors,
        ArrayView<byte> isTerminal,
        int episodeCount)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        currentCases[episodeIdx] = 0;
        totalErrors[episodeIdx] = 0.0f;
        isTerminal[episodeIdx] = 0;
    }

    public static void GetSpiralObservationsKernel(
        Index1D index,
        ArrayView<float> observations,
        ArrayView<float> spiralPoints,
        ArrayView<int> currentCases,
        int episodeCount,
        int totalPoints)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        int caseIdx = currentCases[episodeIdx];
        int obsOffset = episodeIdx * 2;

        if (caseIdx >= totalPoints)
        {
            observations[obsOffset] = 0.0f;
            observations[obsOffset + 1] = 0.0f;
            return;
        }

        int pointOffset = caseIdx * 3;
        observations[obsOffset] = spiralPoints[pointOffset];
        observations[obsOffset + 1] = spiralPoints[pointOffset + 1];
    }

    public static void StepSpiralKernel(
        Index1D index,
        ArrayView<float> actions,
        ArrayView<float> spiralPoints,
        ArrayView<int> currentCases,
        ArrayView<float> totalErrors,
        ArrayView<byte> isTerminal,
        int episodeCount,
        int totalPoints)
    {
        int episodeIdx = index;
        if (episodeIdx >= episodeCount) return;

        if (isTerminal[episodeIdx] != 0) return;

        int caseIdx = currentCases[episodeIdx];
        if (caseIdx >= totalPoints) return;

        int pointOffset = caseIdx * 3;
        float expected = spiralPoints[pointOffset + 2];

        float output = actions[episodeIdx];
        float error = (output - expected) * (output - expected);
        totalErrors[episodeIdx] += error;

        currentCases[episodeIdx]++;

        if (currentCases[episodeIdx] >= totalPoints)
        {
            isTerminal[episodeIdx] = 1;
        }
    }

    public static void FinalizeSpiralFitnessKernel(
        Index1D index,
        ArrayView<float> fitnessValues,
        ArrayView<float> totalErrors,
        int individualCount,
        int episodesPerIndividual,
        int totalPoints)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        int episodeStart = individualIdx * episodesPerIndividual;
        float totalFitness = 0.0f;

        for (int ep = 0; ep < episodesPerIndividual; ep++)
        {
            int episodeIdx = episodeStart + ep;
            float avgError = totalErrors[episodeIdx] / totalPoints;
            totalFitness += -avgError;
        }

        fitnessValues[individualIdx] = totalFitness / episodesPerIndividual;
    }
}
