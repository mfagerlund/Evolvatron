namespace Evolvatron.Evolvion.Refactor;

/*
 * GPU EXECUTION STRATEGIES:
 *
 * The executable format supports multiple parallelization approaches for GPU kernels:
 *
 * STRATEGY 1: Parallelize over individuals (batch dimension)
 * - Thread grid: [batchSize]
 * - Each thread processes one complete individual's forward pass
 * - Sequential execution of nodes in ExecutionOrder within each thread
 * - No synchronization needed (individuals are independent)
 * - Best for: Small networks (10-100 nodes), simple feed-forward topologies
 *
 * Pseudo-kernel:
 *   ProcessIndividual(batchIdx):
 *     for each nodeId in ExecutionOrder:
 *       accumulator = 0
 *       for conn in 0..NodeInDegrees[nodeId]:
 *         flatIdx = nodeId * MaxInDegree + conn
 *         sourceNode = ConnectionSources[flatIdx]
 *         weightId = ConnectionWeightIds[flatIdx]
 *         accumulator += NodeValues[batchIdx, sourceNode] * Weights[batchIdx, weightId]
 *       accumulator += Biases[batchIdx, NodeBiasIds[nodeId]]
 *       NodeValues[batchIdx, nodeId] = Activate(accumulator, Activations[batchIdx, nodeId])
 *
 * STRATEGY 2: Parallelize over nodes (requires layer barriers)
 * - Outer loop over layers (CPU-side kernel launches)
 * - Thread grid: [batchSize, nodesInLayer]
 * - Each thread processes one node for one individual
 * - Synchronization via separate kernel launches per layer
 * - Best for: Large networks (100+ nodes), when batch size is small
 *
 * Pseudo-code:
 *   for each layer:
 *     Kernel ProcessLayer(batchIdx, nodeIdxInLayer):
 *       nodeId = layerStartOffset + nodeIdxInLayer
 *       compute NodeValues[batchIdx, nodeId]
 *     // Implicit barrier between kernel launches
 *
 * STRATEGY 3: Hybrid (batch + intra-layer parallelism)
 * - Thread grid: [batchSize, maxNodesPerLayer]
 * - Parallelize over batch and nodes within each layer
 * - Sequential execution of layers (via kernel launches)
 * - Best for: Medium networks, large batches (1000+ individuals)
 *
 * MEMORY LAYOUT (all strategies):
 * - Topology arrays are constant for all threads (read-only shared memory candidate)
 * - Batch arrays indexed as: array[batchIdx * stride + elementId]
 * - Coalesced access pattern when batch dimension is innermost or contiguous
 *
 * ACTIVATION FUNCTION BRANCHING:
 * - Per-node activation types cause divergence within a warp
 * - For Strategy 1: minimal impact (each thread independent)
 * - For Strategy 2/3: use warp-level intrinsics or lookup tables to reduce divergence
 *
 * FUTURE OPTIMIZATION:
 * - Layer-wise shared memory caching of previous layer activations
 * - Tensor core utilization for dense layers (requires matrix reformulation)
 * - Multi-species batching (pack multiple species into one kernel launch)
 */

public class ExecutableTopology
{
    public int NumNodes { get; init; }
    public int NumWeights { get; init; }
    public int NumBiases { get; init; }
    public int MaxInDegree { get; init; }

    public int[] NodeInDegrees { get; init; } = [];
    public int[] ConnectionSources { get; init; } = [];
    public int[] ConnectionWeightIds { get; init; } = [];
    public int[] NodeBiasIds { get; init; } = [];
    public int[] ExecutionOrder { get; init; } = [];

    public ExecutableTopology(SpeciesDef speciesDef)
    {
        var genomeDef = speciesDef.GenomeDef;
        NumNodes = genomeDef.NodeDefs.Count;
        NumBiases = genomeDef.BiasDefs.Count;
        NumWeights = speciesDef.ActiveWeights.Count;

        var incomingConnections = new Dictionary<int, List<(int sourceNode, int weightId)>>();
        foreach (var nodeDef in genomeDef.NodeDefs)
        {
            incomingConnections[nodeDef.NodeDefId] = new List<(int, int)>();
        }

        foreach (var weightDef in speciesDef.ActiveWeights)
        {
            var linkDef = weightDef.LinkDef;
            int targetNode = linkDef.TargetNodeIndex;
            int sourceNode = linkDef.SourceNodeIndex;
            int weightId = weightDef.WeightDefId;
            incomingConnections[targetNode].Add((sourceNode, weightId));
        }

        MaxInDegree = incomingConnections.Values.Max(list => list.Count);
        NodeInDegrees = new int[NumNodes];
        ConnectionSources = new int[NumNodes * MaxInDegree];
        ConnectionWeightIds = new int[NumNodes * MaxInDegree];
        NodeBiasIds = new int[NumNodes];

        for (int i = 0; i < ConnectionSources.Length; i++)
        {
            ConnectionSources[i] = -1;
            ConnectionWeightIds[i] = -1;
        }

        for (int nodeId = 0; nodeId < NumNodes; nodeId++)
        {
            var connections = incomingConnections[nodeId];
            NodeInDegrees[nodeId] = connections.Count;

            for (int connIdx = 0; connIdx < connections.Count; connIdx++)
            {
                int flatIdx = nodeId * MaxInDegree + connIdx;
                ConnectionSources[flatIdx] = connections[connIdx].sourceNode;
                ConnectionWeightIds[flatIdx] = connections[connIdx].weightId;
            }

            var biasDef = genomeDef.BiasDefs.FirstOrDefault(b => b.NodeDefId == nodeId);
            NodeBiasIds[nodeId] = biasDef?.BiasDefId ?? -1;
        }

        ExecutionOrder = ComputeTopologicalOrder(genomeDef, incomingConnections);
    }

    private static int[] ComputeTopologicalOrder(GenomeDef genomeDef, Dictionary<int, List<(int, int)>> incomingConnections)
    {
        var order = new List<int>();
        var visited = new HashSet<int>();

        void Visit(int nodeId)
        {
            if (visited.Contains(nodeId)) return;
            visited.Add(nodeId);

            foreach (var (sourceNode, _) in incomingConnections[nodeId])
            {
                Visit(sourceNode);
            }

            order.Add(nodeId);
        }

        foreach (var nodeDef in genomeDef.NodeDefs)
        {
            Visit(nodeDef.NodeDefId);
        }

        return order.ToArray();
    }
}

public class ExecutableBatch
{
    public ExecutableTopology Topology { get; init; }
    public int BatchSize { get; init; }

    public float[] Weights { get; init; } = [];
    public float[] Biases { get; init; } = [];
    public int[] Activations { get; init; } = [];

    public float[] NodeValues { get; set; } = [];

    public ExecutableBatch(ExecutableTopology topology, int batchSize)
    {
        Topology = topology;
        BatchSize = batchSize;

        Weights = new float[batchSize * topology.NumWeights];
        Biases = new float[batchSize * topology.NumBiases];
        Activations = new int[batchSize * topology.NumNodes];
        NodeValues = new float[batchSize * topology.NumNodes];
    }

    public void LoadIndividual(int batchIndex, Individual individual)
    {
        int weightOffset = batchIndex * Topology.NumWeights;
        foreach (var kvp in individual.Weights)
        {
            int weightId = kvp.Key;
            float value = kvp.Value.Value;
            Weights[weightOffset + weightId] = value;
        }

        int biasOffset = batchIndex * Topology.NumBiases;
        foreach (var kvp in individual.Biases)
        {
            int biasId = kvp.Key;
            float value = kvp.Value.Value;
            Biases[biasOffset + biasId] = value;
        }

        int activationOffset = batchIndex * Topology.NumNodes;
        foreach (var kvp in individual.Nodes)
        {
            int nodeId = kvp.Key;
            int activation = (int)kvp.Value.Activation;
            Activations[activationOffset + nodeId] = activation;
        }
    }

    public void Execute(float[] inputs, int batchIndex)
    {
        int nodeOffset = batchIndex * Topology.NumNodes;
        int weightOffset = batchIndex * Topology.NumWeights;
        int biasOffset = batchIndex * Topology.NumBiases;
        int activationOffset = batchIndex * Topology.NumNodes;

        for (int i = 0; i < inputs.Length; i++)
        {
            NodeValues[nodeOffset + i] = inputs[i];
        }

        foreach (int nodeId in Topology.ExecutionOrder)
        {
            if (nodeId < inputs.Length) continue;

            float accumulator = 0f;
            int inDegree = Topology.NodeInDegrees[nodeId];

            for (int connIdx = 0; connIdx < inDegree; connIdx++)
            {
                int flatIdx = nodeId * Topology.MaxInDegree + connIdx;
                int sourceNode = Topology.ConnectionSources[flatIdx];
                int weightId = Topology.ConnectionWeightIds[flatIdx];

                if (sourceNode >= 0 && weightId >= 0)
                {
                    float sourceValue = NodeValues[nodeOffset + sourceNode];
                    float weight = Weights[weightOffset + weightId];
                    accumulator += sourceValue * weight;
                }
            }

            int biasId = Topology.NodeBiasIds[nodeId];
            if (biasId >= 0)
            {
                accumulator += Biases[biasOffset + biasId];
            }

            int activationType = Activations[activationOffset + nodeId];
            float activated = ApplyActivation(accumulator, (ActivationType)activationType);
            NodeValues[nodeOffset + nodeId] = activated;
        }
    }

    public float[] GetOutputs(int batchIndex, int numOutputs)
    {
        int nodeOffset = batchIndex * Topology.NumNodes;
        int outputStart = Topology.NumNodes - numOutputs;

        var outputs = new float[numOutputs];
        for (int i = 0; i < numOutputs; i++)
        {
            outputs[i] = NodeValues[nodeOffset + outputStart + i];
        }
        return outputs;
    }

    private static float ApplyActivation(float x, ActivationType type)
    {
        return type switch
        {
            ActivationType.Linear => x,
            ActivationType.Tanh => MathF.Tanh(x),
            ActivationType.Sigmoid => 1f / (1f + MathF.Exp(-x)),
            ActivationType.ReLU => MathF.Max(0f, x),
            ActivationType.LeakyReLU => x > 0 ? x : 0.01f * x,
            ActivationType.ELU => x > 0 ? x : MathF.Exp(x) - 1f,
            ActivationType.Swish => x / (1f + MathF.Exp(-x)),
            ActivationType.Gaussian => MathF.Exp(-x * x),
            _ => x
        };
    }
}
