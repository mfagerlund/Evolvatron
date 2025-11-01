namespace Evolvatron.Evolvion;

public class SpeciesBuilder
{
    private readonly List<int> _rowCounts = new();
    private readonly List<uint> _allowedActivationsPerRow = new();
    private readonly List<(int Source, int Dest)> _edges = new();
    private int _maxInDegree = 6;

    public SpeciesBuilder WithMaxInDegree(int maxInDegree)
    {
        _maxInDegree = maxInDegree;
        return this;
    }

    public SpeciesBuilder AddInputRow(int nodeCount)
    {
        _rowCounts.Add(nodeCount);
        _allowedActivationsPerRow.Add((1u << (int)ActivationType.Linear));
        return this;
    }

    public SpeciesBuilder AddHiddenRow(int nodeCount, params ActivationType[] allowedActivations)
    {
        _rowCounts.Add(nodeCount);

        uint mask = 0;
        foreach (var activation in allowedActivations)
        {
            mask |= (1u << (int)activation);
        }
        _allowedActivationsPerRow.Add(mask);
        return this;
    }

    /// <summary>
    /// Add multiple hidden rows with the same configuration.
    /// </summary>
    public SpeciesBuilder AddHiddenRow(int nodeCount, ActivationType activation, int count)
    {
        for (int i = 0; i < count; i++)
        {
            AddHiddenRow(nodeCount, activation);
        }
        return this;
    }

    public SpeciesBuilder AddOutputRow(int nodeCount, ActivationType activation = ActivationType.Tanh)
    {
        if (activation != ActivationType.Linear && activation != ActivationType.Tanh)
            throw new InvalidOperationException("Output row may only use Linear or Tanh activations");

        _rowCounts.Add(nodeCount);
        _allowedActivationsPerRow.Add((1u << (int)activation));
        return this;
    }

    public SpeciesBuilder InitializeSparse(Random random)
    {
        if (_rowCounts.Count < 2)
            throw new InvalidOperationException("Cannot initialize sparse topology with less than 2 rows");

        int numInputs = _rowCounts[0];
        int numOutputs = _rowCounts[^1];
        int totalHiddenNodes = _rowCounts.Skip(1).Take(_rowCounts.Count - 2).Sum();

        // Calculate minimum edges: aim for 3-4 edges per input/output, more if hidden layers exist
        int minEdges = (numInputs + numOutputs) * 3;
        if (totalHiddenNodes > 0)
        {
            // Add extra edges to ensure hidden layer connectivity
            minEdges += Math.Min(totalHiddenNodes, (numInputs + numOutputs) * 2);
        }

        // Build a temporary spec to use mutation methods
        var tempSpec = new SpeciesSpec
        {
            RowCounts = _rowCounts.ToArray(),
            AllowedActivationsPerRow = _allowedActivationsPerRow.ToArray(),
            MaxInDegree = _maxInDegree,
            Edges = new List<(int, int)>()
        };
        tempSpec.BuildRowPlans();

        // Phase 1: Add edges using EdgeAdd only (guaranteed acyclic)
        int edgeCount = 0;
        while (edgeCount < minEdges)
        {
            bool success = EdgeTopologyMutations.TryEdgeAdd(tempSpec, random);
            if (success)
                edgeCount = tempSpec.Edges.Count;
        }

        // Phase 2: Ensure all inputs/outputs connected to ACTIVE nodes
        int maxAttempts = 100;
        int attempts = 0;
        while (attempts < maxAttempts)
        {
            var activeNodes = ConnectivityValidator.ComputeActiveNodes(tempSpec);
            bool needsFixing = false;

            // Check if ANY nodes are active (besides inputs)
            bool hasActiveNodes = activeNodes.Skip(numInputs).Any(a => a);
            int outputStart = tempSpec.TotalNodes - numOutputs;

            // Check inputs: each must connect to at least one active node
            for (int i = 0; i < numInputs; i++)
            {
                bool connectedToActive = tempSpec.Edges
                    .Where(e => e.Source == i)
                    .Any(e => activeNodes[e.Dest]);

                if (!connectedToActive)
                {
                    // If no active nodes, connect directly to output
                    if (!hasActiveNodes)
                    {
                        int dest = outputStart + (i % numOutputs); // Spread inputs across outputs
                        if (tempSpec.Edges.Count(e => e.Dest == dest) < tempSpec.MaxInDegree)
                        {
                            tempSpec.Edges.Add((i, dest));
                            tempSpec.BuildRowPlans();
                            needsFixing = true;
                        }
                    }
                    else
                    {
                        // Connect to random active node
                        var candidates = Enumerable.Range(numInputs, tempSpec.TotalNodes - numInputs)
                            .Where(n => activeNodes[n])
                            .Where(n => tempSpec.Edges.Count(e => e.Dest == n) < tempSpec.MaxInDegree)
                            .ToList();
                        if (candidates.Count > 0)
                        {
                            int dest = candidates[random.Next(candidates.Count)];
                            tempSpec.Edges.Add((i, dest));
                            tempSpec.BuildRowPlans();
                            needsFixing = true;
                        }
                    }
                }
            }

            // Check outputs: each must connect from at least one ACTIVE node (input counts as active)
            for (int i = 0; i < numOutputs; i++)
            {
                int outputNode = outputStart + i;
                bool connectedFromActive = tempSpec.Edges
                    .Where(e => e.Dest == outputNode)
                    .Any(e => activeNodes[e.Source]);

                if (!connectedFromActive)
                {
                    // Connect from random input (inputs are always active)
                    int currentInDegree = tempSpec.Edges.Count(e => e.Dest == outputNode);
                    if (currentInDegree < tempSpec.MaxInDegree)
                    {
                        int src = random.Next(numInputs);
                        tempSpec.Edges.Add((src, outputNode));
                        tempSpec.BuildRowPlans();
                        needsFixing = true;
                    }
                }
            }

            if (!needsFixing)
                break;

            attempts++;
        }

        // Final validation: ensure we have functional network
        var finalActiveNodes = ConnectivityValidator.ComputeActiveNodes(tempSpec);
        int finalOutputStart = tempSpec.TotalNodes - numOutputs;

        // Check all inputs are connected
        for (int i = 0; i < numInputs; i++)
        {
            if (!tempSpec.Edges.Any(e => e.Source == i && finalActiveNodes[e.Dest]))
                throw new InvalidOperationException($"Failed to connect input {i} after {attempts} attempts");
        }

        // Check all outputs are connected
        for (int i = 0; i < numOutputs; i++)
        {
            int outputNode = finalOutputStart + i;
            if (!tempSpec.Edges.Any(e => e.Dest == outputNode && finalActiveNodes[e.Source]))
                throw new InvalidOperationException($"Failed to connect output {i} after {attempts} attempts");
        }

        // Copy edges back to builder
        _edges.Clear();
        _edges.AddRange(tempSpec.Edges);

        return this;
    }

    public SpeciesBuilder InitializeDense(Random random, float density = 1.0f)
    {
        if (_rowCounts.Count < 2)
            throw new InvalidOperationException("Cannot initialize dense topology with less than 2 rows");

        if (density <= 0.0f || density > 1.0f)
            throw new ArgumentException("Density must be in range (0.0, 1.0]", nameof(density));

        _edges.Clear();

        // For each layer (starting from first hidden/output layer)
        for (int destRowIdx = 1; destRowIdx < _rowCounts.Count; destRowIdx++)
        {
            int destRowStart = GetRowStart(destRowIdx);
            int destRowCount = _rowCounts[destRowIdx];

            int srcRowIdx = destRowIdx - 1;
            int srcRowStart = GetRowStart(srcRowIdx);
            int srcRowCount = _rowCounts[srcRowIdx];

            // For each node in destination layer
            for (int destLocalIdx = 0; destLocalIdx < destRowCount; destLocalIdx++)
            {
                int destNode = destRowStart + destLocalIdx;

                // Build list of candidate source nodes
                var candidateSources = new List<int>();
                for (int srcLocalIdx = 0; srcLocalIdx < srcRowCount; srcLocalIdx++)
                {
                    candidateSources.Add(srcRowStart + srcLocalIdx);
                }

                // Shuffle candidates for random selection (Fisher-Yates)
                for (int i = candidateSources.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (candidateSources[i], candidateSources[j]) = (candidateSources[j], candidateSources[i]);
                }

                // Select first N candidates after shuffle (true uniform random sampling)
                // targetCount = density * candidateCount, clamped to [1, maxInDegree]
                int targetCount = Math.Max(1, Math.Min(_maxInDegree,
                    (int)Math.Round(candidateSources.Count * density)));

                var selectedSources = candidateSources.Take(targetCount).ToList();

                // Add edges from selected sources
                foreach (var src in selectedSources)
                {
                    _edges.Add((src, destNode));
                }
            }
        }

        return this;
    }

    public SpeciesBuilder AddEdge(int sourceNode, int destNode)
    {
        _edges.Add((sourceNode, destNode));
        return this;
    }

    public SpeciesSpec Build()
    {
        if (_rowCounts.Count == 0)
            throw new InvalidOperationException("Cannot build empty topology");

        var spec = new SpeciesSpec
        {
            RowCounts = _rowCounts.ToArray(),
            AllowedActivationsPerRow = _allowedActivationsPerRow.ToArray(),
            MaxInDegree = _maxInDegree,
            Edges = _edges.ToList()
        };

        spec.Validate();
        spec.BuildRowPlans();

        return spec;
    }

    private int GetRowStart(int rowIndex)
    {
        int start = 0;
        for (int i = 0; i < rowIndex; i++)
        {
            start += _rowCounts[i];
        }
        return start;
    }
}
