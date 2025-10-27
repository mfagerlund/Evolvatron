namespace Evolvatron.Evolvion.Refactor;

public enum ActivationType
{
    Linear,
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU,
    ELU,
    Swish,
    Gaussian
}

public class NodeDef
{
    public int NodeDefId { get; }
    public int RowId { get; set; }
    public int ColId { get; set; }
    public ActivationType[] AllowedActivations { get; set; } = [];

    public NodeDef(int nodeDefId, int rowId, int colId, ActivationType[] allowedActivations)
    {
        NodeDefId = nodeDefId;
        RowId = rowId;
        ColId = colId;
        AllowedActivations = allowedActivations;
    }
}

public class BiasDef
{
    public int BiasDefId { get; }
    public int NodeDefId { get; }

    public BiasDef(int biasDefId, int nodeDefId)
    {
        BiasDefId = biasDefId;
        NodeDefId = nodeDefId;
    }
}

public class LayerDef
{
    public int RowId { get; set; }
    public int NodeCount { get; init; }
    public ActivationType[]? AllowedActivations { get; init; }
    public ActivationType[]? FixedActivationsPerNode { get; init; }
    public int? MaxInDegree { get; init; }

    public List<NodeDef> NodeDefs { get; set; } = [];
    
    public bool IsFixed => FixedActivationsPerNode != null;
}

public class GenomeDef
{
    private List<LayerDef> _layers = new();
    private List<NodeDef> _nodeDefs = new();
    private List<BiasDef> _biasDefs = new();
    private List<LinkDef> _linkDefs = new();
    private List<WeightDef> _weightDefs = new();

    public IReadOnlyList<LayerDef> Layers => _layers;
    public IReadOnlyList<NodeDef> NodeDefs => _nodeDefs;
    public IReadOnlyList<BiasDef> BiasDefs => _biasDefs;
    public IReadOnlyList<LinkDef> LinkDefs => _linkDefs;
    public IReadOnlyList<WeightDef> WeightDefs => _weightDefs;

    public GenomeDef AddLayer(int nodeCount, ActivationType[]? allowedActivations = null, int? maxInDegree = null)
    {
        allowedActivations ??= GetDefaultActivations();
        int rowId = _layers.Count;
        var layer = new LayerDef
        {
            RowId = rowId,
            NodeCount = nodeCount,
            AllowedActivations = allowedActivations,
            MaxInDegree = maxInDegree
        };
        _layers.Add(layer);

        for (int col = 0; col < nodeCount; col++)
        {
            int nodeDefId = _nodeDefs.Count;
            var nodeDef = new NodeDef(nodeDefId, rowId, col, allowedActivations);
            _nodeDefs.Add(nodeDef);
            layer.NodeDefs.Add(nodeDef);

            var biasDef = new BiasDef(_biasDefs.Count, nodeDefId);
            _biasDefs.Add(biasDef);
        }

        return this;
    }

    public GenomeDef AddLayerFixed(ActivationType[] activationsPerNode, int? maxInDegree = null)
    {
        int rowId = _layers.Count;
        var layer = new LayerDef
        {
            RowId = rowId,
            NodeCount = activationsPerNode.Length,
            FixedActivationsPerNode = activationsPerNode,
            MaxInDegree = maxInDegree
        };
        _layers.Add(layer);

        for (int col = 0; col < activationsPerNode.Length; col++)
        {
            int nodeDefId = _nodeDefs.Count;
            var nodeDef = new NodeDef(nodeDefId, rowId, col, new[] { activationsPerNode[col] });
            _nodeDefs.Add(nodeDef);
            layer.NodeDefs.Add(nodeDef);

            var biasDef = new BiasDef(_biasDefs.Count, nodeDefId);
            _biasDefs.Add(biasDef);
        }

        return this;
    }

    public GenomeDef AddFullyConnectedLinks(int fromLayerIndex, int toLayerIndex)
    {
        int fromStart = GetLayerStartIndex(fromLayerIndex);
        int fromEnd = fromStart + _layers[fromLayerIndex].NodeCount;
        int toStart = GetLayerStartIndex(toLayerIndex);
        int toEnd = toStart + _layers[toLayerIndex].NodeCount;

        for (int src = fromStart; src < fromEnd; src++)
        {
            for (int dst = toStart; dst < toEnd; dst++)
            {
                var linkDef = new LinkDef(src, dst, _linkDefs.Count);
                _linkDefs.Add(linkDef);

                var weightDef = new WeightDef(linkDef, _weightDefs.Count);
                _weightDefs.Add(weightDef);
            }
        }

        return this;
    }

    private int GetLayerStartIndex(int layerIndex)
    {
        int start = 0;
        for (int i = 0; i < layerIndex; i++)
        {
            start += _layers[i].NodeCount;
        }
        return start;
    }

    private static ActivationType[] GetDefaultActivations() => new[]
    {
        ActivationType.Linear,
        ActivationType.Tanh,
        ActivationType.Sigmoid,
        ActivationType.ReLU,
        ActivationType.LeakyReLU,
        ActivationType.ELU,
        ActivationType.Swish,
        ActivationType.Gaussian
    };
}

public class LinkDef
{
    public int SourceNodeIndex { get; }
    public int TargetNodeIndex { get; }
    public int LinkDefId { get; }

    public LinkDef(int sourceNodeIndex, int targetNodeIndex, int linkDefId)
    {
        SourceNodeIndex = sourceNodeIndex;
        TargetNodeIndex = targetNodeIndex;
        LinkDefId = linkDefId;
    }
}

public class WeightDef
{
    public LinkDef LinkDef { get; }
    public int WeightDefId { get; }

    public WeightDef(LinkDef linkDef, int weightDefId)
    {
        LinkDef = linkDef;
        WeightDefId = weightDefId;
    }
}

public class SpeciesDef
{
    private List<LinkDef> _activeLinks = new();
    private List<WeightDef> _activeWeights = new();

    public GenomeDef GenomeDef { get; }
    public IReadOnlyList<LinkDef> ActiveLinks => _activeLinks;
    public IReadOnlyList<WeightDef> ActiveWeights => _activeWeights;

    public SpeciesDef(GenomeDef genomeDef)
    {
        GenomeDef = genomeDef;

        foreach (var linkDef in genomeDef.LinkDefs)
        {
            _activeLinks.Add(linkDef);
        }

        foreach (var weightDef in genomeDef.WeightDefs)
        {
            _activeWeights.Add(weightDef);
        }
    }

    public void RemoveLink(LinkDef linkDef)
    {
        _activeLinks.Remove(linkDef);

        var weightsToRemove = _activeWeights.Where(w => w.LinkDef == linkDef).ToList();
        foreach (var weight in weightsToRemove)
        {
            _activeWeights.Remove(weight);
        }
    }
}

public class Link
{
    public LinkDef LinkDef { get; }
    public bool IsActive { get; set; }

    public Link(LinkDef linkDef)
    {
        LinkDef = linkDef;
        IsActive = true;
    }
}

public class Weight
{
    public WeightDef WeightDef { get; }
    public float Value { get; set; }

    public Weight(WeightDef weightDef, float initialValue = 0f)
    {
        WeightDef = weightDef;
        Value = initialValue;
    }
}

public class Node
{
    public NodeDef NodeDef { get; }
    public ActivationType Activation { get; set; }

    public Node(NodeDef nodeDef, ActivationType activation)
    {
        NodeDef = nodeDef;
        Activation = activation;
    }
}

public class Bias
{
    public BiasDef BiasDef { get; }
    public float Value { get; set; }

    public Bias(BiasDef biasDef, float initialValue = 0f)
    {
        BiasDef = biasDef;
        Value = initialValue;
    }
}

public class Individual
{
    private Dictionary<int, Link> _links = new();
    private Dictionary<int, Weight> _weights = new();
    private Dictionary<int, Node> _nodes = new();
    private Dictionary<int, Bias> _biases = new();

    public SpeciesDef SpeciesDef { get; }
    public IReadOnlyDictionary<int, Link> Links => _links;
    public IReadOnlyDictionary<int, Weight> Weights => _weights;
    public IReadOnlyDictionary<int, Node> Nodes => _nodes;
    public IReadOnlyDictionary<int, Bias> Biases => _biases;

    public Individual(SpeciesDef speciesDef)
    {
        SpeciesDef = speciesDef;

        foreach (var linkDef in speciesDef.ActiveLinks)
        {
            _links[linkDef.LinkDefId] = new Link(linkDef);
        }

        foreach (var weightDef in speciesDef.ActiveWeights)
        {
            _weights[weightDef.WeightDefId] = new Weight(weightDef);
        }

        foreach (var nodeDef in speciesDef.GenomeDef.NodeDefs)
        {
            var activation = nodeDef.AllowedActivations[0];
            _nodes[nodeDef.NodeDefId] = new Node(nodeDef, activation);
        }

        foreach (var biasDef in speciesDef.GenomeDef.BiasDefs)
        {
            _biases[biasDef.BiasDefId] = new Bias(biasDef);
        }
    }

    public void DisableLink(int linkDefId)
    {
        if (_links.TryGetValue(linkDefId, out var link))
        {
            link.IsActive = false;
        }
    }

    public void MutateWeights(Random random, float mutationRate, float mutationStrength)
    {
        foreach (var weight in _weights.Values)
        {
            if (random.NextDouble() < mutationRate)
            {
                weight.Value += (float)(random.NextDouble() * 2 - 1) * mutationStrength;
            }
        }
    }

    public void MutateBiases(Random random, float mutationRate, float mutationStrength)
    {
        foreach (var bias in _biases.Values)
        {
            if (random.NextDouble() < mutationRate)
            {
                bias.Value += (float)(random.NextDouble() * 2 - 1) * mutationStrength;
            }
        }
    }

    public void MutateActivations(Random random, float mutationRate)
    {
        foreach (var node in _nodes.Values)
        {
            if (node.NodeDef.AllowedActivations.Length > 1 && random.NextDouble() < mutationRate)
            {
                node.Activation = node.NodeDef.AllowedActivations[random.Next(node.NodeDef.AllowedActivations.Length)];
            }
        }
    }
}

public class Population
{
    private List<SpeciesDef> _species = new();
    private List<Individual> _individuals = new();

    public GenomeDef GenomeDef { get; }
    public IReadOnlyList<SpeciesDef> Species => _species;
    public IReadOnlyList<Individual> Individuals => _individuals;

    public Population(GenomeDef genomeDef)
    {
        GenomeDef = genomeDef;
    }

    public SpeciesDef CreateSpecies()
    {
        var species = new SpeciesDef(GenomeDef);
        _species.Add(species);
        return species;
    }

    public Individual CreateIndividual(SpeciesDef speciesDef, Random random)
    {
        var individual = new Individual(speciesDef);

        foreach (var weight in individual.Weights.Values)
        {
            weight.Value = (float)(random.NextDouble() * 2 - 1);
        }

        foreach (var bias in individual.Biases.Values)
        {
            bias.Value = (float)(random.NextDouble() * 2 - 1) * 0.1f;
        }

        foreach (var node in individual.Nodes.Values)
        {
            if (node.NodeDef.AllowedActivations.Length > 1)
            {
                node.Activation = node.NodeDef.AllowedActivations[random.Next(node.NodeDef.AllowedActivations.Length)];
            }
        }

        _individuals.Add(individual);
        return individual;
    }
}

/*
 * DESIGN PRINCIPLES AND OPEN QUESTIONS:
 *
 * ARCHITECTURE SUMMARY:
 * - GenomeDef: Population-wide immutable topology definition (NodeDefs, BiasDefs, LinkDefs, WeightDefs)
 * - SpeciesDef: Species-specific topology subset (active links/weights from GenomeDef)
 * - Individual: Per-individual mutable state (weights, biases, activation function choices)
 * - All Def objects are immutable and shared; LinkDef/NodeDef never change after creation
 * - Topology is 100% fixed per species; individuals differ only in weights, biases, activations
 *
 * KEY DECISIONS MADE (from previous Q&A):
 * 1. Activation functions: Individual-level, evolvable with low mutation rate
 * 2. Weight:Link relationship: 1:1, but WeightDef can be reused
 * 3. Link removal: Physical removal from SpeciesDef
 * 4. Biases: Per-node, separate from weights
 * 5. Mutations: Structural changes in Defs, parameter changes in Individuals
 * 6. Crossover: Match by DefId, links in same order
 * 7. Species creation: Start minimal, complexify via mutations
 *
 * REMAINING OPEN QUESTIONS:
 *
 * 1. STRUCTURAL MUTATION OPERATIONS:
 *    - AddLink: How do we ensure MaxInDegree constraints are respected?
 *    - RemoveLink: Should there be a minimum connectivity requirement?
 *    - SplitLink: When splitting A→B into A→C→B, how do we choose C's position on the grid?
 *    - Should these operations be methods on SpeciesDef or a separate MutationEngine class?
 *
 * 2. MINIMAL SPECIES INITIALIZATION:
 *    - Current implementation starts with all links from GenomeDef
 *    - Should we instead start with minimal connectivity (each input→one output, each output←one input)?
 *    - Or start with a specific pattern (e.g., fully connected input→first hidden, last hidden→output)?
 *
 * 3. WEIGHT TRANSFER BETWEEN SPECIES:
 *    - When creating a new species from a parent individual, how do we copy weights?
 *    - Match by LinkDefId? What happens to unmatched links (new links from split operations)?
 *    - Should unmatched weights initialize to 0, small random, or interpolated from nearby connections?
 *
 * 4. CROSSOVER IMPLEMENTATION:
 *    - For weights/biases: average parents, or randomly pick one?
 *    - For activations: randomly pick one parent's choice?
 *    - How do we handle structural compatibility checks before crossover?
 *
 * 5. SPECIES CREATION WORKFLOW:
 *    - Exact sequence: select parent individual → mutate SpeciesDef → create new SpeciesDef →
 *      create new individual → copy/transfer weights → fill population with mutations?
 *    - When does structural mutation happen vs when does parameter mutation happen?
 *
 * 6. LINK ORDER CONSISTENCY:
 *    - How do we ensure links stay in consistent order for efficient crossover?
 *    - Should SpeciesDef.ActiveLinks be sorted by LinkDefId?
 *    - Or maintain insertion order from parent species?
 *
 * 7. GRID-BASED TOPOLOGY:
 *    - How is the (RowId, ColId) grid used in mutation operations?
 *    - When splitting a link, how do we pick intermediate node positions?
 *    - Are there spatial constraints (e.g., only connect to adjacent layers)?
 *
 * 8. ACTIVATION FUNCTION CONSTRAINT ENFORCEMENT:
 *    - Fixed layers have FixedActivationsPerNode - should Individual enforce this?
 *    - Or trust that NodeDef.AllowedActivations has length 1 for fixed nodes?
 *    - Should we validate during Individual construction?
 *
 * 9. MAXINDEGREE ENFORCEMENT:
 *    - Where is MaxInDegree actually enforced?
 *    - At GenomeDef.AddFullyConnectedLinks (prevent adding too many)?
 *    - At SpeciesDef mutation time (reject mutations that violate)?
 *    - Or never enforced, just advisory?
 *
 * 10. SPECIES COMPATIBILITY METRICS:
 *     - Do we need a distance/compatibility measure between SpeciesDefs?
 *     - For speciation decisions (à la NEAT's compatibility distance)?
 *     - Or is species identity purely based on explicit branching events?
 */
