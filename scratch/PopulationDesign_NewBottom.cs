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
