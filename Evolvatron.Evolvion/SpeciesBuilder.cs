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

    public SpeciesBuilder AddOutputRow(int nodeCount, ActivationType activation = ActivationType.Tanh)
    {
        if (activation != ActivationType.Linear && activation != ActivationType.Tanh)
            throw new InvalidOperationException("Output row may only use Linear or Tanh activations");

        _rowCounts.Add(nodeCount);
        _allowedActivationsPerRow.Add((1u << (int)activation));
        return this;
    }

    public SpeciesBuilder FullyConnect(int fromRow, int toRow)
    {
        int fromStart = GetRowStart(fromRow);
        int fromEnd = fromStart + _rowCounts[fromRow];
        int toStart = GetRowStart(toRow);
        int toEnd = toStart + _rowCounts[toRow];

        for (int src = fromStart; src < fromEnd; src++)
        {
            for (int dst = toStart; dst < toEnd; dst++)
            {
                _edges.Add((src, dst));
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
