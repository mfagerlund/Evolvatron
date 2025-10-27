using Evolvatron.Evolvion;

var spec = new SpeciesBuilder()
    .AddBiasRow()
    .AddInputRow(6)
    .AddHiddenRow(8, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid)
    .AddHiddenRow(6, ActivationType.Tanh, ActivationType.Swish)
    .AddOutputRow(3, ActivationType.Tanh)
    .ConnectBiasToAll()
    .FullyConnect(1, 2)
    .FullyConnect(2, 3)
    .FullyConnect(3, 4)
    .WithMaxInDegree(12)
    .Build();

Console.WriteLine($"Rows: {spec.RowCount}");
Console.WriteLine($"Total nodes: {spec.TotalNodes}");
Console.WriteLine($"Total edges: {spec.TotalEdges}");
Console.WriteLine($"Max in-degree: {spec.MaxInDegree}");

for (int i = 0; i < spec.RowPlans.Length; i++)
{
    var plan = spec.RowPlans[i];
    Console.WriteLine($"Row {i}: {plan}");
}

Console.WriteLine("\nRow 2 allowed activations:");
foreach (ActivationType activation in Enum.GetValues<ActivationType>())
{
    if (spec.IsActivationAllowed(2, activation))
    {
        Console.WriteLine($"  - {activation}");
    }
}
