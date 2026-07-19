namespace Evolvatron.Evolvion.ES;

public class IslandConfig
{
    // Population structure
    public int IslandCount { get; set; } = 5;
    public int MinIslandPop { get; set; } = 512;

    // Strategy selection
    public UpdateStrategyType Strategy { get; set; } = UpdateStrategyType.CEM;

    // Optional PORTFOLIO: one update rule per island instead of a single global one. When non-null, island
    // i uses IslandStrategies[i % count]; when null, every island uses Strategy (unchanged behavior). Each
    // island gets its OWN strategy instance regardless — required because ES/SNES carry per-generation
    // sampling state (noise/z vectors), so a shared instance would corrupt across islands. Combined with
    // island migration (best μ reseeds stagnant islands), a mixed portfolio gives an automatic
    // explorer→refiner handoff: e.g. ES explores, then a stagnating CEM island refines its μ. See
    // docs/engine-sweep.md. The population is split evenly across islands (the 1:1:… ratio).
    public List<UpdateStrategyType>? IslandStrategies { get; set; } = null;

    // CEM parameters (tuned via systematic sweep — see scratch/cem_parameter_sweep.md)
    public float CEMEliteFraction { get; set; } = 0.01f;
    public float CEMMuSmoothing { get; set; } = 0.2f;
    public float CEMSigmaSmoothing { get; set; } = 0.3f;

    // ES parameters
    public float ESSigma { get; set; } = 0.05f;
    public float ESLearningRate { get; set; } = 0.01f;
    public float ESAdamBeta1 { get; set; } = 0.9f;
    public float ESAdamBeta2 { get; set; } = 0.999f;

    // SNES parameters (Wierstra et al. 2014)
    public float SNESEtaMu { get; set; } = 1.0f;
    public float SNESEtaSigma { get; set; } = 0.2f;
    public bool SNESMirrored { get; set; } = false;

    // Shared (tuned via systematic sweep — see scratch/cem_parameter_sweep.md)
    public float InitialSigma { get; set; } = 0.25f;
    public float MinSigma { get; set; } = 0.08f;
    public float MaxSigma { get; set; } = 2.0f;

    // Island lifecycle
    public int StagnationThreshold { get; set; } = 30;
    public float ReinitSigma { get; set; } = 0.1f;
}

public enum UpdateStrategyType
{
    CEM,
    ES,
    SNES
}
