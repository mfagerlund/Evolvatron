namespace Evolvatron.Evolvion.ES;

public class IslandConfig
{
    // Population structure
    public int IslandCount { get; set; } = 5;
    public int MinIslandPop { get; set; } = 512;

    // Strategy selection
    public UpdateStrategyType Strategy { get; set; } = UpdateStrategyType.CEM;

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
