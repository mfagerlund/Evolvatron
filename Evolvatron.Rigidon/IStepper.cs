namespace Evolvatron.Core;

/// <summary>
/// Interface for stepping the simulation forward in time.
/// </summary>
public interface IStepper
{
    /// <summary>
    /// Advances the simulation by one step (cfg.Dt * cfg.Substeps).
    /// </summary>
    void Step(WorldState world, SimulationConfig cfg);
}
