using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Fused step kernel for double pole balancing using dense NN forward pass.
/// Physics identical to DoublePoleStepKernel. Only the NN call differs.
/// One thread per world. Each launch processes TicksPerLaunch ticks.
/// </summary>
public static class DenseDoublePoleStepKernel
{
    // Physics constants matching Colonel's DoublePoleCart
    private const float Gravity = -9.8f;
    private const float MassCart = 1.0f;
    private const float Length1 = 0.5f;
    private const float MassPole1 = 0.1f;
    private const float Length2 = 0.05f;
    private const float MassPole2 = 0.01f;
    private const float ForceMag = 10.0f;
    private const float TimeDelta = 0.01f;
    private const float Mup = 0.000002f;

    private const float ML1 = Length1 * MassPole1;
    private const float ML2 = Length2 * MassPole2;

    public static void StepKernel(
        Index1D worldIdx,
        DenseNNViews nn,
        DoublePoleEpisodeViews episode,
        DenseDoublePoleConfig config,
        ArrayView<float> observations,
        ArrayView<float> actions,
        ArrayView<float> state)
    {
        if (episode.IsTerminal[worldIdx] != 0) return;

        int stateBase = worldIdx * 6;

        float s0 = state[stateBase];
        float s1 = state[stateBase + 1];
        float s2 = state[stateBase + 2];
        float s3 = state[stateBase + 3];
        float s4 = state[stateBase + 4];
        float s5 = state[stateBase + 5];

        int steps = episode.StepCounters[worldIdx];
        int obsBase = worldIdx * config.InputSize;
        int actBase = worldIdx * config.OutputSize;
        int ctxBase = worldIdx * config.ContextSize;

        for (int tick = 0; tick < config.TicksPerLaunch; tick++)
        {
            // === 1. Observations ===
            if (config.IncludeVelocity != 0)
            {
                observations[obsBase]     = s0 / config.TrackLengthHalf;
                observations[obsBase + 1] = s1 / 5f;
                observations[obsBase + 2] = s2 / config.PoleAngleThreshold;
                observations[obsBase + 3] = s3 / 5f;
                observations[obsBase + 4] = s4 / config.PoleAngleThreshold;
                observations[obsBase + 5] = s5 / 5f;
            }
            else
            {
                observations[obsBase]     = s0 / config.TrackLengthHalf;
                observations[obsBase + 1] = s2 / config.PoleAngleThreshold;
                observations[obsBase + 2] = s4 / config.PoleAngleThreshold;
            }

            // === 1b. Recurrent context → extra input slots ===
            if (config.ContextSize > 0)
            {
                int ctxObsStart = obsBase + config.InputSize - config.ContextSize;
                for (int c = 0; c < config.ContextSize; c++)
                    observations[ctxObsStart + c] = episode.ContextBuffer[ctxBase + c];
            }

            // === 2. Dense NN forward pass ===
            DenseNN.ForwardPass(nn, observations, actions, worldIdx, config);

            // === 3. Action + context store ===
            float action = actions[actBase];
            action = XMath.Max(-1f, XMath.Min(1f, action));

            if (config.ContextSize > 0)
            {
                int ctxOutStart = actBase + config.OutputSize - config.ContextSize;
                for (int c = 0; c < config.ContextSize; c++)
                    episode.ContextBuffer[ctxBase + c] = actions[ctxOutStart + c];
            }

            // === 4. Physics: 2x RK4 ===
            PerformRK4(action, ref s0, ref s1, ref s2, ref s3, ref s4, ref s5);
            PerformRK4(action, ref s0, ref s1, ref s2, ref s3, ref s4, ref s5);

            steps++;

            // === 5. Jiggle tracking ===
            int jiggleBase = worldIdx * 100;
            int jiggleIdx = (steps - 1) % 100;
            float jiggleVal = XMath.Abs(s0) + XMath.Abs(s1) + XMath.Abs(s2) + XMath.Abs(s3);
            episode.JiggleBuffer[jiggleBase + jiggleIdx] = jiggleVal;

            // === 6. Terminal check ===
            bool outOfBounds =
                s0 < -config.TrackLengthHalf || s0 > config.TrackLengthHalf ||
                s2 > config.PoleAngleThreshold || s2 < -config.PoleAngleThreshold ||
                s4 > config.PoleAngleThreshold || s4 < -config.PoleAngleThreshold;

            if (outOfBounds || steps >= config.MaxSteps)
            {
                episode.IsTerminal[worldIdx] = 1;

                float fitness = (float)steps;

                if (steps >= config.MaxSteps && config.GruauEnabled != 0)
                {
                    float jiggleSum = 0f;
                    for (int j = 0; j < 100; j++)
                        jiggleSum += episode.JiggleBuffer[jiggleBase + j];
                    if (jiggleSum > 0.001f)
                        fitness += 0.75f * config.MaxSteps / jiggleSum;
                }

                episode.FitnessValues[worldIdx] = fitness;
                break;
            }
        }

        episode.StepCounters[worldIdx] = steps;

        state[stateBase]     = s0;
        state[stateBase + 1] = s1;
        state[stateBase + 2] = s2;
        state[stateBase + 3] = s3;
        state[stateBase + 4] = s4;
        state[stateBase + 5] = s5;
    }

    private static void PerformRK4(
        float action,
        ref float s0, ref float s1, ref float s2, ref float s3, ref float s4, ref float s5)
    {
        float hh = TimeDelta * 0.5f;
        float h6 = TimeDelta / 6f;

        float d0 = s1, d2 = s3, d4 = s5;
        ComputeAccelerations(action, s0, s1, s2, s3, s4, s5,
            out float d1, out float d3, out float d5);

        float yt0 = s0 + hh * d0, yt1 = s1 + hh * d1, yt2 = s2 + hh * d2;
        float yt3 = s3 + hh * d3, yt4 = s4 + hh * d4, yt5 = s5 + hh * d5;

        float dyt0 = yt1, dyt2 = yt3, dyt4 = yt5;
        ComputeAccelerations(action, yt0, yt1, yt2, yt3, yt4, yt5,
            out float dyt1, out float dyt3, out float dyt5);

        yt0 = s0 + hh * dyt0; yt1 = s1 + hh * dyt1; yt2 = s2 + hh * dyt2;
        yt3 = s3 + hh * dyt3; yt4 = s4 + hh * dyt4; yt5 = s5 + hh * dyt5;

        float dym0 = yt1, dym2 = yt3, dym4 = yt5;
        ComputeAccelerations(action, yt0, yt1, yt2, yt3, yt4, yt5,
            out float dym1, out float dym3, out float dym5);

        yt0 = s0 + TimeDelta * dym0; yt1 = s1 + TimeDelta * dym1; yt2 = s2 + TimeDelta * dym2;
        yt3 = s3 + TimeDelta * dym3; yt4 = s4 + TimeDelta * dym4; yt5 = s5 + TimeDelta * dym5;
        dym0 += dyt0; dym1 += dyt1; dym2 += dyt2;
        dym3 += dyt3; dym4 += dyt4; dym5 += dyt5;

        dyt0 = yt1; dyt2 = yt3; dyt4 = yt5;
        ComputeAccelerations(action, yt0, yt1, yt2, yt3, yt4, yt5,
            out dyt1, out dyt3, out dyt5);

        s0 += h6 * (d0 + dyt0 + 2f * dym0);
        s1 += h6 * (d1 + dyt1 + 2f * dym1);
        s2 += h6 * (d2 + dyt2 + 2f * dym2);
        s3 += h6 * (d3 + dyt3 + 2f * dym3);
        s4 += h6 * (d4 + dyt4 + 2f * dym4);
        s5 += h6 * (d5 + dyt5 + 2f * dym5);
    }

    private static void ComputeAccelerations(
        float action,
        float s0, float s1, float s2, float s3, float s4, float s5,
        out float cartAcc, out float pole1Acc, out float pole2Acc)
    {
        float force = action * ForceMag;
        float costheta1 = XMath.Cos(s2);
        float sintheta1 = XMath.Sin(s2);
        float gsintheta1 = Gravity * sintheta1;
        float costheta2 = XMath.Cos(s4);
        float sintheta2 = XMath.Sin(s4);
        float gsintheta2 = Gravity * sintheta2;

        float temp1 = Mup * s3 / ML1;
        float temp2 = Mup * s5 / ML2;

        float fi1 = ML1 * s3 * s3 * sintheta1 +
                     0.75f * MassPole1 * costheta1 * (temp1 + gsintheta1);
        float fi2 = ML2 * s5 * s5 * sintheta2 +
                     0.75f * MassPole2 * costheta2 * (temp2 + gsintheta2);

        float mi1 = MassPole1 * (1f - 0.75f * costheta1 * costheta1);
        float mi2 = MassPole2 * (1f - 0.75f * costheta2 * costheta2);

        cartAcc = (force + fi1 + fi2) / (mi1 + mi2 + MassCart);
        pole1Acc = -0.75f * (cartAcc * costheta1 + gsintheta1 + temp1) / Length1;
        pole2Acc = -0.75f * (cartAcc * costheta2 + gsintheta2 + temp2) / Length2;
    }
}
