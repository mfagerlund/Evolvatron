using ILGPU;
using ILGPU.Algorithms;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Phase-2 maze navigator step kernel (see docs/phase2_maze_spec.md).
///
/// Hierarchical, two networks per step in one fused launch:
///   NN#1 (maze policy, evolved per-world): maze obs → desired velocity command (gentle, ≤CmdSpeedMax)
///   NN#2 (controller, FROZEN shared):      9-D dynamics+command → throttle, gimbal
/// then physics substep + progress/goal/collision/tumble/timeout terminals.
///
/// The controller is identical to Phase 1 and its weights are shared across all worlds — the
/// forward pass uses weightWorldIdx=0 (one shared net) but ioWorldIdx=worldIdx (per-world I/O).
/// One thread per world.
/// </summary>
public static class DenseMazeStepKernel
{
    private const float HalfPi = 1.5707963267949f;
    private const int CtrlInputSize = 9;
    private const int CtrlOutputSize = 2;

    public static void StepKernel(
        Index1D worldIdx,
        PhysicsViews physics,
        DenseNNViews mazeNN,
        EpisodeViews episode,
        MegaKernelConfig config,
        DenseMazeConfig maze,
        MazeViews mv,
        ArrayView<float> mazeObs,
        ArrayView<float> mazeAct)
    {
        if (episode.IsTerminal[worldIdx] != 0) return;

        int bodiesPerWorld = config.BodiesPerWorld;
        int bodyBase = worldIdx * bodiesPerWorld;

        // === COM + velocity (mass-weighted) ===
        float comX = 0f, comY = 0f, velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float m = 1f / b.InvMass;
            comX += b.X * m; comY += b.Y * m; velX += b.VelX * m; velY += b.VelY * m; totalMass += m;
        }
        if (totalMass > 0f) { float inv = 1f / totalMass; comX *= inv; comY *= inv; velX *= inv; velY *= inv; }

        var body0 = physics.Bodies[bodyBase];
        float upX = XMath.Cos(body0.Angle);
        float upY = XMath.Sin(body0.Angle);
        float rightX = upY, rightY = -upX;
        float angVel = body0.AngularVel;
        float speed = XMath.Sqrt(velX * velX + velY * velY);
        float goalX = mv.GoalX[worldIdx];
        float goalY = mv.GoalY[worldIdx];

        // === Maze observation (NN#1 input): goal-rel pos, velocity, attitude, body-frame sensors ===
        int mObsBase = worldIdx * maze.MazeInputSize;
        mazeObs[mObsBase + 0] = (goalX - comX) / maze.PosScale;
        mazeObs[mObsBase + 1] = (goalY - comY) / maze.PosScale;
        mazeObs[mObsBase + 2] = velX / 10f;
        mazeObs[mObsBase + 3] = velY / 10f;
        mazeObs[mObsBase + 4] = upX;
        mazeObs[mObsBase + 5] = upY;

        if (config.SensorCount >= 4)
        {
            float maxRange = config.MaxSensorRange;
            float d0 = maxRange, d1 = maxRange, d2 = maxRange, d3 = maxRange;
            for (int ci = 0; ci < config.SharedColliderCount; ci++)
            {
                var obb = physics.SharedOBBColliders[ci];
                float t;
                t = RayVsOBB(comX, comY, upX, upY, maxRange, obb); if (t < d0) d0 = t;
                t = RayVsOBB(comX, comY, -upX, -upY, maxRange, obb); if (t < d1) d1 = t;
                t = RayVsOBB(comX, comY, -upY, upX, maxRange, obb); if (t < d2) d2 = t;
                t = RayVsOBB(comX, comY, upY, -upX, maxRange, obb); if (t < d3) d3 = t;
            }
            mazeObs[mObsBase + 6] = d0 / maxRange;
            mazeObs[mObsBase + 7] = d1 / maxRange;
            mazeObs[mObsBase + 8] = d2 / maxRange;
            mazeObs[mObsBase + 9] = d3 / maxRange;

            if (config.SensorCount >= 8)
            {
                const float inv2 = 0.70710678f;
                float dax = (upX + rightX) * inv2, day = (upY + rightY) * inv2;   // up-right diagonal
                float dbx = (upX - rightX) * inv2, dby = (upY - rightY) * inv2;   // up-left diagonal
                float e0 = maxRange, e1 = maxRange, e2 = maxRange, e3 = maxRange;
                for (int ci = 0; ci < config.SharedColliderCount; ci++)
                {
                    var obb = physics.SharedOBBColliders[ci];
                    float t;
                    t = RayVsOBB(comX, comY, dax, day, maxRange, obb); if (t < e0) e0 = t;
                    t = RayVsOBB(comX, comY, -dax, -day, maxRange, obb); if (t < e1) e1 = t;
                    t = RayVsOBB(comX, comY, dbx, dby, maxRange, obb); if (t < e2) e2 = t;
                    t = RayVsOBB(comX, comY, -dbx, -dby, maxRange, obb); if (t < e3) e3 = t;
                }
                mazeObs[mObsBase + 10] = e0 / maxRange;
                mazeObs[mObsBase + 11] = e1 / maxRange;
                mazeObs[mObsBase + 12] = e2 / maxRange;
                mazeObs[mObsBase + 13] = e3 / maxRange;
            }
        }

        // === NN#1: maze policy (per-world weights) → velocity command ===
        DenseNN.ForwardPass(
            mazeNN.Weights, mazeNN.Biases, mazeNN.LayerSizes,
            mazeObs, mazeAct, worldIdx,
            maze.MazeNumLayers, maze.MazeTotalWeights, maze.MazeTotalBiases,
            maze.MazeInputSize, maze.MazeOutputSize);

        int mActBase = worldIdx * maze.MazeOutputSize;
        float rawCmdX = mazeAct[mActBase + 0];
        float rawCmdY = mazeAct[mActBase + 1];
        if (float.IsNaN(rawCmdX) || float.IsInfinity(rawCmdX) || float.IsNaN(rawCmdY) || float.IsInfinity(rawCmdY))
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = -1e6f;
            mv.RewardAccum[worldIdx] = -1e6f;
            return;
        }
        float cmdVx = XMath.Max(-1f, XMath.Min(1f, rawCmdX)) * maze.CmdSpeedMax;
        float cmdVy = XMath.Max(-1f, XMath.Min(1f, rawCmdY)) * maze.CmdSpeedMax;

        // === Controller observation (NN#2 input, 9-D — identical to Phase 1) ===
        float errWX = cmdVx - velX, errWY = cmdVy - velY;
        float errFwd = errWX * upX + errWY * upY;
        float errLat = errWX * rightX + errWY * rightY;
        float gmag = XMath.Sqrt(config.GravityX * config.GravityX + config.GravityY * config.GravityY);
        float gUp = gmag > 1e-6f ? -(config.GravityX * upX + config.GravityY * upY) / gmag : 0f;

        int cObsBase = worldIdx * CtrlInputSize;
        mv.CtrlObs[cObsBase + 0] = upX;
        mv.CtrlObs[cObsBase + 1] = upY;
        mv.CtrlObs[cObsBase + 2] = angVel / 10f;
        mv.CtrlObs[cObsBase + 3] = errFwd / 10f;
        mv.CtrlObs[cObsBase + 4] = errLat / 10f;
        mv.CtrlObs[cObsBase + 5] = speed / 10f;
        mv.CtrlObs[cObsBase + 6] = episode.CurrentThrottle[worldIdx];
        mv.CtrlObs[cObsBase + 7] = episode.CurrentGimbal[worldIdx];
        mv.CtrlObs[cObsBase + 8] = gUp;

        // === NN#2: frozen controller (SHARED weights → weightWorldIdx=0, ioWorldIdx=worldIdx) ===
        DenseNN.ForwardPass(
            mv.CtrlWeights, mv.CtrlBiases, mv.CtrlLayerSizes,
            mv.CtrlObs, mv.CtrlAct, 0, worldIdx,
            maze.CtrlNumLayers, maze.CtrlTotalWeights, maze.CtrlTotalBiases,
            CtrlInputSize, CtrlOutputSize);

        int cActBase = worldIdx * CtrlOutputSize;
        float throttle = mv.CtrlAct[cActBase + 0];
        float gimbal = mv.CtrlAct[cActBase + 1];
        if (float.IsNaN(throttle) || float.IsInfinity(throttle) || float.IsNaN(gimbal) || float.IsInfinity(gimbal))
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = -1e6f;
            mv.RewardAccum[worldIdx] = -1e6f;
            return;
        }
        throttle = XMath.Max(0f, XMath.Min(1f, throttle));
        gimbal = XMath.Max(-1f, XMath.Min(1f, gimbal));
        episode.CurrentThrottle[worldIdx] = throttle;
        episode.CurrentGimbal[worldIdx] = gimbal;

        // === Apply actuators (body-axis thrust + gimbal torque) ===
        var body = physics.Bodies[bodyBase];
        float cos = XMath.Cos(body.Angle), sin = XMath.Sin(body.Angle);
        float thrust = throttle * config.MaxThrust;
        float dt = config.Dt;
        body.VelX += cos * thrust * body.InvMass * dt;
        body.VelY += sin * thrust * body.InvMass * dt;
        body.AngularVel += gimbal * config.MaxGimbalTorque * body.InvInertia * dt;
        physics.Bodies[bodyBase] = body;

        // === Physics substep (free space + obstacles via SharedOBBColliders) ===
        InlinePhysics.SubStepOneWorld(physics, worldIdx, config);

        int steps = episode.StepCounters[worldIdx] + 1;
        episode.StepCounters[worldIdx] = steps;

        // === Post-physics COM + goal distance ===
        float comX2 = 0f, comY2 = 0f, totalMass2 = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float m = 1f / b.InvMass;
            comX2 += b.X * m; comY2 += b.Y * m; totalMass2 += m;
        }
        if (totalMass2 > 0f) { float inv = 1f / totalMass2; comX2 *= inv; comY2 *= inv; }

        var body0post = physics.Bodies[bodyBase];
        float upX2 = XMath.Cos(body0post.Angle), upY2 = XMath.Sin(body0post.Angle);
        float angleErr = XMath.Abs(XMath.Atan2(upX2, upY2));

        float ddx = goalX - comX2, ddy = goalY - comY2;
        float dist = XMath.Sqrt(ddx * ddx + ddy * ddy);
        float progress = mv.PrevDist[worldIdx] - dist;
        mv.PrevDist[worldIdx] = dist;
        mv.RewardAccum[worldIdx] += maze.ProgressWeight * progress - maze.StepPenalty;

        // === Terminals ===

        // Goal reached (success) — bonus scaled by remaining-time haste. Reuse HasLanded as success flag.
        if (dist < maze.GoalRadius)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.HasLanded[worldIdx] = 1;
            float haste = (float)(config.MaxSteps - steps) / config.MaxSteps;
            mv.RewardAccum[worldIdx] += maze.GoalBonus * (1f + haste);
            episode.FitnessValues[worldIdx] = mv.RewardAccum[worldIdx];
            return;
        }

        // Obstacle collision (fail)
        if (config.ObstacleDeathEnabled != 0)
        {
            int contactBase = worldIdx * config.MaxContactsPerWorld;
            int nContacts = physics.ContactCounts[worldIdx];
            bool hitObstacle = false;
            for (int c = 0; c < nContacts; c++)
            {
                var contact = physics.Contacts[contactBase + c];
                if (contact.IsValid != 0 && contact.ColliderIndex >= config.FirstObstacleIndex)
                    hitObstacle = true;
            }
            if (hitObstacle)
            {
                episode.IsTerminal[worldIdx] = 1;
                mv.RewardAccum[worldIdx] -= maze.CollisionPenalty;
                episode.FitnessValues[worldIdx] = mv.RewardAccum[worldIdx];
                return;
            }
        }

        // Airborne tumble (fail)
        if (angleErr > HalfPi)
        {
            episode.IsTerminal[worldIdx] = 1;
            mv.RewardAccum[worldIdx] -= maze.TumblePenalty;
            episode.FitnessValues[worldIdx] = mv.RewardAccum[worldIdx];
            return;
        }

        // Timeout
        if (steps >= config.MaxSteps)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = mv.RewardAccum[worldIdx];
        }
    }

    /// <summary>Ray vs oriented box — returns hit distance along (dx,dy) or maxRange if no hit.</summary>
    private static float RayVsOBB(float ox, float oy, float dx, float dy, float maxRange, GPUOBBCollider obb)
    {
        float relX = ox - obb.CX;
        float relY = oy - obb.CY;
        float localOX = relX * obb.UX + relY * obb.UY;
        float localOY = -relX * obb.UY + relY * obb.UX;
        float localDX = dx * obb.UX + dy * obb.UY;
        float localDY = -dx * obb.UY + dy * obb.UX;

        float tMin = 0f, tMax = maxRange;

        if (XMath.Abs(localDX) < 1e-8f)
        {
            if (localOX < -obb.HalfExtentX || localOX > obb.HalfExtentX) return maxRange;
        }
        else
        {
            float invD = 1f / localDX;
            float t1 = (-obb.HalfExtentX - localOX) * invD;
            float t2 = (obb.HalfExtentX - localOX) * invD;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = XMath.Max(tMin, t1);
            tMax = XMath.Min(tMax, t2);
            if (tMin > tMax) return maxRange;
        }

        if (XMath.Abs(localDY) < 1e-8f)
        {
            if (localOY < -obb.HalfExtentY || localOY > obb.HalfExtentY) return maxRange;
        }
        else
        {
            float invD = 1f / localDY;
            float t1 = (-obb.HalfExtentY - localOY) * invD;
            float t2 = (obb.HalfExtentY - localOY) * invD;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = XMath.Max(tMin, t1);
            tMax = XMath.Min(tMax, t2);
            if (tMin > tMax) return maxRange;
        }

        return tMin > 0f ? tMin : maxRange;
    }
}
