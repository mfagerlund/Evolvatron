using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Evolvatron.Core.Rockets;

/// <summary>
/// One circle geom in a body's local frame (rockets are built from capsules = lines of circles).
/// </summary>
public sealed class GeomSpec
{
    public float LocalX { get; set; }
    public float LocalY { get; set; }
    public float Radius { get; set; } = 0.1f;

    public GeomSpec() { }
    public GeomSpec(float localX, float localY, float radius) { LocalX = localX; LocalY = localY; Radius = radius; }
}

/// <summary>
/// One rigid body in the rocket's canonical rest pose (origin = the spawn base point).
/// Placement (X, Y, Angle) is relative to that origin; the factory rotates/translates it to a spawn.
/// </summary>
public sealed class BodySpec
{
    public float X { get; set; }        // center offset from rocket origin, rest pose
    public float Y { get; set; }
    public float Angle { get; set; }    // orientation (radians), rest pose
    public float Mass { get; set; } = 1f;
    public float Inertia { get; set; } = 1f;
    public List<GeomSpec> Geoms { get; set; } = new();
}

/// <summary>Revolute joint between two bodies (by index into RocketSpec.Bodies).</summary>
public sealed class JointSpec
{
    public int BodyA { get; set; }
    public int BodyB { get; set; }
    public float AnchorAX { get; set; }
    public float AnchorAY { get; set; }
    public float AnchorBX { get; set; }
    public float AnchorBY { get; set; }
    public float ReferenceAngle { get; set; }
    public bool EnableLimits { get; set; }
    public float LowerAngle { get; set; }
    public float UpperAngle { get; set; }
    public bool EnableMotor { get; set; } = true;
    public float MotorSpeed { get; set; }
    public float MaxMotorTorque { get; set; } = 1000f;
}

/// <summary>
/// A thruster mounted on a body. Thrust is applied along (LocalDirX, LocalDirY) rotated into world
/// space by the body's angle; throttle ∈ [0,1] scales MaxThrust. If Gimbal, the thruster also
/// contributes a steering torque scaled by gimbal ∈ [-1,1] * MaxGimbalTorque.
/// </summary>
public sealed class ThrusterSpec
{
    public int BodyIndex { get; set; }
    public float LocalDirX { get; set; } = 1f;
    public float LocalDirY { get; set; }
    public float MaxThrust { get; set; } = 200f;
    public bool Gimbal { get; set; } = true;
    public float MaxGimbalTorque { get; set; } = 50f;
}

/// <summary>A body-frame raycast distance sensor (origin = rocket COM; direction = body angle + AngleOffset).</summary>
public sealed class SensorSpec
{
    public int BodyIndex { get; set; }
    public float AngleOffset { get; set; }
    public float MaxRange { get; set; } = 15f;
}

/// <summary>
/// Serializable rocket definition — the single source of truth for rocket geometry, actuation and
/// sensing. Produced by the constructor/editor, consumed by the CPU physics, the GPU evaluators, and
/// replay. See docs/godot_pipeline_plan.md (P0).
/// </summary>
public sealed class RocketSpec
{
    public string Name { get; set; } = "rocket";
    public List<BodySpec> Bodies { get; set; } = new();
    public List<JointSpec> Joints { get; set; } = new();
    public List<ThrusterSpec> Thrusters { get; set; } = new();
    public List<SensorSpec> Sensors { get; set; } = new();

    [JsonIgnore] public int BodyCount => Bodies.Count;
    [JsonIgnore] public int JointCount => Joints.Count;
    [JsonIgnore] public int SensorCount => Sensors.Count;

    [JsonIgnore]
    public int TotalGeoms
    {
        get { int n = 0; foreach (var b in Bodies) n += b.Geoms.Count; return n; }
    }

    /// <summary>
    /// Number of controller outputs (NN action DOFs): one throttle per thruster, plus one shared
    /// gimbal channel if any thruster gimbals. The stock rocket → 1 throttle + 1 gimbal = 2.
    /// </summary>
    [JsonIgnore]
    public int ActuatorDofCount
    {
        get
        {
            bool anyGimbal = false;
            foreach (var t in Thrusters) if (t.Gimbal) anyGimbal = true;
            return Thrusters.Count + (anyGimbal ? 1 : 0);
        }
    }

    static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never
    };

    public string ToJson() => JsonSerializer.Serialize(this, JsonOpts);

    public static RocketSpec FromJson(string json)
        => JsonSerializer.Deserialize<RocketSpec>(json, JsonOpts)
           ?? throw new ArgumentException("RocketSpec JSON deserialized to null");

    /// <summary>Throws if the spec is structurally invalid (bad indices, non-positive mass/inertia, empty bodies).</summary>
    public void Validate()
    {
        if (Bodies.Count == 0)
            throw new InvalidOperationException("RocketSpec has no bodies");
        for (int i = 0; i < Bodies.Count; i++)
        {
            var b = Bodies[i];
            if (b.Mass <= 0f) throw new InvalidOperationException($"Body {i} has non-positive mass {b.Mass}");
            if (b.Inertia <= 0f) throw new InvalidOperationException($"Body {i} has non-positive inertia {b.Inertia}");
            if (b.Geoms.Count == 0) throw new InvalidOperationException($"Body {i} has no geoms");
        }
        foreach (var j in Joints)
        {
            if (j.BodyA < 0 || j.BodyA >= Bodies.Count || j.BodyB < 0 || j.BodyB >= Bodies.Count)
                throw new InvalidOperationException($"Joint references invalid body index ({j.BodyA},{j.BodyB})");
            if (j.BodyA == j.BodyB)
                throw new InvalidOperationException($"Joint connects a body to itself ({j.BodyA})");
        }
        foreach (var t in Thrusters)
            if (t.BodyIndex < 0 || t.BodyIndex >= Bodies.Count)
                throw new InvalidOperationException($"Thruster on invalid body {t.BodyIndex}");
        foreach (var s in Sensors)
            if (s.BodyIndex < 0 || s.BodyIndex >= Bodies.Count)
                throw new InvalidOperationException($"Sensor on invalid body {s.BodyIndex}");
    }

    /// <summary>Throws if the spec exceeds the GPU mega-kernel's fixed-topology limits (see CLAUDE.md GPU-safety note #3).</summary>
    public void ValidateGpuLimits(int maxBodies, int maxGeoms, int maxJoints)
    {
        if (BodyCount > maxBodies)
            throw new InvalidOperationException($"RocketSpec has {BodyCount} bodies > GPU limit {maxBodies}");
        if (TotalGeoms > maxGeoms)
            throw new InvalidOperationException($"RocketSpec has {TotalGeoms} geoms > GPU limit {maxGeoms}");
        if (JointCount > maxJoints)
            throw new InvalidOperationException($"RocketSpec has {JointCount} joints > GPU limit {maxJoints}");
    }
}
