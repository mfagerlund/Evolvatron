using System;

namespace Evolvatron.Core.Rockets;

/// <summary>
/// Canonical rocket specs. <see cref="StockRocket"/> reproduces — exactly — the 3-body rocket
/// hardcoded across the GPU evaluators (fuselage + 2 legs, 5+7+7 geoms, 2 motor-locked joints,
/// thrust+gimbal on body 0). This is the parity anchor: any spec→engine factory must match it.
/// </summary>
public static class RocketSpecLibrary
{
    // Stock rocket dimensions (mirror GPUDense*Evaluator.CreateAndUploadRocketTemplate).
    public const float BodyHeight = 1.5f, BodyRadius = 0.2f, BodyMass = 8f;
    public const float LegLength = 1.0f, LegRadius = 0.1f, LegMass = 1.5f;
    const float BodyAngle = MathF.PI / 2f;                 // upright (thrust axis = world up)
    const float LeftLegAngle = 225f * MathF.PI / 180f;
    const float RightLegAngle = 315f * MathF.PI / 180f;

    public static RocketSpec StockRocket()
    {
        float bodyHalf = BodyHeight * 0.5f;   // 0.75
        float legHalf = LegLength * 0.5f;     // 0.5
        float bodyInertia = BodyMass * (BodyRadius * BodyRadius * 0.25f + BodyHeight * BodyHeight / 12f);
        float legInertia = LegMass * (LegRadius * LegRadius * 0.25f + LegLength * LegLength / 12f);

        // Geom counts match the evaluators' Clamp((halfLen/radius)+2, 3, 7): body→5, leg→7.
        int bodyGeoms = Math.Clamp((int)(bodyHalf / BodyRadius) + 2, 3, 7);
        int legGeoms = Math.Clamp((int)(legHalf / LegRadius) + 2, 3, 7);

        var spec = new RocketSpec { Name = "stock" };

        // Body 0: vertical fuselage, center 0.75 above the origin.
        var body = new BodySpec { X = 0f, Y = bodyHalf, Angle = BodyAngle, Mass = BodyMass, Inertia = bodyInertia };
        for (int i = 0; i < bodyGeoms; i++)
        {
            float t = (float)i / (bodyGeoms - 1);
            body.Geoms.Add(new GeomSpec(-bodyHalf + t * BodyHeight, 0f, BodyRadius));
        }
        spec.Bodies.Add(body);

        // Bodies 1,2: legs at 225° / 315°, centers offset by legHalf along each leg's axis.
        AddLeg(spec, LeftLegAngle, legHalf, legInertia, legGeoms);
        AddLeg(spec, RightLegAngle, legHalf, legInertia, legGeoms);

        // Joints: body ↔ each leg, motor-locked (speed 0) so the legs hold their splay.
        spec.Joints.Add(new JointSpec
        {
            BodyA = 0, BodyB = 1,
            AnchorAX = -bodyHalf, AnchorAY = 0f, AnchorBX = -legHalf, AnchorBY = 0f,
            ReferenceAngle = LeftLegAngle - BodyAngle,
            EnableMotor = true, MotorSpeed = 0f, MaxMotorTorque = 1000f
        });
        spec.Joints.Add(new JointSpec
        {
            BodyA = 0, BodyB = 2,
            AnchorAX = -bodyHalf, AnchorAY = 0f, AnchorBX = -legHalf, AnchorBY = 0f,
            ReferenceAngle = RightLegAngle - BodyAngle,
            EnableMotor = true, MotorSpeed = 0f, MaxMotorTorque = 1000f
        });

        // Single gimbaled main thruster on body 0, firing along its +X axis (world up at rest).
        spec.Thrusters.Add(new ThrusterSpec
        {
            BodyIndex = 0, LocalDirX = 1f, LocalDirY = 0f,
            MaxThrust = 200f, Gimbal = true, MaxGimbalTorque = 50f
        });

        // 8 body-frame raycast sensors at 45° spacing (the maze nav-policy sensor set).
        for (int i = 0; i < 8; i++)
            spec.Sensors.Add(new SensorSpec { BodyIndex = 0, AngleOffset = i * MathF.PI / 4f, MaxRange = 15f });

        return spec;
    }

    static void AddLeg(RocketSpec spec, float legAngle, float legHalf, float legInertia, int legGeoms)
    {
        var leg = new BodySpec
        {
            X = MathF.Cos(legAngle) * legHalf,
            Y = MathF.Sin(legAngle) * legHalf,
            Angle = legAngle,
            Mass = LegMass,
            Inertia = legInertia
        };
        for (int i = 0; i < legGeoms; i++)
        {
            float t = (float)i / (legGeoms - 1);
            leg.Geoms.Add(new GeomSpec(-legHalf + t * LegLength, 0f, LegRadius));
        }
        spec.Bodies.Add(leg);
    }
}
