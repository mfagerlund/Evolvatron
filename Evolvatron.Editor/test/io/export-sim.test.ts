import { describe, it, expect, beforeEach } from 'vitest';
import { exportSimWorld, obstacleToSim } from '../../src/io/export-sim';
import { createDefaultWorld, createObstacle, createCheckpoint, createDangerZone, createSpeedZone, resetIdCounter } from '../../src/model/defaults';

describe('export-sim', () => {
  beforeEach(() => resetIdCounter());

  describe('obstacleToSim', () => {
    it('converts 0-degree obstacle', () => {
      const obs = createObstacle(3, 4);
      const sim = obstacleToSim(obs);
      expect(sim.CX).toBe(3);
      expect(sim.CY).toBe(4);
      expect(sim.UX).toBeCloseTo(1);
      expect(sim.UY).toBeCloseTo(0);
      expect(sim.HalfExtentX).toBe(1);
      expect(sim.HalfExtentY).toBe(0.25);
      expect(sim.IsLethal).toBe(false);
    });

    it('converts 90-degree obstacle', () => {
      const obs = createObstacle(0, 0);
      obs.rotation = 90;
      const sim = obstacleToSim(obs);
      expect(sim.UX).toBeCloseTo(0, 5);
      expect(sim.UY).toBeCloseTo(1, 5);
    });

    it('converts 45-degree obstacle', () => {
      const obs = createObstacle(0, 0);
      obs.rotation = 45;
      const sim = obstacleToSim(obs);
      expect(sim.UX).toBeCloseTo(Math.cos(Math.PI / 4), 5);
      expect(sim.UY).toBeCloseTo(Math.sin(Math.PI / 4), 5);
    });

    it('preserves lethal flag', () => {
      const obs = createObstacle(0, 0);
      obs.isLethal = true;
      expect(obstacleToSim(obs).IsLethal).toBe(true);
    });
  });

  describe('exportSimWorld', () => {
    it('exports landing pad', () => {
      const world = createDefaultWorld();
      const sim = exportSimWorld(world);
      expect(sim.LandingPad.PadX).toBe(0);
      expect(sim.LandingPad.PadY).toBe(-4.5);
      expect(sim.LandingPad.PadHalfWidth).toBe(2);
      expect(sim.LandingPad.LandingBonus).toBe(100);
    });

    it('exports spawn area', () => {
      const world = createDefaultWorld();
      const sim = exportSimWorld(world);
      expect(sim.Spawn.X).toBe(0);
      expect(sim.Spawn.Y).toBe(15);
      expect(sim.Spawn.XRange).toBe(8);
    });

    it('exports obstacles', () => {
      const world = createDefaultWorld();
      world.modules.push(createObstacle(5, 2));
      const sim = exportSimWorld(world);
      expect(sim.Obstacles).toHaveLength(1);
      expect(sim.Obstacles[0].CX).toBe(5);
    });

    it('exports all module types', () => {
      const world = createDefaultWorld();
      world.modules.push(createObstacle(1, 1));
      world.modules.push(createCheckpoint(2, 2, 0));
      world.modules.push(createSpeedZone(3, 3));
      world.modules.push(createDangerZone(4, 4));
      const sim = exportSimWorld(world);
      expect(sim.Obstacles).toHaveLength(1);
      expect(sim.Checkpoints).toHaveLength(1);
      expect(sim.Checkpoints[0].Order).toBe(0);
      expect(sim.SpeedZones).toHaveLength(1);
      expect(sim.SpeedZones[0].MaxSpeed).toBe(5);
      expect(sim.DangerZones).toHaveLength(1);
      expect(sim.DangerZones[0].PenaltyPerStep).toBe(0.5);
    });

    it('exports simulation config', () => {
      const world = createDefaultWorld();
      const sim = exportSimWorld(world);
      expect(sim.SimulationConfig.Dt).toBeCloseTo(1 / 120);
      expect(sim.SimulationConfig.GravityY).toBe(-9.81);
      expect(sim.SimulationConfig.SolverIterations).toBe(6);
    });

    it('exports reward weights', () => {
      const world = createDefaultWorld();
      const sim = exportSimWorld(world);
      expect(sim.RewardWeights.PositionWeight).toBe(1.0);
      expect(sim.RewardWeights.VelocityWeight).toBe(0.5);
    });
  });
});
