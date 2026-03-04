# Evolvatron

A game about designing rockets and teaching them to fly.

## What Is It?

Think moon lander — but you're not flying the rocket yourself. Instead, you're the engineer behind the scenes, solving two problems:

1. **Design the rocket.** A 2D editor lets you piece together components: engines, gimbals, sensors, steering systems. Mount two motors on gimbals for fine control, or go with differential thrust for simplicity. The sky is the limit.

2. **Design the reward landscape.** Your rocket has a brain, but it doesn't know what "good" means yet. That's your job. You shape the rewards — how much it cares about getting close to the pad, how much it penalizes fuel use, how harshly it's punished for coming in too fast. Get the balance right and the rocket teaches itself to land. Get it wrong and it flies off into the void, or slams into the ground, or finds some creative loophole you never imagined.

The rocket starts far away. There are obstacles in the way. The landing pad is small. Your agent has to figure it out — guided only by the reward landscape you built.

## How Does It Play?

You work in two phases.

**Rocket design** comes first. You build your vehicle in the editor, choosing what sensors it has, how its engines are arranged, what it can perceive and what it can control. This defines the agent's inputs and outputs — not pixels, just clean signals. Maybe 5 inputs and 3 outputs. Maybe 15 and 25. It depends on your design.

**Reward engineering** comes second. Once your rocket is built, you switch to the reward editor. Here you tune the landscape live — adjusting weights, adding terms, watching how the agent's behavior shifts in response. The playback history stays valid as long as you're only changing rewards. Swap out a sensor or add an engine, and training starts fresh.

The core loop is: tweak the rewards, watch the rocket train, see what happens, tweak again. You're not writing code or tuning hyperparameters. You're shaping behavior through incentives, and the rocket responds.

## Why Is It Interesting?

Reward engineering is one of the hardest problems in training autonomous agents. Small changes to the reward function produce wildly different behavior. This game puts that challenge front and center — not as a chore, but as the puzzle.

You get to experience firsthand why a rocket that's rewarded for "being close to the pad" might learn to hover forever instead of landing. Why penalizing fuel too aggressively produces kamikaze trajectories. Why the difference between a graceful landing and a catastrophic failure sometimes comes down to a single weight in your reward function.

It's a game about unintended consequences, creative problem-solving, and the surprisingly deep challenge of telling a machine what you actually want.

## Documentation

See `CLAUDE.md` for architecture, build commands, and technical details.
