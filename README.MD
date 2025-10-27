# Evolvatron

Evolvatron is a game where the player performs rocket systems design and reward engineering.
Think moon lander in continous space where the rocket starts further away with some obstacles in 
the way. It's the players job to set up a reward landscape to coax the rocket to arrive at the 
correct location at the correct speed to perform a good landing. 

The player designs the reward landscape "live" where the playback history mostly remains
valid during edits. If the agent is externally modified (new effectors, new sensors) the replay 
buffer will need to be cleared and the agent started from scratch. So I expect the user will work
on a rocket design and then on the rewards.

The rocket design will be a 2d editor where rocket components are pieced together, you 
might have a sensor mounted on a gimbal and two motors separated for differential 
steering. Or two motors mounted on gimbals. The sky is the limit.

Note that the agents will NOT be trained on pixels, they will be given clear signals
from the envionment, 2-15 signals and 2-25 outputs.

## GPU vs CPU Performance

**CPUStepper**: Pure C# reference implementation. Use for gameplay and weak GPUs.

**GPUStepper**: ILGPU-accelerated for batch simulations (thousands of parallel agents).
Only use with actual CUDA/OpenCL GPUs. CPU fallback mode has overhead from memory
transfers and missing features (velocity stabilization, friction, damping).
Not recommended for gameplay.