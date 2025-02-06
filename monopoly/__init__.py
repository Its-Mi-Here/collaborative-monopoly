from gymnasium.envs.registration import register

register(
    id="monopoly",
    entry_point="gym_example.envs:MonopolyEnv"
)

register(
    id="simple-monopoly",
    entry_point="gym_example.envs:SimpleMonopolyEnv"
)