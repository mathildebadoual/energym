from gym.envs.registration import register

register(
        id='energy_market-v0',
        entry_point='energym.envs:EnergyMarketEnv',
        )

register(
        id='battery-v0',
        entry_point='energym.envs:BatteryEnv',
)
