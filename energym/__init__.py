from gym.envs.registration import register

register(
        id='energy_market-v0',
        entry_point='energym.envs:EnergyMarketEnv',
        )

register(
        id='battery-v0',
        entry_point='energym.envs:BatteryEnv',
)

register(
        id='energy_market_battery-v0',
        entry_point='energym.envs:EnergyMarketBatteryEnvV0',
)

register(
        id='energy_market_battery-v1',
        entry_point='energym.envs:EnergyMarketBatteryEnvV1',
)

register(
        id='spm_environment36',
        entry_point='energym.envs:SPM',
)