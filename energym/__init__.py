from gym.envs.registration import register

register(
        id='smartgrid-v0',
        entry_point='energym.envs:SmartgridEnv',
        )
