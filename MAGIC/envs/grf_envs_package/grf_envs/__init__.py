from gym.envs.registration import register

register(
    id='GRFWrapper-v0',
    entry_point='envs.grf_envs_package.grf_envs.grf_wrapper_env:GRFWrapperEnv',
)

