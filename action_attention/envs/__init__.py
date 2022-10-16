from gym.envs.registration import register

register(
    'ShapesTrain-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=50,
    kwargs={'render_type': 'shapes'},
)

register(
    'CubesTrain-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='action_attention.envs.block_pushing:BlockPushing',
    max_episode_steps=50,
    kwargs={'render_type': 'cubes'},
)
