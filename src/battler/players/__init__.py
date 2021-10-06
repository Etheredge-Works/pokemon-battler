import gym
gym.register(
    id='Pokemon-v7',
    entry_point='battler.players.players:RLPlayer7',
    #max_episode_steps=1000,
    #nondeterministic=True,
)


gym.register(
    id='Pokemon-v8',
    entry_point='battler.players.players:RLPlayer8',
    # TODO this adds wraps?
    #max_episode_steps=1000,
    #nondeterministic=True,
)
