def env_info(env):
    if env.spec:
        print(env.spec.id)
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space)
    print('Reward Range:', env.reward_range)
    print('Metadata:', env.metadata)
    print('Initial State:',env.reset())
    print('First Step State:',env.step(env.action_space.sample()))


