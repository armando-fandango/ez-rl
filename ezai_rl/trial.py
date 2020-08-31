# t_max = 0 : run until done
# t_max > 0 : run each episode for t_max steps at max
# g_max = 0 : run until done
# g_max > 0 : run each episode for g_max return at max
from tqdm import trange
import numpy as np
import sys
from .policies import base_policy
import matplotlib.pyplot as plt

import csv

def run_episode(env, policy, env_render_mode, episode_num=0,
                t_max=0, g_max=0, r_max=0,
                learn_mode = False, debug=False,
                save_rewards=False, save_frames=False ):
    #0. initialize the counters
    rewards_filename = 'rewards.csv'
    frames_filename = 'frames.csv'
    g = 0
    t = 0
    episode_done = False
    episode_rewards=[]
    episode_frames=[]

    #1. observe initial state
    s = env.reset()

    while not episode_done:

        #2. select an action, and observe the next state
        a = policy.get_action(s)
        s_, r, episode_done, info = env.step(a)

        if debug:
            print('SARS=',s,a,r,s_, episode_done)
            #env.render()
        if save_frames:
            episode_frames.append(env.render(mode = env_render_mode))
        if save_rewards:
            episode_rewards.append(r)

        #3. update the policy internals if policy is in learn_mode
        policy.update(s,a,r,s_, episode_done)

        # set next state as current state
        s = s_

        # update the counters
        g += r
        t += 1

        if (r_max and r >= r_max) or (g_max and g >= g_max) or (t_max and t >= t_max):
            break

    # Episode End Processing
    with open(rewards_filename, mode='a+') as rewards_file:
        writer = csv.writer(rewards_file,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(episode_rewards)
        rewards_file.flush()
    #TODO Frames save logic

    if policy.learn_mode:
        # decay the epsilon i.e. explore rate
        policy.decay_er(episode_num)

    return g, episode_rewards


def run_episodes(env, policy:base_policy, n_episodes=1,
                 t_max=0, g_max=0, r_max=0,
                 learn_mode = False, debug=False,
                 save_rewards=0, save_frames=0):
    """

    :param env:
    :param policy: Which policy to use
    :param n_episodes: How many episodes to run it for
    :param t_max: >0 until time-steps at max or episode_done, =0 until episode_done
    :param g_max: >0 until g return at max or episode_done, =0 until episode_done
    :param r_max: >0 until r reward at max or episode_done, =0 until episode_done
    :param learn_mode:
    :param debug:
    :return:
    """
    returns = np.zeros(shape=(n_episodes))
    policy.learn_mode = learn_mode

    if save_frames:
        if 'rgb_array' in env.metadata['render.modes']:
            render_mode = 'rgb_array'
        else:
            render_mode = 'ansi'
    else:
        render_mode = None

    for i in trange(n_episodes, file=sys.stdout, dynamic_ncols=True ):

        g, episode_rewards = run_episode(env=env, policy=policy, env_render_mode = render_mode,
                    episode_num = i,
                       t_max=t_max, g_max=g_max, r_max=r_max,
                       learn_mode = learn_mode, debug=debug,
                       save_rewards=save_rewards, save_frames=save_frames)
        # Episode End Processing
        #TODO Frames save logic

        returns[i] = g

        if debug and (i + 1) % 100 == 0:
            print(
                'E#={}, G= Mean:{:0.2f},Min:{:0.2f},Max:{:0.2f}'
                    .format(i+1,
                            episode_rewards.mean(),
                            episode_rewards.min(),
                            episode_rewards.max()))
    if n_episodes > 1:
        print(
            'Policy:{}, E\'s={}, G= Mean:{:0.2f},Min:{:0.2f},Max:{:0.2f}'.
                format(policy.__name__, n_episodes, np.mean(returns),
                       np.min(returns), np.max(returns)))
        plt.plot(returns)
        plt.title('Environment {} with algo {}'.format(env.spec.id,policy.__name__))
        plt.xlabel('Episode #')
        plt.ylabel('Cumulative Rewards (G)')
        plt.show()
    else:
        print('Policy:{}, G= {}'.format(policy.__name__, returns[0]))

    return returns