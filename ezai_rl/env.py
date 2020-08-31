def env_info(env):
    if env.spec:
        print(env.spec.id)
    print('Action Space:', env.action_space)
    print('Observation Space:', env.observation_space)
    print('Reward Range:', env.reward_range)
    print('Metadata:', env.metadata)
    print('Initial State:',env.reset())
    print('First Step State:',env.step(env.action_space.sample()))

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (30, 20)
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output
from io import StringIO
import time

def env_render_frames(env_frames):
    if isinstance(env_frames[0],StringIO):
        for i in range(len(env_frames)):
            if i>0:
                clear_output()
            print(env_frames[i].getvalue())
            time.sleep(1)
    else:
        fig = plt.figure()
        plt.axis('off')

        plot = plt.imshow(env_frames[0])

        def init():
            pass

        def update(i):
            plot.set_data(env_frames[i])
            return plot,

        anim = FuncAnimation(
            plt.gcf(),
            update,
            frames=len(env_frames),
            init_func=init,
            interval=20,
            repeat=True,
            repeat_delay=20)
        plt.close(anim._fig)
        display(HTML(anim.to_jshtml()))