import random

import vizdoom

if __name__ == '__main__':
    
    # Instantiate a VizDoom game instance.
    game = vizdoom.DoomGame()
    game.load_config('basic.cfg')
    game.init()

    # Define possible actions. Each number represents the state of a button (1=active).
    sample_actions = [
        [1, 0, 0],  # Move left
        [0, 1, 0],  # Move right
        [0, 0, 1],  # Attack
    ]

    n_episodes = 10
    current_episode = 0
    
    while current_episode < n_episodes:
        game.make_action(random.choice(sample_actions))

        if game.is_episode_finished():
            current_episode += 1
            game.new_episode()

    game.close()