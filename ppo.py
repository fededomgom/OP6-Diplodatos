import random
import vizdoom
import numpy as np
import torch
import skimage.transform

class PPOAgent:
    def __init__(self):
        # Initialize the neural network.
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        )

        # Initialize the optimizer.
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())

    def policy(self, state):
        # Get the logits from the neural network.
        logits = self.policy_net(state)

        # Sample an action from the policy distribution.
        action_probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(action_probs, 1).item()

        return action

    def learn(self, experiences):
        # Calculate the loss.
        loss = self.ppo_loss(experiences)

        # Update the neural network parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ppo_loss(self, experiences):
        # Calculate the policy loss.
        policy_loss = -torch.mean(torch.sum(experiences['log_probs'] * experiences['advantages'], dim=1))

        # Calculate the value loss.
        value_loss = torch.mean((experiences['values'] - experiences['targets'])**2)

        # Combine the losses.
        loss = policy_loss + value_loss

        return loss

if __name__ == '__main__':
    

    def preprocess(img):
        img = skimage.transform.resize(img, (30, 45))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    agent = PPOAgent()

    game = vizdoom.DoomGame()
    game.load_config('basic.cfg')
    game.init()

    sample_actions = [
        [1, 0, 0],  # Move left
        [0, 1, 0],  # Move right
        [0, 0, 1],  # Attack
    ]

    for episode in range(1000):
        # Start a new episode.
        game.new_episode()

        # Get the initial state.
        state = preprocess(game.get_state().screen_buffer)
   
        state_tensor = torch.from_numpy(state)

        # Collect experiences.
        experiences = []
        while not game.is_episode_finished():

            # Get the action from the agent.
            action = agent.policy(state_tensor)

            # Take the action.
            game.make_action(sample_actions[action])

            # Get the next state and reward.
            next_state = game.get_state()
            reward = game.get_reward()

            # Calculate the advantage.
            advantage = reward + agent.policy_net.value(next_state) - agent.policy_net.value(state)

            # Calculate the log probability of the action.
            log_prob = torch.log(agent.policy_net(state_tensor)[action])

            # Store the experience.
            experiences.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'advantage': advantage,
                'log_probs': log_prob
            })

            # Update the state.
            state = next_state

        # Learn from the collected experiences.
        agent.learn(experiences)

        # Print the episode reward.
        print('Episode {}: {}'.format(episode, game.get_total_reward()))

    # Close the VizDoom game instance.
    game.close()







        def ppo_loss(self, experiences):
        # Calculate the policy loss.
        log_probs = [experience['log_probs'] for experience in experiences]
        advantages = [experience['advantage'] for experience in experiences]
        
        log_probs_tensor = torch.tensor(log_probs, requires_grad=True)
        advantages_tensor = torch.tensor(advantages, requires_grad=True)    

        log_probs_tensor = log_probs_tensor.unsqueeze(1)  # Agrega una dimensión adicional
        advantages_tensor = advantages_tensor.unsqueeze(1)  # Agrega una dimensión adicional

        policy_loss = -torch.mean(torch.sum(log_probs_tensor * advantages_tensor, dim=1))

        return policy_loss