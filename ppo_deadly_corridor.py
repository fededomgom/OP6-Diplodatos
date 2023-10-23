import vizdoom
import pandas as pd
import torch
import numpy as np

torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Obtiene el dispositivo actual
    device = torch.cuda.current_device()
    print(f'Estás utilizando CUDA en la GPU {device}')
else:
    print('CUDA no está disponible. El código se está ejecutando en la CPU.')
class PPOAgent:
    def __init__(self, state_dim, action_dim,  learning_rate=0.001, alpha=0.99, epsilon=0.1):
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate, alpha=alpha, eps=epsilon)

    def policy(self, state):
        logits = self.policy_net(state)

        logits = torch.squeeze(logits)

        action_probs = torch.softmax(logits, dim=0)
        action = torch.multinomial(action_probs, 1).item()

        return action

    def learn(self, experiences):
        loss = self.ppo_loss(experiences)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ppo_loss(self, experiences):
        log_probs = [experience['log_probs'] for experience in experiences]
        advantages = [experience['advantage'] for experience in experiences]
        
        log_probs_tensor = torch.tensor(log_probs, requires_grad=True)
        advantages_tensor = torch.tensor(advantages, requires_grad=True)    

        ## Agrego una dimension adicional sino da error 
        log_probs_tensor = log_probs_tensor.unsqueeze(1)  
        advantages_tensor = advantages_tensor.unsqueeze(1) 

        policy_loss = -torch.mean(torch.sum(log_probs_tensor * advantages_tensor, dim=1))

        return policy_loss

# Configura el entorno VizDoom
game = vizdoom.DoomGame()
game.load_config("deadly_corridor.cfg")
game.set_doom_scenario_path("deadly_corridor.wad")
game.init()

state_dim = 4  
action_dim = 7 

agent = PPOAgent(state_dim, action_dim)

sample_actions = [
    [1, 0, 0, 0, 0, 0, 0],  # MOVE_LEFT
    [0, 1, 0, 0, 0, 0, 0],  # MOVE_RIGHT
    [0, 0, 1, 0, 0, 0, 0],  # ATTACK
    [0, 0, 0, 1, 0, 0, 0],  # MOVE_FORWARD
    [0, 0, 0, 0, 1, 0, 0],  # MOVE_BACKWARD
    [0, 0, 0, 0, 0, 1, 0],  # TURN_LEFT
    [0, 0, 0, 0, 0, 0, 1],  # TURN_RIGHT
]

for episode in range(10000):

    game.new_episode()

    state = torch.from_numpy(np.zeros((4,))).float() 

    experiences = []
    while not game.is_episode_finished():
        action = agent.policy(state)

        game.make_action(sample_actions[action])

        next_state = torch.from_numpy(np.zeros((4,))).float()
        reward = game.get_last_reward()

        advantage = reward 

        log_prob = torch.log(agent.policy_net(state)[action])

        experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'advantage': advantage,
            'log_probs': log_prob
        })
        
        state = next_state

    agent.learn(experiences)
    df = pd.DataFrame(experiences)
    df.to_csv('datos.csv')
    print('Episode {}: {}'.format(episode, game.get_total_reward()))

game.close()
