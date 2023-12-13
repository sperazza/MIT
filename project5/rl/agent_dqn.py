"""Tabular QL agent"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 300
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

model = None
optimizer = None


def epsilon_greedy(state_vector, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    if np.random.rand() < epsilon:
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:
        state_vector = state_vector.unsqueeze(0)  # Adding batch dimension
        q_values_action, q_values_object = model(state_vector)
        action_index = q_values_action.max(1)[1].item()  # Finding the action with the max Q-value
        object_index = q_values_object.max(1)[1].item()  # Finding the object with the max Q-value

    return (action_index, object_index)

class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state), self.state2object(state)


# pragma: coderesponse template
def deep_q_learning(current_state_vector, action_index, object_index, reward, next_state_vector, terminal):
    current_state_vector = current_state_vector.unsqueeze(0)  # Add batch dimension
    next_state_vector = next_state_vector.unsqueeze(0)

    # Get current Q values
    q_values_action_cur, q_values_object_cur = model(current_state_vector)
    q_value_cur_action = q_values_action_cur[0][action_index].unsqueeze(0)  # Unsqueezed
    q_value_cur_object = q_values_object_cur[0][object_index].unsqueeze(0)  # Unsqueezed

    with torch.no_grad():
        # Get next Q values
        q_values_action_next, q_values_object_next = model(next_state_vector)
        maxq_next_action = q_values_action_next.max(1)[0]
        maxq_next_object = q_values_object_next.max(1)[0]
        maxq_next = 0.5 * (maxq_next_action + maxq_next_object)  # Average of max Q-values

        # Calculate target value
        target_value = reward + GAMMA * maxq_next * (not terminal)
        target_value = torch.tensor([target_value], dtype=torch.float32)  # Ensure it's a 1D tensor

    # Calculate loss
    loss_action = F.smooth_l1_loss(q_value_cur_action, target_value)
    loss_object = F.smooth_l1_loss(q_value_cur_object, target_value)
    loss = 0.5 * (loss_action + loss_object)  # Average loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




# pragma: coderesponse end


def run_episode(for_training):
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0  # Initialize cumulative reward for the episode
    step = 0  # Initialize step counter

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        current_state = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(current_state, dictionary))

        action_index, object_index = epsilon_greedy(current_state_vector, epsilon)

        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index)
        
        next_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(next_room_desc + next_quest_desc, dictionary))

        if for_training:
            deep_q_learning(current_state_vector, action_index, object_index, reward, next_state_vector, terminal)

        if not for_training:
            epi_reward += reward * (GAMMA ** step)
            step += 1  # Increment step counter

        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc

    if not for_training:
        return epi_reward



def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(device)
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
