"""Tabular QL agent"""
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
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)



def epsilon_greedy(state_1, state_2, q_func, epsilon):
    # Make sure that state_1 and state_2 are integers
    state_1_index = dict_room_desc[state_1] if not isinstance(state_1, int) else state_1
    state_2_index = dict_quest_desc[state_2] if not isinstance(state_2, int) else state_2
    
    if np.random.rand() < epsilon:
        # Select a random action and object
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:
        # Select the action and object with the highest Q-value for the current state
        # Note: q_func[state_1_index, state_2_index] should give a 2D array over which argmax is computed
        action_object_pair = np.unravel_index(
            np.argmax(q_func[state_1_index, state_2_index]),
            (NUM_ACTIONS, NUM_OBJECTS)
        )
        action_index, object_index = action_object_pair

    return (action_index, object_index)


# pragma: coderesponse end



def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    # Convert states to indices if they are not integers
    current_state_1_index = dict_room_desc[current_state_1] if not isinstance(current_state_1, int) else current_state_1
    current_state_2_index = dict_quest_desc[current_state_2] if not isinstance(current_state_2, int) else current_state_2
    next_state_1_index = dict_room_desc[next_state_1] if not isinstance(next_state_1, int) else next_state_1
    next_state_2_index = dict_quest_desc[next_state_2] if not isinstance(next_state_2, int) else next_state_2

    # Rest of your code for tabular_q_learning
    if not terminal:
        max_q_next_state = np.max(q_func[next_state_1_index, next_state_2_index, :, :])
    else:
        max_q_next_state = 0

    # Update the Q-value for the current state-action pair
    q_func[current_state_1_index, current_state_2_index, action_index, object_index] = \
        (1 - ALPHA) * q_func[current_state_1_index, current_state_2_index, action_index, object_index] + \
        ALPHA * (reward + GAMMA * max_q_next_state)

    # No return needed since you're updating q_func in place


# pragma: coderesponse end


# pragma: coderesponse template

def run_episode(for_training):
    global q_func  # To use the global Q-function

    # Set the appropriate epsilon value for training or testing
    epsilon = TRAINING_EP if for_training else TESTING_EP
    # Initialize cumulative discounted reward
    epi_reward = 0
    # Start a new game
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    step = 0

    while not terminal:
        # Choose action and object indices using the epsilon-greedy policy
        action_index, object_index = epsilon_greedy(current_room_desc, current_quest_desc, q_func, epsilon)
        
        # Take a step in the game
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(current_room_desc, current_quest_desc, action_index, object_index)
        
        # Update cumulative discounted reward if not training
        if not for_training:
            epi_reward += reward * (GAMMA ** step)

        # Update the Q-function if in training mode
        if for_training:
            tabular_q_learning(q_func, current_room_desc, current_quest_desc, action_index, object_index, reward, next_room_desc, next_quest_desc, terminal)

        # Prepare for the next step
        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc
        step += 1

    if not for_training:
        return epi_reward
    

# def run_episode(for_training):
#     """ Runs one episode
#     If for training, update Q function
#     If for testing, computes and return cumulative discounted reward

#     Args:
#         for_training (bool): True if for training

#     Returns:
#         None or float: cumulative discounted reward if for testing
#     """
#     # Epsilon value for the epsilon-greedy strategy
#     epsilon = TRAINING_EP if for_training else TESTING_EP
#     # Initialize cumulative reward for the episode
#     epi_reward = 0
#     # Discount factor for future rewards
#     discount_factor = GAMMA
#     # Initialize for each episode
#     (current_room_desc, current_quest_desc, terminal) = framework.newGame()
#     global q_func
#     step = 0

#     while not terminal:
#         # Choose next action and execute
#         action_index, object_index = epsilon_greedy(epsilon, q_func, current_room_desc, current_quest_desc)
#         # Take a step in the game
#         next_state = framework.step_game(current_room_desc, current_quest_desc, action_index, object_index)
#         # Unpack next state
#         (next_room_desc, next_quest_desc, reward, terminal) = next_state
#                 # Use q_func instead of q_values


#         if for_training:
#             # Update Q-function.
#             best_next_action = np.argmax(q_func[next_room_desc, next_quest_desc])
#             td_target = reward + discount_factor * q_func[next_room_desc, next_quest_desc, best_next_action]
#             td_delta = td_target - q_func[current_room_desc, current_quest_desc, action_index, object_index]
#             q_func[current_room_desc, current_quest_desc, action_index, object_index] += ALPHA * td_delta


#         if not for_training:
#             # Update reward with discounting
#             epi_reward += reward * (discount_factor ** step)

#         # Prepare for the next step
#         current_room_desc, current_quest_desc = next_room_desc, next_quest_desc

#         # Increase the step for discounting
#         step += 1

#     if not for_training:
#         # Return the cumulative discounted reward
#         return epi_reward
    
# def run_episode(for_training):
#     """ Runs one episode
#     If for training, update Q function
#     If for testing, computes and return cumulative discounted reward

#     Args:
#         for_training (bool): True if for training

#     Returns:
#         None
#     """
#     epsilon = TRAINING_EP if for_training else TESTING_EP

#     epi_reward = None
#     # initialize for each episode
#     # TODO Your code here

#     (current_room_desc, current_quest_desc, terminal) = framework.newGame()

#     while not terminal:
#         # Choose next action and execute
#         # TODO Your code here

#         if for_training:
#             # update Q-function.
#             # TODO Your code here
#             pass

#         if not for_training:
#             # update reward
#             # TODO Your code here
#             pass

#         # prepare next step
#         # TODO Your code here

#     if not for_training:
#         return epi_reward


# pragma: coderesponse end


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
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

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
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

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
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
