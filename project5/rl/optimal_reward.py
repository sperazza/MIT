# Given values
gamma = 0.5

# Scenarios
reward_same_room = 1
reward_adjacent_room = -0.01 + gamma * 1
reward_two_rooms_away = -0.01 + gamma * (-0.01 + gamma * 1)

# Probabilities
prob_same_room = 1/4
prob_adjacent_room = 2/4  # because there are two adjacent rooms
prob_two_rooms_away = 1/4  # because there's only one optimal path two rooms away

# Expected reward calculation
expected_reward = (reward_same_room * prob_same_room +
                   reward_adjacent_room * prob_adjacent_room +
                   reward_two_rooms_away * prob_two_rooms_away)

print(expected_reward)


# Given values
gamma = 0.5

# Scenarios
reward_same_room = 1
reward_adjacent_room = -0.01 + gamma * 1
reward_two_rooms_away = -0.01 + gamma * (-0.01 + gamma * 1)

# Probabilities
prob_same_room = 1/4
prob_adjacent_room1 = 1/4  # because there are two adjacent rooms
prob_adjacent_room2 = 1/4  # because there are two adjacent rooms
prob_two_rooms_away = 1/4  # because there's only one optimal path two rooms away

# Expected reward calculation
expected_reward = (reward_same_room * prob_same_room +
                   reward_adjacent_room * prob_adjacent_room1 +
                   reward_adjacent_room * prob_adjacent_room2 +
                   reward_two_rooms_away * prob_two_rooms_away)

print(expected_reward)
