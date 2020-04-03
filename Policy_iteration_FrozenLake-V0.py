from gym import envs
import gym
import numpy as np
import time
from IPython.display import clear_output 


# show all environments
total_env = envs.registry.all()
env_ids = [i.id for i in total_env]
print(len(env_ids))


#show the FrozenLake-v0 environment
try:
    env = gym.make('FrozenLake-v0')
except:
    print("unable to load env!")
env.render()
env.close()


# several key points of the environment
action_space = env.action_space
observation_space = env.observation_space
print(action_space, type(action_space))
print(observation_space, type(observation_space))
print(action_space.sample()) # randomly sample from 0,1,2,3
print(observation_space.contains(14)) # judge wheather the state is in the state set
print(action_space.n) # number of actions
print(env.nA, env.nS) # number of actions in this env, number of states in this env.1


'''
check the 14th state's dynamic
each state has four actions: 0=>up; 1=>down; 2=>left; 3=>right

Notice that dynamic is different from policy.
Policy determins the probability of choosing a action at a specific state.
Dynamic relates to the environment, for example, imagine there's a boat driving on 
the lake, at a specific state it turned left, however, the place it finally reached
is determined by the whole environment such as wind, stream, etc.

When arrived at state[15], get 1 reward, then the episode is finished.
When agent stuck into a "Hole" the episode finishs without any reward!
'''
print("state 15")
for key, value in env.P[14].items():
    print(key,':', value)
print('\n'*2)
print('state 5')
for key, value in env.P[4].items():
    print(key, ':', value)


# define policy execute one episode similar to MC
def execute_policy(env, policy, render=False):
    total_steps,total_reward, steps = 0, 0, 0
    observation = env.reset()
    while True:
        if render:
            env.render()
            clear_output(wait=True)
            time.sleep(0.3)
        action = np.random.choice(action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        steps += 1
        total_steps += 1
        total_reward += reward
        if done:
            if render:
                print('total steps: %d' % steps)
                print("return of this episode: ", total_reward)
                time.sleep(3)
                clear_output()
            break
    return total_steps, total_reward


def evaluate_policy(env, policy, gamma=1.0, threshold=1e-2):
    value_table = np.zeros(observation_space.n)
    while True:
        delta = 0.0
        # for each state
        for state in range(env.nS):
            action = np.random.choice(action_space.n, p=policy[state])
            # the dynamic of each (state, action) tuple
            q = sum([trans_prob*(reward + gamma*value_table[next_state]*(1.0 - done))                       for trans_prob, next_state, reward, done in env.P[state][action]])
            vs = sum(policy[state]*q)            
            delta = max(delta, abs(vs-value_table[state]))
            value_table[state] = vs
        if  delta < threshold:
            break
    return value_table


# policy improvement
def improve_policy(env, value_table, policy, gamma=1.0):    
    optimal = True
    execute_policy(env, policy, render=True)
    # calculate q
    for state in range(observation_space.n):
        Q_table = np.zeros(env.nA)
        for action in range(action_space.n):
            Q_table[action] = sum([trans_prob*(reward + gamma*value_table[next_state]*(1.0-done))                            for trans_prob, next_state, reward, done in env.P[state][action]]) 
        a = np.argmax(Q_table)
        if policy[state][a] != 1.:
            optimal = False
            policy[state] = 0.
            policy[state][a] = 1.
    return optimal


# policy iteration
def iterate_policy(env, gamma=1.0, threshold=1e-2):
    policy = np.ones((env.nS, env.nA)) / env.nA
    for i in range(1000):
        print('{:=^80}'.format('iteration %i'%(i+1)))
        time.sleep(3)
        value_table = evaluate_policy(env, policy, gamma, threshold)
        if improve_policy(env, value_table, policy, gamma):
            print("iterated %d times" %(i+1))
            break
    return policy, value_table


# check the so called optimal policy's performance
def check_policy(policy, episodes=100):
    successed_nums = 0
    total_steps = 0
    for i in range(episodes):
        one_episode_steps, one_episode_return = execute_policy(env, policy)
        total_steps += one_episode_steps
        if one_episode_return == 1.0:
            successed_nums += 1
    return total_steps / episodes, successed_nums / episodes


optimal_policy, optimal_value_tabel = iterate_policy(env)
print("policy: ", optimal_policy, sep='\n')
print("value_tabel: ", optimal_value_tabel, sep='\n')


ave_steps, acc = check_policy(optimal_policy, episodes=5000)
print("ave_steps: ", ave_steps)
print("acc: ", acc)