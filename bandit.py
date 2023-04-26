import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


true_reward_mean = [1,2,3,2,1]
true_reward_std  = [1,1,1,1,1]

action_space = [0,1,2,3,4]


def env(action):
    #bandit machine
    reward = np.random.randn()*true_reward_std[action]/10+true_reward_mean[action]
    return reward


def train(max_iter, eps):
    Q_history = []
    Q = np.zeros(len(action_space))
    N = np.zeros(len(action_space))
    for i in tqdm(range(max_iter)):
        if np.random.rand() < eps:
            action = np.random.choice(action_space)
        else:
            action = action_space[np.argmax(Q)]
        reward = env(action)
        N[action] += 1
        Q[action] += 1/N[action]*(reward - Q[action])
        Q_history.append(np.array(Q))
    return np.array(Q_history)


def main():
    max_iter = 1000
    eps = 0.1
    Q_history = train(max_iter, eps)
    Q = Q_history[-1]
    best_action = action_space[np.argmax(Q)]
    print(f'true best action = {action_space[np.argmax(true_reward_mean)]}')
    print(f'best action = {best_action}')

    colors = 'rgbkm'
    plt.close()
    for i in range(len(action_space)):
        plt.plot(Q_history[:,i], c=colors[i], label=f'action={i}')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    main()

