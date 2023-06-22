
# Case Study 01: Slot Machines

In the case of slot machines as a specific example, the objective is to maximize the cumulative reward by selecting the best slot machine to play. 


## Bernouli Slot Machines

Each slot machine, also known as an arm, in our case study is modeled as a **Bernoulli distribution**. A Bernoulli distribution is a *discrete probability* distribution with two possible outcomes, usually labeled as **success (reward) and failure (no reward)**. The probability of success, denoted as ($p$), represents the underlying probability of receiving a reward from pulling the arm of the slot machine.

To simulate the Bernoulli slot machines in our implementation, we have created a class called ```BernoulliArm```. This class provides two main functions: ```pull()``` and ```update_cumulative()```, which are responsible for pulling the arm and updating the cumulative reward, respectively.

### ```pull()````

$$\begin{equation*}
  R(a_t) =
  \begin{cases}
    1 & \text{w.p. $\theta_{s}$} \\
    0 & \text{w.p. $1-\theta_{s}$}
  \end{cases} \\ \forall  s \in S
\end{equation*}$$

In this formulation, if the chosen action $a_t$ corresponds to pulling machine $s$ at time $t$, the reward $R(a_t)$ will be 1 with a probability of $\theta_s$, representing a successful outcome, and 0 with a probability of $1-\theta_s$, representing an unsuccessful outcome.

### ```update_cumulative()```

The `update_cumulative()` function updates the estimate of the cumulative reward for a specific arm after observing a reward.

The corresponding mathematical formula implemented:

$$
\hat{R}(a_t|r_t) = \frac{{R(a_t) \cdot (N(a_t) - 1) + r_t}}{{N(a_t)}}
$$

### Simulation

The success probability of each arm is unknown to us. In our case, we have 10 arms, each representing a different slot machine, and we want to determine the arm with the highest success probability. This way, we will benchmark our 3 solvers simulating 10000 time steps.

The success probabilities of the 10 arms are as follows:

| Arm | Probability ($p$) |
| --- | ----------- |
| 1   | 0.0         |
| 2   | 0.1         |
| 3   | 0.2         |
| 4   | 0.3         |
| 5   | 0.4         |
| 6   | 0.5         |
| 7   | 0.6         |
| 8   | 0.7         |
| 9   | 0.8         |
| 10  | 0.9         |


#### Results obtained for a specific simulation

| Solver                       | Arm p=0.0 | Arm p=0.1 | Arm p=0.2 | Arm p=0.3 | Arm p=0.4 | Arm p=0.5 | Arm p=0.6 | Arm p=0.7 | Arm p=0.8 | Arm p=0.9 |
| ---------------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| EpsilonGreedy(epsilon=0.01)  | 1.10%                 | 0.14%                 | 0.07%                 | 0.12%                 | 0.15%                 | 3.27%                 | 0.14%                 | 0.09%                 | 1.35%                 | 93.57%                |
| UCB1(c=1.0)                  | 0.21%                 | 0.27%                 | 0.38%                 | 0.48%                 | 0.40%                 | 0.69%                 | 1.35%                 | 2.22%                 | 10.61%                | 83.39%                |
| Thomson Sampling(c=0.0)      | 0.03%                 | 0.04%                 | 0.03%                 | 0.04%                 | 0.13%                 | 0.04%                 | 0.17%                 | 0.32%                 | 0.59%                 | 98.61%                |


![](./docs/bernoulli_slot_machine_case_study/fig1.png)
![](./docs/bernoulli_slot_machine_case_study/fig2.png)
![](./docs/bernoulli_slot_machine_case_study/fig3.png)
![](./docs/bernoulli_slot_machine_case_study/fig4.png)
