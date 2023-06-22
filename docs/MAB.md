# Introduction

The multi-armed bandit problem is a classic dilemma in decision-making and probability theory. It involves selecting actions from a set of options, often referred to as "arms," to maximize cumulative rewards. The problem arises in various fields, including gambling, clinical trials, online advertising, and recommendation systems.

In the multi-armed bandit problem, each arm is associated with an unknown reward distribution, and the decision-maker must strike a balance between exploration and exploitation. 

* **Exploration**: tries different arms to gather information about the potential rewards.
* **Exploitation** refers to choosing the arms with promising results or higher expected rewards based on the available information.


However, exploring all possibilities without penalty is often not feasible due to the associated costs. Additionally, randomly choosing actions at each time step may not guarantee maximizing rewards. Therefore, intelligent strategies are needed to address these challenges.

Achieving the optimal balance between exploration and exploitation is critical. Focusing solely on exploitation may lead to getting stuck with a suboptimal arm and missing out on higher rewards from other arms. On the other hand, excessive exploration can result in spending too much time gathering information and delaying the exploitation of arms with higher expected rewards.

The multi-armed bandit problem aims to devise a strategy known as a policy or algorithm that maximizes the cumulative reward over a given period. The objective is to find the optimal balance between exploration and exploitation to achieve the highest possible compensation.

There are several strategies to solve the multi-armed bandit problem. Some of the commonly used strategies are:

* **Epsilon-Greedy Algorithm**: This algorithm balances *exploration* and *exploitation* by selecting the best-known action most of the time (*exploitation*) while occasionally exploring other actions (*exploration*). It randomly chooses an action with a small probability epsilon ($\epsilon$) and chooses the action ($a_t$) in  with the highest estimated reward the rest of the time ($t$).

* **Upper Confidence Bounds (UCB)**: The UCB algorithm maintains an upper confidence bound for each arm, representing the upper limit of the expected reward for that arm. At each time step ($t$), the algorithm selects the arm with the highest upper confidence bound, allowing it to explore arms with uncertain rewards while favoring arms with potentially higher rewards.

* **Thompson Sampling**: Thompson Sampling leverages Bayesian inference to balance *exploration* and *exploitation*. It maintains a posterior distribution for each arm's reward parameter and randomly samples a reward parameter from each arm's posterior distribution at each time step. The arm with the highest sampled value is selected, allowing for exploration of uncertain arms while exploiting arms with higher expected rewards.


## Problem formulation

Let's assume:

* Set of Arms: $A = {a_1,a_2,\cdot,a_K}$.
* Each arm $a_k$ is associated with unknown probability distribution: $P_k(\cdot)$.

The goal is to maximize the cumulative reward over a given time horizon $T$ by selecting the arms strategically.

$$ max \, \sum_{t=0}^{T} R(a_t)  $$ 

At each time step $t$, the decision-maker selects an arm $a_t \in A$ based on a policy or algorithm. After selecting an arm, the decision-maker observes the true reward $r_t$, which is sampled from the reward distribution of the selected arm.

Considerations:

* $P(\cdot)$ can be any probability distribution, including discrete and continuous distributions such as: **Bernoulli, Normal, Uniform, Exponential or Custom**. 

* $\hat{R}(a_t|r_t )$ the rule to update the reward is problem-dependent, and you need to decide which is the best for your goal.


## Epsilon-greedy algorithm

![](./docs/epsilon-greedy.png)

```tex
\begin{algorithm}[H]
\SetAlgoLined
\KwIn{Set of arms $A$, exploration parameter $\epsilon$, action distribution $P(\cdot)$}

Initialization: $R(a) \leftarrow 0$ for all $a \in A$\;
Initialization: $N(a) \leftarrow 0$ for all $a \in A$\;
\For{$t = 1$ to $T$}{
    \eIf{$\text{random}(0,1) > \epsilon$}{
        Choose action $a_t = \arg\max_{s \in S} R(a)$\;
    }{
        Choose a random action $a_t \sim P(\cdot)$\;
    }
    Perform action $a_t$ on the chosen arm\;
    Observe reward $r_t$ obtained from the chosen action\;
    Update action count: $N(a_t) \leftarrow N(a_t) + 1$\;
    Update estimated reward value:
    $R(a_t) \leftarrow \hat{R}(a_t|r_t )$\;
    \Indm
}
\end{algorithm}
```


## Upper Confidence Bounds (UCB)

The UCB algorithm maintains an upper confidence bound for each arm, representing the upper limit of the expected reward for that arm. At each time step, the algorithm selects the arm with the highest upper confidence bound, allowing it to explore arms with uncertain rewards while favouring arms with potentially higher rewards. So, it is possible to explore bad actions tried in the past (*cost to pay*).

In UCB algorithm, we always select the greediest action to maximize the upper confidence bound $ \hat{U}(a_t)$ :

$$a_{t}^{UCB} = argmax_{a \in A} \hat{R}(a_t) + \hat{U}(a_t)$$


To estimate the $\hat{U}(a_t)$ a traditional method is **UCB1 Heuristic** :

$$UCB1(a_t) = \hat{R}(a_t) + c \sqrt{\frac{2*log(t)}{N(a_t)}}$$

where:
* $\hat{R}(a_t)$ is the estimated reward of arm $a_t$,
* $N(a_t)$ is the number of times arm $a_t$ has been selected,
* $t$ is the current time step, and
* $c$ is a constant that controls the trade-off between exploration and exploitation. Higher values of $c$ encourage more exploration.

![](./docs/ucb1.png)

```tex
\begin{algorithm}[H]
\SetAlgoLined
\KwIn{List of arms $A$, exploration parameter $c$}

Initialize $t \leftarrow 0$\;
Initialize $N(a) \leftarrow 0$ for all arms $a \in A$\;
Initialize $R(a) \leftarrow 0$ for all arms $a \in A$\;

\While{$t < T$}{
    Increment $t \leftarrow t + 1$\;
    Compute the upper confidence bounds: $UCB1(a) \leftarrow R(a) + c \sqrt{\frac{2\log(t)}{N(a) + \epsilon}}$ for all arms $a \in A$\;
    Select the arm with the highest upper confidence bound: $a_t \leftarrow \arg\max_{a \in A} UCB1(a)$\;
    Pull arm $a_t$ and observe the reward $r_t$\;
    Increment the count of arm $a_t$: $N(a_t) \leftarrow N(a_t) + 1$\;
    Update the estimated reward of arm $a_t$: $R(a_t) \leftarrow \hat{R}(a_t|r_t )$\;
}
\end{algorithm}
```

## Thomson Sampling

![](./docs/thomoson_sampling.png)

```tex
\begin{algorithm}[H]
\SetAlgoLined
\KwIn{Set of arms $A$}

Initialization: Set prior distribution parameters for each arm $a \in A$\;
\For{$t = 1$ to $T$}{
    \For{$a \in A$}{
        Sample a reward from the posterior distribution: $r(a) \leftarrow$ sample a reward from $P(a)$ based on the current posterior parameters\;
    }
    Choose action $a_t = \arg\max_{a \in A} r(a)$\;
    Perform action $a_t$ on the chosen arm\;
    Observe reward $r_t$ obtained from the chosen action\;
    Update the posterior distribution parameters for the selected arm based on the observed reward\;
}
\end{algorithm}
```

# Design

@TODO: Add UML design 



