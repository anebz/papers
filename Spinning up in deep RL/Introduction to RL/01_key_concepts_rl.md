# [Part 1: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

Definition: RL is the study of agents and how they learn by trial and error. 

![ ](https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png)

The **environment** is the world that the agent lives in and interacts with. At every step of the interaction, the **agent** sees a (possibly partial) observation of the state of the world, and then decides on an action to take. The environment changes when the agent acts on it, and can also change on its own.

The agent receives a **reward** signal from the environment, a number saying how good/bad the current world state is. The goal of the agent is to maximize its cumulative reward, called **return**. RL methods are ways that the agent can learn behaviors to achieve its goal.

##  1. states and observations

A state `s` is a complete description of the state of the world. No information about the world is hidden from the state. An observation `o` is a partial description of a state, which might omit information.

In deep RL, states and observations are represented by a real-valued vector/matrix/higher order tensor. Could be RGB matrix, or angles and velocities of a robot.

When the agent is able to observe the complete state of the environment, then the environment is fully observed. If it can only see a partial observation, it's partially observed.

In notation, sometimes `s` is used when it really should be `o`. This happens when talking about how the agent decides an action: we often signal in notation that the action is conditioned on the state, when in practice, the action is conditioned on the observation because the agent does not have access to the state.

##  2. Action spaces

The **action space** is the set of all valid actions in a given environment. Some environments like Atari and Go have **discrete** action spaces, where only a finite number of moves are available to the agent. If it's a robot in the physical world, the action space is **continuous**.

##  3. Policies

A **policy** is a rule used by an agent to decide what actions to take. It can be deterministic, in which case it's denoted by &mu;:

![ ](https://spinningup.openai.com/en/latest/_images/math/73fcacd255a221d20d5d9300acf86e4d3bf5ea1b.svg)

or it might be stochastic, denoted by &pi;:

![ ](https://spinningup.openai.com/en/latest/_images/math/89757355805c4084ac93610e9581c060f2e61610.svg)

Because the policy is the agent's brain, sometimes agent is used instead of policy. In deep RL, policies are parameterized: computable functions which depend on weights and biases of a NN, which we can adjust to change the behavior via some optimization algorithm. We denote the parameters of such a policy by &theta; and write this as a subscript to highlight the connection:

![ ](https://spinningup.openai.com/en/latest/_images/math/831f731859658682b2af7e217a76648697c9de46.svg)

### 3.1. Deterministic policies

Each action always gets the same output.

### 3.2. Stochastic policies

Two most common kind of stochastic policies: 

* categorical policies
    - can be used in discrete action spaces
* diagonal Gaussian policies
    - continuous action spaces

Two key computations are important for stochastic policies:

* sampling actions from the policy
* computation of log likelihoods of particular actions: ![ ](https://spinningup.openai.com/en/latest/_images/math/cc2095cba170e09137c55cb4f1786955b3174336.svg)

#### 3.2.1. Categorical policies

IT's a classifier over discrete actions, the NN design is the same as for a classification task. The input is the observation, a number of layers (convolutional, densely-connected), and then a final linear layer giving you logits for each action, and a softmax to convert logits into probabilities.

Given the probabilities for each action, Tensorflow has built-in tools for sampling.

#### 3.2.2. Diagonal Gaussian policies

A multivariate Gaussian/normal distribution is described by a mean vector &mu; and a covariance matrix. A diagonal Gaussian distribution is a special case where the covariance matrix has only entries on the diagonal. So we can represent it as a vector.

A diagonal Gaussian policy always has a NN which maps from observation to mean actions &mu;<sub>&theta;</sub>(s). The covariance matrix can be represented in 2 ways typically:

1. There's a single vector of log standard deviations, log&sigma;, which is *not* a function of state. the log&sigma; are standard parameters.
2. There's a NN that maps from states to log standard deviations log&sigma;<sub>&theta;</sub>(s). It might share some weights with the mean network.

In both cases the output is log standard deviations because log stds are free to take on any values (-inf, inf) while stds must be nonnegative.

> Sampling

Given the mean action and standard deviation and a vector z of noise from a spherical Gaussian, an action sample can be computed with:

![ ](https://spinningup.openai.com/en/latest/_images/math/b18a4163a861b1fc18c6a6824af3f5540d4e2468.svg)

with the elementwise product of two vectors.

##  4. Trajectories

A trajectory &tau; is a sequence of states and actions in the world.

![ ](https://spinningup.openai.com/en/latest/_images/math/8337d86159a1cd98dfcd0601993d7b6b2fbb54d9.svg)

The first state of the world *s<sub>0</sub>* is randomly sampled from the **start-state distribution** sometimes denoted by &rho;<sub>0</sub>.

State transitions are what happens to the world between the state at time *t*, s<sub>t</sub> and the state at *t+1*, s<sub>t+1</sub>. These transitions are governed by the natural laws of the environment and depend only on the most recent action, a<sub>t</sub>. They can be either deterministic:

![ ](https://spinningup.openai.com/en/latest/_images/math/16da6346104894fb6a673473cbfc9ffeba6471fa.svg)

or stochastic:

![ ](https://spinningup.openai.com/en/latest/_images/math/872390af4f5b2541d17e7ef2bfaecbe1e9746d94.svg)

##  5. Reward and return

The reward function *R* depends on the current state of the world, the action just taken, and the next state of the world.

![ ](https://spinningup.openai.com/en/latest/_images/math/6ed565b0911f12c8ef64d93a617d8bb30380d5d5.svg)

but frequently it's simplified to just a dependence on the current state-action pair ![ ](https://spinningup.openai.com/en/latest/_images/math/3a66e4711a16a69ca64bd10d96985363d6e4bc5c.svg).

The goal of the agent is to maximize a cumulative reward over a trajectory: R(&tau;). Different kinds of returns:

1. Finite-horizon undiscounted return, the sum of rewards obtained in a fixed window of steps ![alt text](https://spinningup.openai.com/en/latest/_images/math/b2466507811fc9b9cbe2a0a51fd36034e16f2780.svg)
2. Infinite-horizon discounted return, the sum of all rewards *ever* obtained by the agent, but discounted by how far off in the future they're obtained. The discount factor is between 0 and 1. ![alt text](https://spinningup.openai.com/en/latest/_images/math/bf49428c66c91a45d7b66a432450ee49a3622348.svg)

Why discount factor? An infinite-horizon sum of rewards might not converge to a finite value, but with a discount factor and under some conditions, it does.

##  6. The RL optimization problem

The goal in RL is to select a policy which maximizes **expected return** when the agent acts according to it.

To talk about expected return, we first have to talk about probability distributions over trajectories.

Let’s suppose that both the environment transitions and the policy are stochastic. In this case, the probability of a T -step trajectory is:

![ ](https://spinningup.openai.com/en/latest/_images/math/69369e7fae3098a2f05a79680fbecbf48a4e7f66.svg)

The expected return (for whichever measure), denoted by J(\pi), is then:

![ ](https://spinningup.openai.com/en/latest/_images/math/f0d6e3879540e318df14d2c8b68af828b1b350da.svg)

The central optimization problem in RL can then be expressed by:

![ ](https://spinningup.openai.com/en/latest/_images/math/2de61070bf8758d03104b4f15df45c8ff5a86f5a.svg)

with &pi;\* being the **optimal policy**.

##  7. Value functions

The value of a state or a state-action pair is the expected return if you start in that state or state-action pair, and then act according to a particular policy forever after. There are 4 main value functions:

1. On-Policy value function, which gives the expected return if you start in state `s` and always act according to policy &pi;
    * ![ ](https://spinningup.openai.com/en/latest/_images/math/e043709b46c9aa6811953dabd82461db6308fe19.svg)
2. On-Policy Action-value function, which gives the expected return if you start in state `s`, take an arbitrary action `a` and then forever after act according to &pi;
    * ![ ](https://spinningup.openai.com/en/latest/_images/math/85d41c8c383a96e1ed34fc66f14abd61b132dd28.svg)
3. Optimal value function, which gives the expected return if you start in state `s` and always act according to the *optimal* policy in the environment
    * ![ ](https://spinningup.openai.com/en/latest/_images/math/01d48ea453ecb7b560ea7d42144ae24422fbd0eb.svg)
4. Optimal-action-value function, which gives the expected return if you start in state `s`, take an arbitrary action `a` and then forever after act according to the *optimal* policy
    * ![ ](https://spinningup.openai.com/en/latest/_images/math/bc92e8ce1cf0aaa212e144d5ed74e3b115453cb6.svg)

### 7.1. [Bellman equations](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations)

The value of your starting point is the reward you expect to get from being there + the value of wherever you land next.

### 7.2. [Advantage functions](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions)

The advantage function is the relative advantage of the action relating to the other functions in average. IT describes how much better is to take a specific action `a` in state `s` over randomly selecting an action according to the policy.

![ ](https://spinningup.openai.com/en/latest/_images/math/3748974cc061fb4065fa46dd6271395d59f22040.svg)

## 8. [Markov Decision Processes](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#optional-formalism) (MDPs)

An MDP is a 5-tuple, (S, A, R, P, &rho;<sub>0</sub>) where

* S is the set of all valid states
* A is the set of all valid actions
* R is the reward function, with r<sub>t</sub> = R(s<sub>t</sub> , a<sub>t</sub> , s<sub>t+1</sub>)
* P is the transition probability function, with P(s'|s, a) being the probability of transitioning into state `s'` if you start in state `s` and take action `a`
* &rho;<sub>0</sub> is the starting state distribution

If a system is a MDP, it obeys the Markov property: transitions only depend on the most recent state and action, and no prior history.
