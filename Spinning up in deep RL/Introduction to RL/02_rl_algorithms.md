# [Part 2: Kinds of RL algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

![ ](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

## 1. Model-free vs model-based RL

The difference is whether the agent has acces to, or learns a model of the environment. Model of the environment == a function which predicts state transitions and rewards.

If there's a model, the agent can plan, think ahead, see what would happen for a range of possible choices, and explicitly decide between its options. But usually there's no ground-truth model of the environment. Usually the agent wants to use a model, it has to learn it purely from experience.

Then, bias in the model can be exploited resulting in an agent that performs well with respect to the learnt model, but behaves sub-optimally in the real environment.

Model-free methods don't have the gains in sample efficiency from using a model, but they're easier to implement and tune. As of sept 2018, model-free methods have been more extensively developed and tested than model-based methods.

## 2. What to learn

Another branching point in an RL algorithm is the question of **what to learn**:

* Policies, either stochastic or deterministic
* Action-value functions (Q functions)
* Value functions
* and/or environment models

### 2.1. What to learn in model-free RL

#### 2.1.1. Policy optimization

These methods optimize &theta; in &pi;<sub>&theta;</sub>(a|s) either directly by gradient ascent on the performance objective J(&pi;<sub>&theta;</sub>), or indirectly by maximizing local approximations of J(&pi;<sub>&theta;</sub>).

This optimization is almost always performed **on policy**, each update only uses data collected while acting according to the most recent version of the policy. Policy optimization also usually involves learning an approximator for the on-policy value function. Examples:

* A2C/A3C
* PPO

#### 2.1.2. Q-Learning

These methods learn an approximator for the optimal action-value function Q\*(s, a). They use an objective function based on the Bellman equation, this optimization is almsot always performed off-policy. Each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained. Examples:

* DQN
* C51

#### 2.1.3. Trade-offs between policy optimization and Q-learning

In policy optimization, you directly optimize for the thing you want. This makes them stable and reliable. Q-learning methods only indirectly optimize for agent performance, for training Q<sub>&theta;</sub> to satisfy a self-consistency equation. Less stable. But when they work, they're more sample efficient because they can reuse data more effectively than policy optimization techniques.

#### 2.1.4. Interpolating between policy optimization and Q-Learning

Policy optimization and Q-Learning aren't incompatible, under some circumstances are also equivalent, and there are many algorithms living between the two extremes:

* DDPG
* SAC

### 2.2. What to learn in model-based RL

There are many orthogonal ways of using models. Few examples:

#### 2.2.1. Background: pure planning

The most basic approach *never* explicitly represents the policy, instead uses pure planning techniques like model-predictive control to select actions. Each time the agent observes the environment, it computes a plan optimal with respect to the model. The agent then executes the first action of the plan, and immediately discards the rest of it. It computes a new plan each time it prepares to interact with the environment, to avoid using an action from a plan with a shorter-than-desired planning horizon.

#### 2.2.2. Expert iteration

This involves using and learning an explicit representation of the policy. The agent uses a planning algorithm (like Monte Carlo Tree Search) in the model, generating candidate actions for the plan by sampling from its current policy. The planning algorithm produces an action which is better than what the policy alone would have produced, hence it is an “expert” relative to the policy.

#### 2.2.3. Data augmentation for model-free methods

Uses a model-free RL algorithm to train a policy or Q-function, but either augments real experiences with fictitious ones in updating the agent, or uses only fictitious experience for updating the agent.

#### 2.2.4. Embedding planning loops into policies

This embeds the planning procedure directly into a policy as a subroutine, so that complete plans become side information for the policy, while training the output of the policy with any standard model-free algorithm. In this framework, the policy can learn to choose how and when to use the plans. This makes model bias less of a problem, because if the model is bad for planning in some states, the policy can simply learn to ignore it.