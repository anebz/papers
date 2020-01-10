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
* computation of log likelihoods of particular actions

![ ](https://spinningup.openai.com/en/latest/_images/math/cc2095cba170e09137c55cb4f1786955b3174336.svg)



##  4. trajectories



##  5. different formulations of return



##  6. the RL optimization problem



##  7. and value function


