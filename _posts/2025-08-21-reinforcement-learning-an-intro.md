# Reinforcement Learning Intro 🤖

## What makes reinforcement learning different from other paradigms ?:

- There's no target label hence it's not _supervised learning_. Reinforcement Learning different from _unsupervised learning_. Unsupervised Learning is about finding a hidden structure in a dataset. Reinforcement Learning on it's core just simply maximize the reward signals.
- **Trail and Error** search, **Delayed Reward** these are important characteristics of reinforcment learning.
- **Exploration and Exploitation** is unique to Reinforcement Learning. We have to balance exploration and exploitation.
- Time realy matters; When agent takes decision that decision may not be optimal decision in _Future_ But even though it seems like optimal decision at the time.
- No i.i.d data. Agent have to learn from dynamic data. When agent moes around in environment data is changing dynamically.
- The distinction b/w problem and solution methods for that problem is very important.

### Example of Reinforcement Learning:

- Agent trying to learn to control helicopter. State dynamics are changing frequently, agent has to make decision based on what environment now it's interacting with.
- Atari games controlled by DQN; Games have rules; Without ever telling agent what to do on that evironment agent learns by **trail & error** by interacting on environment.

## Reinforcement Learning Problem:

- **reward** $\operatorname{R}_{t}$, Just a scalar value signal at timestep t.
- Job of the agent trying to maximize the reward.
- The hypothesis objective just cumulative of these reward.

  ### Reward in different environments:

  - Agent defeat the world champian at backgammon. if agent wins it get +ve reward else -ve reward.
  - Robonaid walking, if it's walks forward +ve reward, if it's falls _-ve_ reward.
  - Helicopter follows the correct tregictory +ve reward, if it's crashes high _-ve_ reward.

  ### Goal:

  - Intuitively we care about total rewards (goal). Select the best action at each time to maximize total reward.
  - Reward not instantaneous; reward may get now or later in a future.
  - Sometimes have to take less **_optimal action_** that gives low reward in order to get one big reward in a future.

    **Examples**:

    - In invesment have to lose some $ to get big profit later.
    - Helicopter have to land for fuel it's less reward may prevent from craching in future.

## Agent

![alt text](/assets/images/image-2.png)

- At each time step agent Observse (partial info or partial state) environment & get reward and takes action.
- Action that will lead to new state and optimal reward.

### History and State:

- History is the sequence of observations, actions, rewards
  $\hspace{5em} H(t) = O_{1}, A_{1}, R_{1}, ... O_{t}, A_{t}, R_{t}$
- But history is large and we want summary of the history. **State** is summary of history at _t_ time used to determine what action agent should took, so state is function of history;
  $\hspace{5em} \operatorname{S}_{t} = f(\operatorname{H}_{t})$

### States:

- Different types of states agent may interact with.

  #### 1.Environment state:

  ![alt text](/assets/images/image-1.png)

  - The environment state $S_{t}^e$ is the environment's private state.
  - Agent can't see this state only observe (get only partial infor) this state.
  - Imagine robot walking on the street only observes what it's infront of it; It didn't know what going on airport.
  - Even if environment state is visible to agent it's not relevant; Robot walking on street don't have to know what's going on airport.

  #### 2.Agent state:

  ![alt text](/assets/images/image-3.png)

  - _Agent state_ $S^a_{t}$; Set of number lives inside of agent. Just like environment state; Except agent state if visible to agent.

  #### 3. Information State:

  - Information state (a.k.a Markov state) contains all necessary information about history.
  - State $S_{t}$ is Markov only if,
    $\hspace{5em}P[S_{t+1} | S_{t}] = P[S_{t+1}| S_{1}....S_{t}]$
  - _Markov_ by definition, Future only depends on present, if present is given,
    $\hspace{5em}H\\_{1:t} \rArr S\\_{t} \rArr H\\_{t+1}$
  - Environment state is Markov state.
  - _History_ is also markov state but it's irrelevant.

### Fully Observable Environments:

- Agent Lives in Environment and directly observe the environment state.
  $\hspace{5em} O_{t} = S^a_{t} = S^e_{t}$

- So agent state = environment state = Information state
- We usually use **Markov Decision Process - MDP** as Formalization

### Partially Observable Environments:

- Agent indirectly observes environment. For e.g, agent gets visuall information only from it's front camera. Don't have enough visuall information about it's environment.
- So agent state $\not =$ Environment state; Agent don't fully observe environment.
- Agent state must be constructed by _Agent's Belief_ based on it's past experience.
- **Partially Observable Markvon Decision Process** used as Formallization.

## Major Components of RL Agent:

1. Policy
2. Reward
3. Value Function
4. Model

These are the main components of the reinforcement learning agent. This is not all but these are main pieces. These all Components may or not needed.

#### Policy:

- Policy defines learning agent's behaviour; **Policy map's state to action**.
- Any simple function or more complex LLM can be consider as policy ; Agent's Policy denoted by $\pi$.
- Deterministic Policy: $a = \pi(s).$
- Stochastic Policy: $\pi(s) = P[A = a | S = a]$.

#### Reward:

- Reward is a simple scaler value sended to agent by environment for action it's takes on the state.
- Reward defines agent's taken action is good or bad.
- It may simple stochastic function of the state of the environment.
- In RLHF we have _reward model_ that takes LLM's response and gives reward for that response.

#### Value Function:

- Expected total reward agent gets if it follows particular state. Value function considers **future rewards** on some extends.
- If agent has to choose state b/w s1, s2 then value function would say, How much total reward we will get from s1 and s2; Then follow state that gives Highest Expected total reward.
- **Value function** depends on Behaviour **(Policy)** of the agent; Cause walking and falling agent will get different future reward.
  $\hspace{5em} v_{\pi}(s) = E[R_{t} + \gamma R_{t+1} + 
  \gamma^2 R_{t+2}+ ... | S = s]$

  - $\gamma - \space \text{Discount Factor;} $ We don't care so much about future rewards but only immediate rewards. So dicount factor $\gamma$ gives less important to future rewards, and give more important to immediate rewards.

#### Model:

- Model mimics the behaviour of environment; It's not actuall or complete environment but it's agent view of how Environment will be; Agent uses model for **_planning_**.
  - **Transition Model - P:**
    - Transition Model **P** is the one predicts What are the next dynamics of Environment. Predicts what's the next state would be when given current state and action.
      $\hspace{5em} P^a_{SS'} = P[S' = s' | S = s, A = a]$
  - **Reward Model - R**
    - Reward Model **R** is the one predicts what the next immediate Reward.
      $\hspace{5em} R^a_{S} = E[R | S = s, A = a]$
- Methods for solving reinforcement learning problem that involves model and planning called **Model based Methods**. Model based Methods not requirement, In practice **Model free Methods (Trail & Error)** are used.

## Categorizing RL Agent:

![alt text](/assets/images/image-4.png)

- **Value Based Agent**:
  - Agent uses _Value Function_ to achive goal.
  - No Policy is learned; By simply looking at Value Function agent makes decisions.
- **Policy Based Agent**:
  - Agent learns policy to make decision at each state on Environment.
  - No value function is needed to tell agent what is the good action on each state.
- **Actor Critic**:
  - Policy and Value Function used.
  - Actor Critic agent's learns policy and uses value function to make decisons;

### Fundamental Difference:

- **Model Based Agent:**
  - Agent figure out's dynamics of environment and use for planing.
  - Policy / Value function
- **Model free Agent:**
  - Agent don't consturct model of environment.
  - Learns Policy/ Value function to make decisions.

## Problems with Reinforcement Learning:

### 1. Fundamental Problems in sequentail decision making:

- Reinforcement Learning:
  - The Environement initially Unknown; Like throw a robot on factory floor, ask do the job for me 🤖.
  - Agent interacts with environment and learns policy.
  - **Example:** Agent playing atari games; Initially environment is Unkown; Agent has to play-game (interact) to learn policy (How to control joystick).
- Planning:
  - Agent already have Internal model of environment; Environment is known.
  - The agent improves policy; Don't learn policy from scratch.

### 2. Prediction and Control Problem:

- Prediction:
  - Someone gave policy, If we follow that policy How much future reward we will get; Evaluating a Future.
- Control:
  - Find a optimal policy to maximize total reward; Evaluating all policies to find Optimal Policy.
  - In order to solve **Control problem** agent have to solve **Prediction problem**.
