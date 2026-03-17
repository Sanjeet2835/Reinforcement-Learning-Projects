# 🧠 How the Snake AI Learns (Deep Q-Learning Architecture)

This document breaks down the Deep Q-Network (DQN) architecture used to train this Snake AI. It serves as a revision guide to understand how the Environment, Agent, and Neural Network interact to create an intelligent system.

---

## 1. The Environment & The State Space

The AI does not "see" the game screen using pixels. Instead, at every frame, the environment calculates a simplified **State**—a 7-number array of 1s and 0s that summarizes exactly what the snake needs to know at that exact millisecond.

The 7 variables are:
* **Danger Proximity (3 values):** Is there an obstacle immediately straight, right, or left of the snake's head?
* **Food Location (4 values):** Is the food currently straight ahead, behind, to the left, or to the right of the snake?

> **Note:** We use *relative* directions instead of absolute coordinates (North/South/East/West). This makes the AI "rotation invariant," meaning it doesn't have to learn a separate set of rules for finding food when it is facing Up versus facing Down.

---

## 2. The Model (The Brain)

The **Model** is purely the mathematical engine. It is a PyTorch Neural Network (`Linear_QNet`) with one hidden layer.

* **Inputs:** The 7-number State array.
* **Outputs:** 3 numbers representing the predicted Q-values for the actions `[Go Straight, Turn Right, Turn Left]`.
* **Its Job:** When given a State, it predicts which action will yield the highest long-term reward. It holds no memory and makes no decisions on its own.

---

## 3. The Agent & Experience Replay (Memory)

The **Agent** is the "player." It manages the data, holds the memory, rolls the randomness dice, and decides when to ask the Model for predictions.

The Agent uses **Experience Replay** to learn. Every time the snake takes a single step, the Agent creates a "flashcard" of that exact moment and stores it in a memory buffer (a `deque` list). 

Each memory flashcard contains 5 pieces of data:
1. `State` (What it saw)
2. `Action` (What it did)
3. `Reward` (What it got: +10, -10, or 0)
4. `Next_State` (What the world looked like after the move)
5. `Done` (Did it die?)

---

## 4. The Training Pipeline

The training loop happens in two distinct phases:

### Phase A: Short-Term Learning (Frame-by-Frame)
1. **Get State:** The Agent looks at the game.
2. **Choose Action:** The Agent picks a move (either randomly, or by asking the Model).
3. **Execute Action:** The snake moves, generating a `Reward` and a `Next_State`.
4. **Save to Memory:** The flashcard is added to the backpack.
5. **Short-Term Train:** The Model's weights are tweaked based *only* on that single, immediate flashcard.

### Phase B: Long-Term Learning (End of Episode)
When the snake dies (`Done = True`), the game pauses. 
The Agent reaches into its memory buffer, grabs a random batch of 1,000 flashcards, and feeds them all into the Neural Network at once. Training on a randomized batch prevents "catastrophic forgetting" and ensures the Model learns generalized rules rather than just memorizing its most recent sequence of moves.

---

## 5. Exploration vs. Exploitation (`Epsilon`)

The Agent uses an **Epsilon-Greedy Strategy** to learn the game.
* **`Epsilon`** is the randomness dial. 
* **Exploration (Early Games):** Epsilon is high. The Agent ignores the Model and moves randomly. This forces the snake to explore the map, die in different ways, and gather diverse data for its memory buffer.
* **Exploitation (Later Games):** As the game count increases, Epsilon decays to 0. The Agent stops guessing and relies entirely on the trained Neural Network to predict the best moves.

---

## 6. The Math: Tabular Q-Learning vs. Deep Q-Learning

In traditional Tabular Q-Learning, the Q-Table is updated manually using the full Bellman Equation:

$$Q(s, a) = Q(s, a) + \alpha [R + \gamma \max Q(s', a') - Q(s, a)]$$

In Deep Q-Learning, this equation is translated directly into neural network gradient descent. 

We separate the equation into two parts: the **Target** ($y$) and the **Prediction** ($\hat{y}$):
* $y = R + \gamma \max Q(s', a')$
* $\hat{y} = Q(s, a)$

The difference between these two is the **Temporal Error**: $(y - \hat{y})$.

Instead of manually adding this error to a table, we pass $y$ and $\hat{y}$ to PyTorch's Mean Squared Error Loss function: $Loss = (y - \hat{y})^2$. 

When the PyTorch Optimizer performs backpropagation to update the neural network's weights ($W$), it uses the derivative of the loss:
$$W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W}$$

Because the derivative of the loss contains our Temporal Error $(y - \hat{y})$, the neural network's weights are automatically pushed in the correct direction to minimize the error, effectively performing the Bellman update automatically!
