# HW 6: Hidden Markov Model

---

## Description

The `hmm/hmm.py` script contains the `HiddenMarkovModel` class, which implements a **Hidden Markov Model (HMM)** with **Forward** and **Viterbi** algorithms. This class allows initialization using predefined states and probabilities and provides methods for computing the likelihood of an observation sequence and decoding the most probable sequence of hidden states.

## Class Methods and Attributes

### 1. **Initialization (`__init__`)**
**Description:**  
Initializes the Hidden Markov Model (HMM) with observation states, hidden states, and probability matrices.

**Attributes:**
- `self.observation_states` → List of possible observed states.
- `self.hidden_states` → List of possible hidden states.
- `self.prior_p` → Prior probabilities of the hidden states.
- `self.transition_p` → Transition probability matrix between hidden states.
- `self.emission_p` → Emission probability matrix for hidden-to-observed state mapping.
- `self.observation_states_dict` → Dictionary mapping observed states to their indices.
- `self.hidden_states_dict` → Dictionary mapping hidden states to their indices.

**Parameters:**
- `observation_states` *(np.ndarray)*: List of observed states.
- `hidden_states` *(np.ndarray)*: List of hidden states.
- `prior_p` *(np.ndarray)*: Initial probability distribution of hidden states.
- `transition_p` *(np.ndarray)*: Probability matrix for transitions between hidden states.
- `emission_p` *(np.ndarray)*: Probability matrix for emitting an observed state given a hidden state.

---

### 2. **Computing Observation Sequence Likelihood (`forward`)**
**Description:**  
Runs the **Forward Algorithm** to compute the probability of observing a given sequence.

**Algorithm Overview:**
1. **Initialize**: Compute initial probabilities based on prior and emission probabilities.
2. **Recursion**: Compute the probability of each state at each time step using previous states.
3. **Termination**: Sum over the probabilities in the last time step.

**Parameters:**
- `input_observation_states` *(np.ndarray)* → The sequence of observed states.

**Returns:**
- *(float)* → The computed probability of observing the given sequence.

**Error Handling:**
- Raises `ValueError` if the transition or emission matrices have incorrect dimensions.
- Raises `ValueError` if an observed state is not found in `self.observation_states_dict`.

---

### 3. **Decoding Most Likely State Sequence (`viterbi`)**
**Description:**  
Runs the **Viterbi Algorithm** to find the most likely sequence of hidden states for a given observation sequence.

**Algorithm Overview:**
1. **Initialize**: Compute initial probabilities for each hidden state.
2. **Recursion**: Compute the most probable path to each hidden state at each time step.
3. **Backtrace**: Trace back the best path using stored backpointers.

**Parameters:**
- `decode_observation_states` *(np.ndarray)* → The sequence of observed states to decode.

**Returns:**
- *(list[str])* → The most likely sequence of hidden states.

**Error Handling:**
- Raises `ValueError` if the transition or emission matrices have incorrect dimensions.
- Raises `ValueError` if an observed state is not found in `self.observation_states_dict`.

---

## **Test Functions**

### **1. `test_mini_weather()`**
**Description:**  
Tests the `HiddenMarkovModel` implementation using a **miniature weather dataset**.

**Process:**
1. Loads a **mini weather HMM** from `mini_weather_hmm.npz`.
2. Runs the **Forward Algorithm** on `mini_weather_sequences.npz` and validates:
   - Output type is a `float`.
3. Runs the **Viterbi Algorithm** and validates:
   - Output type is a `list`.
   - Length matches expected sequence length.
   - Predicted sequence matches expected sequence.

**Expected Outcome:**  
- The **Forward Algorithm** returns a `float` probability.
- The **Viterbi Algorithm** returns the correct sequence of hidden states.

---

### **2. `test_full_weather()`**
**Description:**  
Tests the `HiddenMarkovModel` implementation on a **full weather dataset**.

**Process:**
1. Loads a **full weather HMM** from `full_weather_hmm.npz`.
2. Runs the **Forward Algorithm** on `full_weather_sequences.npz` and validates:
   - Output type is a `float`.
3. Runs the **Viterbi Algorithm** and validates:
   - Output type is a `list`.
   - Length matches expected sequence length.
   - Predicted sequence matches expected sequence.

**Expected Outcome:**  
- The **Forward Algorithm** returns a `float` probability.
- The **Viterbi Algorithm** returns the correct sequence of hidden states.

---

### **3. Edge Cases**
The test suite ensures the robustness of the implementation by checking:
1. **Empty input sequence**:
   - The **Forward Algorithm** should return `0`.
   - The **Viterbi Algorithm** should return `0` or an empty sequence.
2. **Incorrect transition/emission matrix dimensions**:
   - The class should raise a `ValueError` if the transition matrix is not square.
   - The class should raise a `ValueError` if the emission matrix does not match state-observation dimensions.

---

## **Usage**
To run the tests, use:
```bash
pytest test_hmm.py
