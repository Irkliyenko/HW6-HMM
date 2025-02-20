import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        if len(input_observation_states) == 0:
            return 0

        if self.transition_p.shape != (len(self.hidden_states), len(self.hidden_states)):
            raise ValueError("Transition probability matrix has incorrect dimensions.")

        if self.emission_p.shape != (len(self.hidden_states), len(self.observation_states)):
            raise ValueError("Emission probability matrix has incorrect dimensions.")

        # Step 1. Initialize variables
        fw_mat = np.zeros((len(self.hidden_states), len(input_observation_states)))
        
        obs_index = self.observation_states_dict.get(input_observation_states[0], None)   
        if obs_index is None:
            raise ValueError(f"Observation state '{input_observation_states[0]}' not found in observation_states_dict.")

        fw_mat[:, 0] = self.prior_p * self.emission_p[:, obs_index]
        
        # Step 2. Calculate probabilities
        for t in range(1, len(input_observation_states)):
            obs_index = self.observation_states_dict.get(input_observation_states[t], None)
            if obs_index is None:
                raise ValueError(f"Observation state '{input_observation_states[t]}' not found in observation_states_dict.") 
            for s in range(len(self.hidden_states)):
                fw_mat[s, t] = np.sum(fw_mat[:, t-1] * self.transition_p[:, s]) * self.emission_p[s, obs_index]

        # Step 3. Return final probability 
        return np.sum(fw_mat[:, -1])
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """ 

        if len(decode_observation_states) == 0:
            return 0

        if self.transition_p.shape != (len(self.hidden_states), len(self.hidden_states)):
            raise ValueError("Transition probability matrix has incorrect dimensions.")

        if self.emission_p.shape != (len(self.hidden_states), len(self.observation_states)):
            raise ValueError("Emission probability matrix has incorrect dimensions.")       
        
        # Step 1. Initialize variables
        
        viterbi_table = np.zeros((len(self.hidden_states), len(decode_observation_states)))
        backpointer = np.zeros((len(self.hidden_states), len(decode_observation_states)), dtype=int)

        obs_index = self.observation_states_dict.get(decode_observation_states[0], None)      
        if obs_index is None:
            raise ValueError(f"Observation state '{decode_observation_states[0]}' not found in observation_states_dict.")

        # Step 2. Calculate probabilities

        viterbi_table[:, 0] = self.prior_p * self.emission_p[:, obs_index]
        backpointer[:, 0] = 0

        for t in range(1, len(decode_observation_states)):
            obs_index = self.observation_states_dict.get(decode_observation_states[t], None)
            if obs_index is None:
                raise ValueError(f"Observation state '{decode_observation_states[t]}' not found in observation_states_dict.") 
            for s in range(len(self.hidden_states)):
                viterbi_table[s, t] = np.max(viterbi_table[:, t-1] * self.transition_p[:, s]) * self.emission_p[s, obs_index]  
                backpointer[s, t] = np.argmax(viterbi_table[:, t-1] * self.transition_p[:, s])    
            
        # Step 3. Traceback 
        best_path_pointer = np.argmax(viterbi_table[:, -1])
        best_path = [best_path_pointer]

        for t in range(len(decode_observation_states)-1, 0, -1):
            best_path.append(backpointer[best_path[-1], t])
        
        best_path.reverse()

        # Step 4. Return best hidden state sequence 
        return [str(self.hidden_states[state]) for state in best_path]