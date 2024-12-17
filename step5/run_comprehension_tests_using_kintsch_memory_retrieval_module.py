import os
import difflib
import random
import json
import datetime
import pprint
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from modules.llm_envs.LLMMemories import LLMWorkingMemory
from utils import constants as const

# -------------------------------
# Main Class: AcrossTrialsComprehensionTests
# -------------------------------

class AcrossTrialsComprehensionTests:
    """
    A class to traverse all trials to process the LTM and perform comprehension tests
    (e.g., MCQ, free recalls) according to Kintsch's memory retrieval mechanism 
    (More visited sentences have higher probability to be retrieved for comprehension tests).
    """

    def __init__(self, text_level_inputs_wo_kintsch_memory_retrieval: List[Dict[str, Any]], output_dir: str, output_folder_dir: str, 
                 text_similarity_threshold: float = 0.5, exploration_rate: float = 0.5):
        """
        Initialize the AcrossTrialsComprehensionTests with the provided inputs and LTM text.

        Args:
            inputs (List[Dict[str, Any]]): List of reading log entries. Covers all the trials.
            ltm_text (str): Hierarchical text representing the LTM structure.
        """
        self.inputs = text_level_inputs_wo_kintsch_memory_retrieval

        # Initialize the LTM Parser 
        self.ltm_parser = IndividualTrailLTMParser()

        # Initialize ReadingHistoryManager with the input logs
        self.reading_history_manager = IndividualTrialReadingHistoryManager()

        # Initialize MemoryManager with the parsed LTM structure
        self.memory_manager = IndividualTrialMemoryManager()

        # Initialize the Working Memory for memory retrieval
        self.working_memory = LLMWorkingMemory(config="config.yaml")

        # Load the MCQ metadata
        with open(const.MCQ_METADATA_PATH, "r") as file:
            self.mcq_metadata = json.load(file)
        
        # Initialize the output directory
        self.output_dir = output_dir

        # Initialize the text similarity threshold
        self.text_similarity_threshold = text_similarity_threshold

        # Initialize the exploration rate
        self.exploration_rate = exploration_rate

        # Create a folder for the output files
        self.output_folder_name = output_folder_dir
    
    def traverse_trials(self):
        """
        Traverse all trials to process the LTM and perform comprehension tests.
        """
        across_trials_results = []
        for trial_index, trial_data in enumerate(self.inputs, start=1):
            
            #####################################################
            ############## TODO debug, delete later #############
            #####################################################
            # if trial_index >= 9:
            #     break

            # Get this trial's metadata
            time_constraint = trial_data['episodic_info']['task']['time_constraint']
            stimulus_index = trial_data['episodic_info']['stimulus']['stimulus_index']
            num_sentence_visits = len(trial_data['episodic_info']['text_level_steps'])
            print(f"\nProcessing Trial {trial_index}......      The time constraints was: {time_constraint}     The stimulus index is: {stimulus_index}     The total number of visits to sentences is: {num_sentence_visits}")
            print("---------------------------------------------------------------------------------------------------------------------------------------------------------")

            # Get the LTM text for the current trial from the last "text_level_steps"
            ltm_text = self.get_last_ltm_text(trial_dict=trial_data)
            # print(f"\nLTM Text for Trial {ltm_text}:")
            # print("---------------------------------------------------")

            # Parse LTM texts to dictionary/list
            self.ltm_parser.reset(ltm_text=ltm_text, indent_size=2)
            ltm_dict = self.ltm_parser.parse()
            # print("\nGenerated LTM Structure:")
            # print(ltm_dict)
            # print(f"The type of ltm_dict is: {type(ltm_dict)}")

            # Count the visits to each raw sentences in the trial data
            self.reading_history_manager.reset(inputs=[trial_data])
            reading_history = self.reading_history_manager.get_reading_history()
            # pprint.pprint(reading_history)
            # self.reading_history_manager.display_reading_history()

            # Label LTM clusters with reading history and visit counts
            self.memory_manager.reset(ltm_dict['LTM'], reading_history, threshold=self.text_similarity_threshold)
            # Display cluster strengths
            # print("Cluster Strengths:")
            # cluster_strengths = self.memory_manager.get_cluster_strengths()
            # for cluster, strength in cluster_strengths.items():
            #     print(f"Cluster: {cluster}, Visit Count: {strength}")
            # Retrieve a memory path based on Kintsch's model
            # self.memory_manager.display_retrieved_memory(exploration_rate=0.2)
            retrieved_memory = self.memory_manager.get_memory_path(exploration_rate=self.exploration_rate)
            # # Visualize the LTM structure (optional)
            # self.memory_manager.visualize_ltm()
            # print(f"\nRetrieved Memory (the type of it is: {type(retrieved_memory)}):")
            # print(retrieved_memory)

            # Perform the comprehension tests based on the retrieved memory
            self.working_memory.reset()
            # trial_data = self.perform_comprehension_tests(trial_data=trial_data, stimulus_index=stimulus_index, retrieved_memory=retrieved_memory)
            trial_data = self.perform_comprehension_tests(trial_data=trial_data, stimulus_index=stimulus_index)

            # Update the visits and retrieved memory in the MemoryManager
            trial_data['episodic_info']['reading_history'] = reading_history
            trial_data['episodic_info']['retrieved_memory'] = retrieved_memory
            trial_data['episodic_info']['memory_retrieve_metadata'] = {
                "text_similarity_threshold": self.text_similarity_threshold,
                "exploration_rate": self.exploration_rate
            }

            # Update the trial data with the comprehension test results
            across_trials_results.append(trial_data)
        
        ############################################################################################################
        # Save the results to a file
        # Get the updated file name
        filename = self.generate_simulation_filename(
            base_name=const.SIM_DATA_TEXT_LEVEL_W_KINTSCH_JS_FILE_NAME,
            text_similarity_threshold=self.text_similarity_threshold,
            exploration_rate=self.exploration_rate
        )
        output_file_path = os.path.join(self.output_dir, self.output_folder_name, filename)
        with open(output_file_path, "w") as file:
            json.dump(across_trials_results, file, indent=4)
        # Print the output file path
        print(f"\nComprehension test conditions: text_similarity_threshold={self.text_similarity_threshold}, exploration_rate={self.exploration_rate}")
        print(f"Comprehension test results saved to: {output_file_path}")
        ###########################################################################################################
    
    @staticmethod
    def generate_simulation_filename(base_name, text_similarity_threshold, exploration_rate):
        """
        Generate a simulation filename based on given parameters.
        
        Parameters:
            base_name (str): The base name for the simulation file.
            text_similarity_threshold (float): The threshold value for text similarity.
            exploration_rate (float): The exploration rate.
            
        Returns:
            str: The generated filename.
        """
        # # Replace '.' with 'dot' in the parameters for filename compatibility
        # text_similarity_threshold_str = str(text_similarity_threshold).replace('.', 'dot')
        # exploration_rate_str = str(exploration_rate).replace('.', 'dot')

        # Get the base name before .json
        base_name_parts_without_json = base_name.split(".")[0]
        
        # Generate the filename
        filename = (
            f"{base_name_parts_without_json}_text_similarity_threshold_{text_similarity_threshold}"
            f"_exploration_rate_{exploration_rate}.json"
        )
        return filename
    
    # Function to extract LTM text from the last text_level_steps
    def get_last_ltm_text(self, trial_dict: Dict[str, Any]) -> str:
        """
        Extracts the LTM text from the last text_level_steps entry.

        Args:
            trial_dict (Dict[str, Any]): The trial dictionary.

        Returns:
            str: The extracted LTM text or an empty string if not found.
        """
        try:
            text_level_steps = trial_dict['episodic_info']['text_level_steps']
            if not text_level_steps:
                print("No text_level_steps found in the trial.")
                return ""
            
            last_step = text_level_steps[-1]
            ltm_text = last_step['memories'].get('LTM', "")
            
            if not ltm_text:
                print("LTM text not found in the last text_level_steps entry.")
            
            return ltm_text
        
        except KeyError as e:
            print(f"KeyError encountered: {e}")
            return ""

    def update_visit_counts(self):
        """
        Update the visit counts in MemoryManager based on the reading history.
        """
        reading_history = self.reading_history_manager.get_reading_history()
        for sentence_index, data in reading_history.items():
            sentence = data["sentence_content"]
            visits = data["visit_count"]
            cluster_id = self.memory_manager.sentence_to_clusters.get(sentence)
            if cluster_id is not None:
                self.memory_manager.visit_counts[cluster_id] += visits
            else:
                print(f"No cluster mapping found for sentence index {sentence_index}: '{sentence}'")
    
    def perform_comprehension_tests(self, trial_data: Dict, stimulus_index: int):
        """
        Perform comprehension tests for a given stimulus index.

        Args:
            trial_data (Dict): The trial data dictionary.
            stimulus_index (int): The stimulus index for which to perform the tests.
        """
        # Perform the tests for the given stimulus index
        print(f"Performing comprehension tests for stimulus index {stimulus_index}...")

        # Delete mcq and free recall sections in the original trial data dictionary
        trial_data['episodic_info'].pop("mcq_logs", None)
        trial_data['episodic_info'].pop("free_recall_answer", None)

        # Retrieve memory for free recall (general)
        retrieved_memory_free_recall = self.memory_manager.get_memory_path(
            exploration_rate=self.exploration_rate,
            include_descendants=False
        )

        # Retrieve memory for MCQ (include all descendants)
        retrieved_memory_mcq = self.memory_manager.get_memory_path(
            exploration_rate=self.exploration_rate,
            include_descendants=True
        )

        # MCQ Test
        trial_data = self.perform_mcq_test(
            trial_data=trial_data,
            stimulus_index=stimulus_index,
            retrieved_memory=retrieved_memory_mcq
        )

        # Free Recall Test
        trial_data = self.perform_free_recall_test(
            trial_data=trial_data,
            retrieved_memory=retrieved_memory_free_recall
        )

        # Update the trial data with the retrieved memories
        trial_data['episodic_info']['retrieved_memory'] = {
            'mcq': retrieved_memory_mcq,
            'free_recall': retrieved_memory_free_recall
        }

        return trial_data

    def perform_mcq_test(self, trial_data: Dict, stimulus_index: int, retrieved_memory: str):
        """
        Perform a Multiple-Choice Question (MCQ) test for the given stimulus index.

        Args:
            trial_data (Dict): The trial data dictionary.
            stimulus_index (int): The stimulus index for which to perform the test.
            retrieved_memory (str): The retrieved memory including all descendants.
        """
        # Get the MCQs for the given stimulus
        mcqs = self.mcq_metadata[str(stimulus_index)]
        # Initialize the log dict
        mcq_logs = []
        # Answer the MCQs
        for mcq_idx, mcq in mcqs.items():
            # Answer the MCQ
            question = mcq["question"]
            options = mcq["options"]
            mcq_answer = self.working_memory.retrieve_memory(
                question_type=const.QUESTION_TYPES["MCQ"],
                question=question,
                options=options,
                ltm_gists=retrieved_memory
            )
            mcq_logs.append({
                "mcq_idx": mcq_idx,
                "question": question,
                "options": options,
                "answer": mcq_answer,
                "correct_answer": mcq["correct_answer"],
            })

            print(f"The answer to the question '{question}' is: {mcq_answer}. \n")

        trial_data['episodic_info']['mcq_logs'] = mcq_logs

        return trial_data
    
    def perform_free_recall_test(self, trial_data: Dict, retrieved_memory: str):
        """
        Perform a Free Recall test using the retrieved memory without including all descendants.

        Args:
            trial_data (Dict): The trial data dictionary.
            retrieved_memory (str): The retrieved memory for free recall.
        """
        free_recall_answer = self.working_memory.retrieve_memory(
            question_type=const.QUESTION_TYPES["FRS"],
            ltm_gists=retrieved_memory
        )

        print(f"The free recall results are: \n{free_recall_answer}. \n")

        trial_data['episodic_info']['free_recall_answer'] = free_recall_answer

        return trial_data


    def run(self):
        """
        Run the entire comprehension test process.
        """

        self.traverse_trials()


# -------------------------------
# Helper Classes
# -------------------------------


class IndividualTrialMemoryManager:
    """
    A class to manage Long-Term Memory (LTM) structures, map sentences to clusters,
    track visit counts, and retrieve memories based on reinforcement.
    """

    def __init__(self):
        """
        Initialize the MemoryManager with the given LTM structure.

        Args:
            ltm_structure (Dict[str, Any]): Nested dictionary representing the LTM.
        """
        self.ltm = None
        self.sentence_to_clusters = None
        self.visit_counts = None

    def reset(self, ltm_structure: Dict[str, Any], reading_history: Dict[int, Dict[str, Any]], threshold: float = 0.25):
        """
        Reset the MemoryManager with the given LTM structure and reading history.

        Args:
            ltm_structure (Dict[str, Any]): Nested dictionary representing the LTM.
            reading_history (Dict[int, Dict[str, Any]]): The reading history with sentence contents and visit counts.
        """
        self.ltm = ltm_structure
        self.assign_ids()
        self.sentence_to_clusters = {}
        self.visit_counts = {}
        self.initialize_visit_counts()
        # Map sentences to clusters
        sentences = [data['sentence_content'] for data in reading_history.values()]
        self.map_sentences(sentences, threshold)  # Adjust threshold as needed
        # Update visit counts based on reading history
        self.update_visit_counts(reading_history)
        # Optionally aggregate visit counts hierarchically
        self.aggregate_visit_counts_hierarchical()
    
    def assign_ids(self):
        """
        Assign unique IDs to each node in the LTM structure.
        """
        self.current_id = 1  # Start IDs from 1
        self.id_to_node = {}
        stack = [(self.ltm, None)]  # Stack holds tuples of (node, parent)

        while stack:
            node, parent = stack.pop()
            node['id'] = self.current_id
            self.id_to_node[self.current_id] = node
            if parent:
                node['parent'] = parent
            self.current_id += 1

            for key, child in node.items():
                if isinstance(child, dict) and key not in ['id', 'parent']:
                    stack.append((child, node))

    def initialize_visit_counts(self):
        """
        Initialize visit counts for each node in the LTM.
        """
        for node_id in self.id_to_node:
            self.visit_counts[node_id] = 0
    
    def map_sentences(self, sentences: List[str], threshold: float = 0.5):
        """
        Map each sentence to all matching clusters in the LTM whose similarity exceeds the threshold.

        Args:
            sentences (List[str]): List of original sentences.
            threshold (float): Similarity threshold for mapping.
        """
        # nlp = spacy.load('en_core_web_sm')
        nlp = spacy.load('en_core_web_lg')
        self.sentence_to_clusters = {}

        # Prepare cluster texts and their embeddings
        cluster_texts = []
        cluster_embeddings = []
        cluster_ids = []
        for node_id, node in self.id_to_node.items():
            for key in node.keys():
                if key not in ['id', 'parent']:
                    cluster_texts.append(key)
                    cluster_ids.append(node_id)
                    doc = nlp(key)
                    cluster_embeddings.append(doc.vector)
                    break

        cluster_embeddings = np.array(cluster_embeddings)

        # Process each sentence
        for sentence in sentences:
            doc = nlp(sentence)
            sentence_embedding = doc.vector.reshape(1, -1)
            similarities = cosine_similarity(sentence_embedding, cluster_embeddings)[0]
            # Find all clusters above the threshold
            matching_indices = np.where(similarities >= threshold)[0]
            if len(matching_indices) > 0:
                cluster_list = [cluster_ids[idx] for idx in matching_indices]
                self.sentence_to_clusters[sentence] = cluster_list
            else:
                print(f"No suitable cluster found for sentence: '{sentence}'")
                # Optionally, handle unmapped sentences

    # def preprocess_text(text: str) -> str:
    #     doc = nlp(text.lower())
    #     tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    #     return ' '.join(tokens)
    
    def update_visit_counts(self, reading_history: Dict[int, Dict[str, Any]]):
        """
        Update visit counts in the MemoryManager based on the reading history.

        Args:
            reading_history (Dict[int, Dict[str, Any]]): The reading history with sentence contents and visit counts.
        """
        for data in reading_history.values():
            sentence = data['sentence_content']
            visits = data['visit_count']
            cluster_ids = self.sentence_to_clusters.get(sentence)
            if cluster_ids:
                for cluster_id in cluster_ids:
                    self.visit_counts[cluster_id] += visits
            else:
                print(f"No cluster mapping found for sentence: '{sentence}'")
    
    def visualize_ltm(self, 
                      output_dir: str = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s", 
                      filename: str = "ltm_structure.png", title: str = "LTM Structure with Visit Counts"):
        """
        Visualize the LTM structure with node sizes proportional to visit counts and save it to a file.

        Args:
            output_dir (str): Directory where the plot image will be saved.
            filename (str): Name of the output image file.
            title (str): Title of the plot.
        """
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        G = nx.DiGraph()

        def add_edges(node: Dict[str, Any]):
            node_id = node['id']
            for key, child in node.items():
                if key in ['id', 'parent']:
                    continue
                child_id = child['id']
                G.add_edge(node_id, child_id)
                add_edges(child)

        add_edges(self.ltm)

        # Assign node sizes based on visit_counts
        node_sizes = [self.visit_counts.get(node, 1) * 100 for node in G.nodes()]
        labels = {node: f"ID {node}\nVisits: {self.visit_counts.get(node, 0)}" for node in G.nodes()}

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, with_labels=True, labels=labels, node_size=node_sizes,
            node_color='skyblue', edge_color='gray',
            linewidths=1, font_size=8
        )
        plt.title(title)

        # Save the plot to the specified directory
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()  # Close the figure to free up memory

        print(f"LTM visualization saved to {output_path}")
    
    def weighted_choice(self, choices: List[int], weights: List[float]) -> Optional[int]:
        """
        Select a choice based on the provided weights.

        Args:
            choices (List[int]): List of choice IDs.
            weights (List[float]): Corresponding weights.

        Returns:
            Optional[int]: Selected choice ID.
        """
        total = sum(weights)
        if total == 0:
            return random.choice(choices) if choices else None
        r = random.uniform(0, total)
        upto = 0
        for c, w in zip(choices, weights):
            if upto + w >= r:
                return c
            upto += w
        return None  # Fallback
    
    def retrieve_memory(self, exploration_rate: float = 0.0, include_descendants: bool = False) -> List[str]:
        """
        Retrieve a memory from the LTM based on reinforcement.

        Args:
            exploration_rate (float): Probability to explore less reinforced memories.
            include_descendants (bool): Whether to include all descendants of selected nodes.

        Returns:
            List[str]: List of strings representing the retrieved memory path.
        """
        memory_path = []

        def traverse(node: Dict[str, Any], depth: int = 0):
            if not isinstance(node, dict) or not node:
                return
            children = [key for key in node.keys() if key not in ['id', 'parent']]
            if not children:
                return
            child_ids = [node[key]['id'] for key in children]
            child_weights = [self.visit_counts.get(node[key]['id'], 0.1) for key in children]

            # Apply exploration
            if exploration_rate > 0.0 and random.random() < exploration_rate:
                selected_child_id = random.choice(child_ids)
            else:
                selected_child_id = self.weighted_choice(child_ids, child_weights)

            if selected_child_id is None:
                return

            # Retrieve the selected child key
            selected_child_key = None
            selected_child = None
            for key in node:
                if key in ['id', 'parent']:
                    continue
                if node[key]['id'] == selected_child_id:
                    selected_child_key = key
                    selected_child = node[key]
                    break

            if selected_child_key:
                memory_path.append(f"{'  ' * depth}- {selected_child_key}")
                if include_descendants:
                    # Include all descendants of the selected child
                    def collect_descendants(current_node, current_depth):
                        for child_key, child_node in current_node.items():
                            if child_key in ['id', 'parent']:
                                continue
                            memory_path.append(f"{'  ' * current_depth}- {child_key}")
                            collect_descendants(child_node, current_depth + 1)
                    collect_descendants(selected_child, depth + 1)
                else:
                    traverse(selected_child, depth + 1)

        traverse(self.ltm)
        return memory_path


    # def get_memory_path(self, exploration_rate: float = 0.0) -> str:
    #     """
    #     Get the retrieved memory path as a formatted string.

    #     Args:
    #         exploration_rate (float): Probability to explore less reinforced memories.

    #     Returns:
    #         str: Formatted memory path.
    #     """
    #     memory = self.retrieve_memory(exploration_rate)
    #     return "\n".join(memory)
    

    def get_memory_path(self, exploration_rate: float = 0.0, include_descendants: bool = False) -> str:
        """
        Get the retrieved memory path as a formatted string.

        Args:
            exploration_rate (float): Probability to explore less reinforced memories.
            include_descendants (bool): Whether to include all descendants of selected nodes.

        Returns:
            str: Formatted memory path.
        """
        memory = self.retrieve_memory(exploration_rate, include_descendants)
        return "\n".join(memory)

    
    def display_retrieved_memory(self, exploration_rate: float = 0.5):
        """
        Print the retrieved memory.

        Args:
            exploration_rate (float): Probability to explore less reinforced memories.
        """
        memory = self.get_memory_path(exploration_rate)
        print("Retrieved Memory:")
        print(memory if memory else "No memory retrieved.")

    def aggregate_visit_counts_hierarchical(self):
        """
        Aggregate visit counts up the hierarchy.
        Each node's visit count includes its own plus all children's visit counts.
        """
        aggregated_counts = {node_id: 0 for node_id in self.id_to_node}

        def aggregate(node: Dict[str, Any]) -> int:
            node_id = node['id']
            count = self.visit_counts.get(node_id, 0)
            for key, child in node.items():
                if key in ['id', 'parent']:
                    continue
                count += aggregate(child)
            aggregated_counts[node_id] = count
            return aggregated_counts[node_id]

        aggregate(self.ltm)
        self.visit_counts = aggregated_counts
    
    def get_cluster_strengths(self) -> Dict[str, int]:
        """
        Get the strengths of each cluster based on visit counts.

        Returns:
            Dict[str, int]: Mapping from cluster descriptions to visit counts.
        """
        strengths = {}
        for node_id, node in self.id_to_node.items():
            # Extract the key/name of the cluster
            cluster_name = None
            for key in node.keys():
                if key not in ['id', 'parent']:
                    cluster_name = key
                    break
            if cluster_name:
                strengths[cluster_name] = self.visit_counts.get(node_id, 0)
            else:
                strengths[f"Node {node_id}"] = self.visit_counts.get(node_id, 0)
        return strengths

class IndividualTrailLTMParser:
    """
    A class to parse hierarchical text structures into a nested dictionary representing Long-Term Memory (LTM).
    """

    def __init__(self):
        """
        Initialize the LTMParser with the provided text.

        Args:
            text (str): Multiline string representing the hierarchical structure.
            indent_size (int): Number of spaces that represent one indentation level.
                               Default is 2.
        """
        self.text = None
        self.indent_size = None
        self.ltm_structure = None

    def reset(self, ltm_text: str, indent_size: int = 2):
        """
        Reset the LTM structure to the initial state.
        """
        self.text = ltm_text
        self.indent_size = indent_size
        self.ltm_structure = {"LTM": {}}

    def parse(self) -> Dict[str, Any]:
        """
        Parse the input text and build the nested LTM structure.

        Returns:
            Dict[str, Any]: Nested dictionary representing the LTM.
        """
        lines = self.text.strip().split('\n')
        stack = [(0, self.ltm_structure["LTM"])]  # Stack holds tuples of (indent_level, current_dict)

        for line_number, line in enumerate(lines, start=1):
            # Skip empty lines
            if not line.strip():
                continue

            # Determine indentation level
            indent = len(line) - len(line.lstrip(' '))
            if indent % self.indent_size != 0:
                raise ValueError(f"Inconsistent indentation detected at line {line_number}: '{line}'")

            level = indent // self.indent_size

            # Extract the node name after the dash
            stripped_line = line.strip()
            if not stripped_line.startswith('- '):
                raise ValueError(f"Line {line_number} does not start with '- ': '{line}'")

            node_name = stripped_line[2:].strip()

            # Handle the root node
            if node_name.lower() == '[root]':
                current_dict = self.ltm_structure["LTM"]
                stack = [(0, current_dict)]  # Reset stack to root
                continue

            # If the current level is deeper than the stack, it must be exactly one level deeper
            if level > stack[-1][0] + 1:
                raise ValueError(f"Unexpected indentation at line {line_number}: '{line}'")

            # Adjust the stack to the current level
            while stack and stack[-1][0] >= level:
                stack.pop()

            if not stack:
                raise ValueError(f"Invalid indentation at line {line_number}: '{line}'")

            # Current parent dictionary
            parent_dict = stack[-1][1]

            # Add the new node
            parent_dict[node_name] = {}
            # Push the new node to the stack
            stack.append((level, parent_dict[node_name]))

        return self.ltm_structure

    def get_ltm_structure(self) -> Dict[str, Any]:
        """
        Get the parsed LTM structure.

        Returns:
            Dict[str, Any]: Nested dictionary representing the LTM.
        """
        return self.ltm_structure

    def display_ltm(self):
        """
        Display the LTM structure in a readable format.
        """
        import pprint
        pprint.pprint(self.ltm_structure)
    

class IndividualTrialReadingHistoryManager:
    """
    A class to manage and generate reading history from reading logs.
    """

    def __init__(self):
        """
        Initialize the ReadingHistoryManager with the provided inputs.

        Args:
            inputs (List[Dict[str, Any]]): List of reading log entries.
        """
        self.inputs = None
        self.reading_history = None

    def reset(self, inputs: List[Dict[str, Any]]):
        """
        Reset the ReadingHistoryManager with the given inputs.

        Args:
            inputs (List[Dict[str, Any]]): List of reading log entries.
        """
        self.inputs = inputs
        self.reading_history = {}
        self.process_inputs()

    def process_inputs(self):
        """
        Process the input logs to build the reading history.
        """
        for episode_index, episode in enumerate(self.inputs):
            episodic_info = episode.get("episodic_info", {})
            text_level_steps = episodic_info.get("text_level_steps", [])

            for step_index, step in enumerate(text_level_steps):
                reading_sentence_index = step.get("reading_sentence_index")
                sentence_just_read = step.get("sentence_just_read", "").strip()

                # Skip if sentence index is invalid or sentence is empty
                if reading_sentence_index is None or sentence_just_read == "":
                    continue

                # Initialize the history entry if not present
                if reading_sentence_index not in self.reading_history:
                    self.reading_history[reading_sentence_index] = {
                        "sentence_content": sentence_just_read,
                        "visit_count": 0
                    }
                else:
                    # Check if the sentence content is consistent
                    existing_content = self.reading_history[reading_sentence_index]["sentence_content"]
                    if existing_content != sentence_just_read:
                        print(f"Warning: Inconsistent sentence content for index {reading_sentence_index} in episode {episode_index}, step {step_index}.")
                        print(f"Existing: '{existing_content}'")
                        print(f"New: '{sentence_just_read}'")
                        # Decide how to handle inconsistencies. For now, we'll keep the first occurrence.

                # Increment visit count by 1 for each occurrence
                self.reading_history[reading_sentence_index]["visit_count"] += 1

    def get_reading_history(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the complete reading history.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping from sentence index to sentence content and visit count.
        """
        return self.reading_history

    def display_reading_history(self):
        """
        Print the reading history in a readable format.
        """
        print("Reading History:")
        for index in sorted(self.reading_history.keys()):
            content = self.reading_history[index]["sentence_content"]
            visits = self.reading_history[index]["visit_count"]
            print(f"Sentence Index: {index}, Visits: {visits}")
            print(f"Content: {content}\n")

    def export_to_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Export the reading history to a dictionary.

        Returns:
            Dict[int, Dict[str, Any]]: Reading history dictionary.
        """
        return self.reading_history

    def export_to_list(self) -> List[Dict[str, Any]]:
        """
        Export the reading history to a list of dictionaries, sorted by sentence index.

        Returns:
            List[Dict[str, Any]]: List of reading history entries.
        """
        history_list = []
        for index in sorted(self.reading_history.keys()):
            entry = {
                "sentence_index": index,
                "sentence_content": self.reading_history[index]["sentence_content"],
                "number_of_visits": self.reading_history[index]["visit_count"]
            }
            history_list.append(entry)
        return history_list

    def export_reading_history_pprint(self):
        """
        Pretty-print the reading history.
        """
        print("Formatted Reading History:")
        pprint.pprint(self.reading_history)


if __name__ == "__main__":

    text_log_dir = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s"
    filename = "simulate_xep_text_level_wo_kintsch_memory_retrieval.json"
    with open(os.path.join(text_log_dir, filename), 'r') as f:
        inputs = json.load(f)

    # Grid run the parameters
    # text_similarity_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # exploration_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    text_similarity_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    exploration_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # text_similarity_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # exploration_rates = [1.0]

    # Create the output directory
    current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    output_folder_name = "Kintsch_memory_retrieval_simulations_acorss_parameters_" + current_time
    # Create the output folder if it does not exist
    if not os.path.exists(os.path.join(text_log_dir, output_folder_name)):
        os.makedirs(os.path.join(text_log_dir, output_folder_name))

    for threshold in text_similarity_thresholds:
        for rate in exploration_rates:
            comprehension_tests = AcrossTrialsComprehensionTests(text_level_inputs_wo_kintsch_memory_retrieval=inputs, output_dir=text_log_dir, output_folder_dir=output_folder_name,
                                                                 text_similarity_threshold=threshold, exploration_rate=rate)
            comprehension_tests.run()

