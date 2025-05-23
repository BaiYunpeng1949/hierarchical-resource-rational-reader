import json


class SimulatorV0523:
    
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            self.text_data = json.load(f)
        
        # TODO debug delete later
        print(self.text_data)
        
    def get_text_data(self):
        pass
    
    def get_llm_comprehension_test(self, question_type: str="proportional recall"):
        pass
    

if __name__ == "__main__":
    simulator = SimulatorV0523(json_file_path="assets/example_texts.json")
    simulator.get_llm_comprehension_test("proportional recall")