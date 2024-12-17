import os
import json

def generate_json_from_images(stimuli_folder, output_file, question_text="What does the text say? Please summarize."):
    """
    Generate a JSON file with data for each image in the stimuli folder.

    Parameters:
    - stimuli_folder (str): Path to the folder containing image files.
    - output_file (str): Path where the output JSON file will be saved.
    - question_text (str): The question text to include for each image.

    Returns:
    - None
    """
    data = []
    
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(stimuli_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort the files to ensure consistent order
    
    for idx, image_file in enumerate(image_files, start=0):
        image_id = image_file  # Use the image filename as image_id
        question_id = f"qid_{idx:03d}"  # Generate a unique question ID
        
        # Fixed arrays for X, Y, T_start, and T_end
        X = [150, 150, 150]
        Y = [150, 150, 150]
        T_start = [0, 500, 1000]
        T_end = [500, 1000, 1500]
        
        subject_answer = "placeholder_answer"
        accuracy = -1  # Placeholder value
        height = 1080  # Fixed height; adjust if needed
        width = 1920   # Fixed width; adjust if needed
        length = len(X)  # Number of fixations
        
        answer = f"correct_answer_{idx:03d}"  # Placeholder correct answer
        
        entry = {
            "image_id": image_id,
            "question_id": question_id,
            "question_text": question_text,
            "X": X,
            "Y": Y,
            "T_start": T_start,
            "T_end": T_end,
            "subject_answer": subject_answer,
            "accuracy": accuracy,
            "height": height,
            "width": width,
            "length": length,
            "answer": answer
        }
        data.append(entry)
    
    # Save the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON file has been saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Path to your stimuli folder containing the images
    stimuli_folder = "stimuli"  # Replace with your actual path if different
    
    # Path where you want to save the output JSON file
    output_file = "AiR_fixations_test.json"
    
    # Optional: Customize the question text if needed
    question_text = "Please summarise what does the text say?"
    
    generate_json_from_images(stimuli_folder, output_file, question_text)
