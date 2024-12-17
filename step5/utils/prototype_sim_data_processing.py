import json
import os
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import imageio

import step5.utils.auxiliaries as aux
import step5.utils.constants as constants

def load_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def get_fixation_points(word_level_steps, max_fixations=None):
    fixation_points = []
    for step in word_level_steps:
        fixation_info = step.get('fixation_info', {})
        norm_fix_x = fixation_info.get('norm_fix_x')
        norm_fix_y = fixation_info.get('norm_fix_y')
        sampled_letters = fixation_info.get('sampled_letters', 'NA')
        is_terminate_step = fixation_info.get('is_terminate_step', False)
        sentence_level_steps = step.get('sentence_level_steps', 'NA')
        word_step = step.get('word_level_or_fixation_steps', 'NA')
        word_index = step.get('target_word_index_in_stimulus', 'NA')  # Store the word index

        # Exclude fixation points where 'sampled_letters' is empty or 'NA' or where 'is_terminate_step' is True
        if sampled_letters not in ['NA', ''] and not is_terminate_step:
            if norm_fix_x is not None and norm_fix_y is not None:
                # Include coordinates along with sentence, word level steps, and word index
                fixation_points.append({
                    'norm_fix_x': norm_fix_x,
                    'norm_fix_y': norm_fix_y,
                    'sentence_step': sentence_level_steps,
                    'word_step': word_step,
                    'word_index': word_index  # Include word index
                })
                # Break if we've reached the maximum number of fixations
                if max_fixations is not None and len(fixation_points) >= max_fixations:
                    break
    return fixation_points

def map_to_pixel_coordinates(fixation_points, image_width, image_height):
    pixel_points = []
    for fixation in fixation_points:
        norm_x = fixation['norm_fix_x']
        norm_y = fixation['norm_fix_y']
        sentence_step = fixation['sentence_step']
        word_step = fixation['word_step']
        word_index = fixation['word_index']  # Store the word index

        # Convert normalized coordinates to pixel coordinates
        x_pixel = aux.normalise(norm_x, -1, 1, 0, image_width)
        y_pixel = aux.normalise(norm_y, -1, 1, 0, image_height)
        
        # Append pixel points along with sentence and word steps for annotations
        pixel_points.append({
            'fix_x': x_pixel,
            'fix_y': y_pixel,
            'sentence_step': sentence_step,
            'word_step': word_step,
            'word_index': word_index
        })
    return pixel_points

def plot_scanpath_image(image_path, pixel_points, skipped_saccades, output_path, show_annotations=False):
    """Plot and save the full scanpath as an image with skipped saccades and revisited fixations."""
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Set the figure size to match the image resolution
    dpi = 100
    plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)

    # Display the image
    plt.imshow(img)

    # Plot the saccades
    for i in range(len(pixel_points) - 1):
        x_vals = [pixel_points[i]['fix_x'], pixel_points[i + 1]['fix_x']]
        y_vals = [pixel_points[i]['fix_y'], pixel_points[i + 1]['fix_y']]
        if (i, i + 1) in skipped_saccades:
            plt.plot(x_vals, y_vals, '-', color='blue', linewidth=2, alpha=0.7)  # Skipped saccades in blue
        else:
            plt.plot(x_vals, y_vals, '-', color='red', linewidth=2, alpha=0.5)   # Regular saccades in red

    # Initialize counters
    num_revisit_fixations = 0
    total_fixations = 0
    max_word_index_reached = None

    # Plot the fixation points with appropriate colors
    for p in pixel_points:
        word_index = p['word_index']
        if word_index != 'NA':
            word_index = int(word_index)
            total_fixations += 1

            if max_word_index_reached is None:
                # First fixation
                max_word_index_reached = word_index
                color = 'red'  # Not a revisit
            else:
                if word_index < max_word_index_reached:
                    # Revisit fixation
                    color = 'green'
                    num_revisit_fixations += 1
                else:
                    # Not a revisit
                    color = 'red'
                    max_word_index_reached = word_index

            # Plot the fixation point
            plt.plot(p['fix_x'], p['fix_y'], 'o', color=color, markersize=8, alpha=0.5)

    # Add annotations if requested
    if show_annotations:
        for p in pixel_points:
            plt.text(
                p['fix_x'], p['fix_y'],
                f"({p['sentence_step']}, {p['word_step']})",
                fontsize=9, color='blue', ha='right', va='bottom', alpha=0.4
            )

    plt.title('Scanpath Visualization')
    plt.axis('off')

    # Save the image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Calculate revisit percentage during plotting
    if total_fixations > 0:
        revisit_percentage_plot = (num_revisit_fixations / total_fixations) * 100
    else:
        revisit_percentage_plot = 0

    # Return the revisit percentage for comparison
    return revisit_percentage_plot

def plot_frame(image_path, pixel_points, skipped_saccades, frame_number, frames_dir, show_annotations=False):
    """Plot a single frame with skipped saccades and revisited fixations."""
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Set the figure size
    dpi = 100
    plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)

    # Display the image
    plt.imshow(img)

    # Initialize counters
    num_revisit_fixations = 0
    total_fixations = 0
    max_word_index_reached = None

    # Plot saccades up to the current frame
    for i in range(frame_number - 1):
        x_vals = [pixel_points[i]['fix_x'], pixel_points[i + 1]['fix_x']]
        y_vals = [pixel_points[i]['fix_y'], pixel_points[i + 1]['fix_y']]
        if (i, i + 1) in skipped_saccades:
            plt.plot(x_vals, y_vals, '-', color='blue', linewidth=2, alpha=0.7)
        else:
            plt.plot(x_vals, y_vals, '-', color='red', linewidth=2, alpha=0.5)

    # Plot fixation points up to the current frame
    for i in range(frame_number):
        p = pixel_points[i]
        word_index = p['word_index']
        if word_index != 'NA':
            word_index = int(word_index)
            total_fixations += 1

            if max_word_index_reached is None:
                # First fixation
                max_word_index_reached = word_index
                color = 'red'  # Not a revisit
            else:
                if word_index < max_word_index_reached:
                    # Revisit fixation
                    color = 'green'
                    num_revisit_fixations += 1
                else:
                    # Not a revisit
                    color = 'red'
                    max_word_index_reached = word_index

            # Plot the fixation point
            plt.plot(p['fix_x'], p['fix_y'], 'o', color=color, markersize=8, alpha=0.5)

    # Add annotations if requested
    if show_annotations:
        for p in pixel_points[:frame_number]:
            plt.text(
                p['fix_x'], p['fix_y'],
                f"({p['sentence_step']}, {p['word_step']})",
                fontsize=9, color='blue', ha='right', va='bottom', alpha=0.7
            )

    plt.title(f'Scanpath Visualization - Frame {frame_number}')
    plt.axis('off')

    # Save the frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Calculate revisit percentage up to this frame
    if total_fixations > 0:
        revisit_percentage_plot = (num_revisit_fixations / total_fixations) * 100
    else:
        revisit_percentage_plot = 0

    return frame_path, revisit_percentage_plot

def create_scanpath_video(image_path, pixel_points, skipped_saccades, output_dir, show_annotations=False, fps=10):
    """Create a video showing the scanpath as it develops over time."""
    frames_dir = os.path.join(output_dir, 'video_frames')

    # TODO debug delete later
    print(f"Creating video frames at {frames_dir}")

    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    for i in range(1, len(pixel_points) + 1):
        frame_path, _ = plot_frame(image_path, pixel_points, skipped_saccades, i, frames_dir, show_annotations)
        frame_paths.append(frame_path)

    video_path = os.path.join(output_dir, 'scanpath_video.mp4')
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print(f"Saved scanpath video at {video_path}")
    return video_path

def calculate_skipped_saccades(fixation_data):      
    """Identify skipped saccades and return the indices of skipped saccades."""
    skipped_saccades = []
    fixated_word_indices = [int(fixation['word_index']) for fixation in fixation_data if fixation['word_index'] != 'NA']

    for i in range(len(fixated_word_indices) - 1):
        current_word_idx = fixated_word_indices[i]
        next_word_idx = fixated_word_indices[i + 1]
        if next_word_idx - current_word_idx > 1:
            skipped_saccades.append((i, i + 1))  # Store the index of the skipped saccade

    return skipped_saccades

def save_scanpath_json(output_path, pixel_points, word_skip_percentage_by_saccades, revisit_percentage_by_saccades):
    """
    Save the fixation points and episodic info into a scanpath.json file.
    The fixations used are the pixel coordinates, appended with the sentence and word steps in the end.
    """

    scanpath_data = {
        'fixations': [{'fix_x': p['fix_x'], 'fix_y': p['fix_y'], 'sentence_step': p['sentence_step'], 'word_step': p['word_step'], 'word_index': p['word_index']} for p in pixel_points],
        'metrics': {
            'word_skip_percentage_by_saccades': word_skip_percentage_by_saccades,
            'revisit_percentage_by_saccades': revisit_percentage_by_saccades
        }
    }

    with open(output_path, 'w') as json_file:
        json.dump(scanpath_data, json_file, indent=4)

    print(f"Saved scanpath JSON at {output_path}")

def calculate_word_skip_percentage(fixation_data):
    """
    Calculate word skip percentage based on fixation data.
    Skips are detected if the fixation jumps over words in the sequence.
    """
    fixated_word_indices = [int(fixation['word_index']) for fixation in fixation_data if fixation['word_index'] != 'NA']
    
    if not fixated_word_indices:
        return 0, 0  # No words fixated
    
    total_num_skip_saccades = 0
    for i in range(len(fixated_word_indices) - 1):
        current_word_idx = fixated_word_indices[i]
        next_word_idx = fixated_word_indices[i + 1]
        if next_word_idx - current_word_idx > 1:
            total_num_skip_saccades += 1

    total_words_to_last_read_word = max(fixated_word_indices) + 1
    total_saccades = len(fixated_word_indices) - 1

    if total_words_to_last_read_word > 0:
        word_skip_percentage_by_reading_progress = (total_num_skip_saccades / total_words_to_last_read_word) * 100
    else:
        word_skip_percentage_by_reading_progress = 0

    if total_saccades > 0:
        word_skip_percentage_by_total_num_saccades = (total_num_skip_saccades / total_saccades) * 100
    else:
        word_skip_percentage_by_total_num_saccades = 0

    return word_skip_percentage_by_total_num_saccades, word_skip_percentage_by_reading_progress

def calculate_revisit_percentage(fixation_data):
    """
    Calculate revisit percentage based on fixation data.
    A revisit is counted when the gaze moves to a word with an index
    less than the maximum word index reached so far.
    """
    fixated_word_indices = [
        int(fixation['word_index'])
        for fixation in fixation_data
        if fixation['word_index'] != 'NA'
    ]

    num_revisit_fixations = 0
    total_fixations = len(fixated_word_indices)
    max_word_index_reached = None

    for word_index in fixated_word_indices:
        if max_word_index_reached is None:
            # First fixation
            max_word_index_reached = word_index
            # Not a revisit
        else:
            if word_index < max_word_index_reached:
                # Revisit fixation
                num_revisit_fixations += 1
            else:
                # Update the max word index reached
                max_word_index_reached = word_index
                # Not a revisit

    if total_fixations > 0:
        revisit_percentage = (num_revisit_fixations / total_fixations) * 100
    else:
        revisit_percentage = 0

    return revisit_percentage



def main(json_file_path, image_dir, output_dir=None, num_fixations=None, show_annotations=False, generate_video=False):
    # Load data
    data = load_data(json_file_path)

    # Process each episode
    for episode_index, episode in enumerate(data):
        episodic_info = episode.get('episodic_info', {})
        stimulus = episodic_info.get('stimulus', {})
        stimulus_index = stimulus.get('stimulus_index')
        stimulus_width = stimulus.get('stimulus_width')
        stimulus_height = stimulus.get('stimulus_height')

        # Use the stimulus index to load the corresponding image
        image_filename = f'image_{stimulus_index}.png'
        image_path = os.path.join(image_dir, image_filename)

        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_filename} not found at {image_path}")

        # Extract fixation points
        word_level_steps = episodic_info.get('word_level_steps', [])
        fixation_points = get_fixation_points(word_level_steps, max_fixations=num_fixations)

        # Map normalized fixation points to pixel coordinates using the normalise function
        pixel_points = map_to_pixel_coordinates(fixation_points, stimulus_width, stimulus_height)

        # Calculate word skip and revisit percentages
        word_skip_percentage_by_saccades, word_skip_percentage_by_reading_progress = calculate_word_skip_percentage(fixation_points)
        revisit_percentage_outside = calculate_revisit_percentage(fixation_points)
        print(f"=========================================================================================")
        print(f"Episode {episode_index + 1} - Stimulus {stimulus_index}")
        print(f"Word Skip Percentage: {round(word_skip_percentage_by_saccades, 2)}%, Revisit Percentage: {round(revisit_percentage_outside, 2)}%")
        print(f"=========================================================================================")

        # Calculate skipped saccades for visualization
        skipped_saccades = calculate_skipped_saccades(fixation_points)

        # Create the output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the scanpath image
        scanpath_image_path = os.path.join(output_dir, 'scanpath_image.png')
        # Plot and get revisit percentage from plotting
        revisit_percentage_plot = plot_scanpath_image(
            image_path=image_path,
            pixel_points=pixel_points,
            skipped_saccades=skipped_saccades,
            output_path=scanpath_image_path,
            show_annotations=False
        )
        # Compare the two percentages
        print(f"Revisit Percentage Calculated Outside: {revisit_percentage_outside:.2f}%")
        print(f"Revisit Percentage from Plotting Function: {revisit_percentage_plot:.2f}%")

        # Create the scanpath video if generate_video is True
        if generate_video:
            create_scanpath_video(image_path, pixel_points, skipped_saccades, output_dir, show_annotations, fps=4)

        # Save the scanpath.json file
        scanpath_json_path = os.path.join(output_dir, 'scanpath.json')
        save_scanpath_json(scanpath_json_path, pixel_points, word_skip_percentage_by_saccades, revisit_percentage_outside)

if __name__ == "__main__":          
    
    # Read the configuration file
    config_path = '../config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # sim_raw_data_10_30_17_14/stimulus_0_time_constraint_90s, sim_raw_data_10_31_16_33/stimulus_2_time_constraint_90s, sim_raw_data_11_01_12_49/stimulus_0_time_constraint_90s, sim_raw_data_11_01_16_25/stimulus_0_time_constraint_90s, 
    #       sim_raw_data_11_01_18_06/stimulus_0_time_constraint_90s
    # sim_raw_data_10_30_17_14/stimulus_0_time_constraint_60s, sim_raw_data_10_31_16_33/stimulus_2_time_constraint_60s 
    # sim_raw_data_10_30_17_14/stimulus_0_time_constraint_30s, sim_raw_data_10_31_16_33/stimulus_2_time_constraint_30s, 
    
    # Update the simulation data directory as needed
    sim_data_dir_name = 'sim_raw_data_11_05_13_23_3episodes/stimulus_0_time_constraint_30s'   
    simulation_data_dir = os.path.join('../data/sim_data_in_batches/', sim_data_dir_name)
    image_dir = os.path.join('../data/gen_envs', config['resources']['img_env_dir'], constants.SIMULATE)

    main(
        json_file_path=os.path.join(simulation_data_dir, 'simulate_xep_word_level.json'),
        image_dir=image_dir,
        output_dir=os.path.join(simulation_data_dir, 'visualizations'),
        num_fixations=None,        # Specify the number of fixations to draw
        show_annotations=False,    # Control whether annotations are shown or not
        generate_video=True       # Set to True if you want to generate the video
    )
