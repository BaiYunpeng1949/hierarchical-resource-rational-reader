import os
import json
import imageio
import matplotlib.pyplot as plt
from PIL import Image

class ScanpathVisualizer:
    def __init__(self):
        pass

    def load_data(self, json_file_path):
        """Load data from the given JSON file."""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data

    def get_fixation_points(self, data, max_fixations=None):
        """Extract fixation points from the data."""
        fixation_points = []
        # Data is a list of episodes or trials
        for trial in data:
            fixation_data = trial.get('fixation_data', [])
            for fixation in fixation_data:
                norm_fix_x = fixation.get('norm_fix_x')
                norm_fix_y = fixation.get('norm_fix_y')
                fix_x = fixation.get('fix_x')
                fix_y = fixation.get('fix_y')
                word_index = fixation.get('word_index', 'NA')

                # Exclude fixations where normalized coordinates are missing
                if norm_fix_x is not None and norm_fix_y is not None:
                    fixation_points.append({
                        'norm_fix_x': norm_fix_x,
                        'norm_fix_y': norm_fix_y,
                        'fix_x': fix_x,
                        'fix_y': fix_y,
                        'word_index': word_index
                    })
                    if max_fixations is not None and len(fixation_points) >= max_fixations:
                        return fixation_points  # Return early if max fixations reached
        return fixation_points

    def plot_scanpath_image(self, image_path, pixel_points, skipped_saccades, output_path, show_annotations=False):
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
            for idx, p in enumerate(pixel_points):
                plt.text(
                    p['fix_x'], p['fix_y'],
                    f"{idx}",
                    fontsize=9, color='blue', ha='right', va='bottom', alpha=0.4
                )

        plt.title('Scanpath Visualization')
        plt.axis('off')

        # Save the image
        # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.savefig(output_path, pad_inches=0)  # Removed bbox_inches='tight' to avoid cropping
        plt.close()

        # Calculate revisit percentage during plotting
        if total_fixations > 0:
            revisit_percentage_plot = (num_revisit_fixations / total_fixations) * 100
        else:
            revisit_percentage_plot = 0

        # Return the revisit percentage for comparison
        return revisit_percentage_plot
    
    def plot_frame(self, image_path, pixel_points, skipped_saccades, frame_number, frames_dir, show_annotations=False):
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
            for idx, p in enumerate(pixel_points[:frame_number]):
                plt.text(
                    p['fix_x'], p['fix_y'],
                    f"{idx}",
                    fontsize=9, color='blue', ha='right', va='bottom', alpha=0.7
                )

        plt.title(f'Scanpath Visualization - Frame {frame_number}')
        plt.axis('off')

        # Save the frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
        # plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
        plt.savefig(frame_path, pad_inches=0)  # Removed bbox_inches='tight' to avoid cropping
        plt.close()

        # Calculate revisit percentage up to this frame
        if total_fixations > 0:
            revisit_percentage_plot = (num_revisit_fixations / total_fixations) * 100
        else:
            revisit_percentage_plot = 0

        return frame_path, revisit_percentage_plot

    def create_scanpath_video(self, image_path, pixel_points, skipped_saccades, output_dir, show_annotations=False, fps=10):
        """Create a video showing the scanpath as it develops over time."""
        frames_dir = os.path.join(output_dir, 'video_frames')

        # Create directory for frames
        os.makedirs(frames_dir, exist_ok=True)

        frame_paths = []
        for i in range(1, len(pixel_points) + 1):
            frame_path, _ = self.plot_frame(image_path, pixel_points, skipped_saccades, i, frames_dir, show_annotations)
            frame_paths.append(frame_path)

        video_path = os.path.join(output_dir, 'scanpath_video.mp4')
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        print(f"Saved scanpath video at {video_path}")
        return video_path

    def calculate_skipped_saccades(self, fixation_data):      
        """Identify skipped saccades and return the indices of skipped saccades."""
        skipped_saccades = []
        fixated_word_indices = [int(fixation['word_index']) for fixation in fixation_data if fixation['word_index'] != 'NA']

        for i in range(len(fixated_word_indices) - 1):
            current_word_idx = fixated_word_indices[i]
            next_word_idx = fixated_word_indices[i + 1]
            if next_word_idx - current_word_idx > 1:
                skipped_saccades.append((i, i + 1))  # Store the index of the skipped saccade

        return skipped_saccades

    def save_scanpath_json(self, output_path, pixel_points, word_skip_percentage_by_saccades, revisit_percentage_by_saccades):
        """
        Save the fixation points and episodic info into a scanpath.json file.
        """
        scanpath_data = {
            'fixations': [{'fix_x': p['fix_x'], 'fix_y': p['fix_y'], 'word_index': p['word_index']} for p in pixel_points],
            'metrics': {
                'word_skip_percentage_by_saccades': word_skip_percentage_by_saccades,
                'revisit_percentage_by_saccades': revisit_percentage_by_saccades
            }
        }

        with open(output_path, 'w') as json_file:
            json.dump(scanpath_data, json_file, indent=4)

        print(f"Saved scanpath JSON at {output_path}")

    def normalise(self, value, old_min, old_max, new_min, new_max):
        """Helper function to normalize values to a new range."""
        return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    def process_scanpaths_in_batches(self, json_file_path, stimulus_dir, output_dir, show_annotations=False, max_fixations=None, create_videos=False):
        """Main method to process scanpath data and generate visualizations."""
        # Load data
        data = self.load_data(json_file_path)

        # Get data from trials of experiments
        for trial_index, trial_data in enumerate(data):
            
            print('\n\n----------------------------------------------------------------------------------------------------------')
            print(f"Processing scanpath for trial {trial_index} out of {len(data)-1} (in indexes) trials...")
            stimulus_index = trial_data.get('stimulus_index')
            time_constraint = trial_data.get('time_constraint')

            # Create a folder for each trial, if the folder exists, skip the trial
            trial_output_dir = os.path.join(output_dir, f'trial{trial_index}_stimulusidx{stimulus_index}_timeconstraints{time_constraint}')
            if not os.path.exists(trial_output_dir):
                os.makedirs(trial_output_dir)
            else:
                print(f"Trial {trial_index}, folder {trial_output_dir} already processed. Skipping...")
                continue

            # Identify the stimulus image\
            image_filename = f'image_{stimulus_index}.png'
            stimulus_image_path = os.path.join(stimulus_dir, image_filename)

            # Check if the image exists
            if not os.path.exists(stimulus_image_path):
                raise FileNotFoundError(f"Image {image_filename} not found at {stimulus_dir}")

            # Extract fixation points
            fixation_points = self.get_fixation_points([trial_data], max_fixations=max_fixations)

            # Load image to get dimensions
            img = Image.open(stimulus_image_path)
            img_width, img_height = img.size
            assert img_width == 1920 and img_height == 1080, "Image dimensions should be 1920x1080."

            # # Map to pixel coordinates
            # pixel_points = self.map_to_pixel_coordinates(fixation_points, img_width, img_height)

            # Calculate skipped saccades
            skipped_saccades = self.calculate_skipped_saccades(fixation_points)

            # Plot scanpath image
            output_image_path = os.path.join(trial_output_dir, f'scanpath_visualization.png')
            revisit_percentage = self.plot_scanpath_image(stimulus_image_path, fixation_points, skipped_saccades, output_image_path, show_annotations=show_annotations)

            # Save scanpath JSON
            output_json_path = os.path.join(output_dir, 'scanpath_visualization.json')
            self.save_scanpath_json(output_json_path, fixation_points, word_skip_percentage_by_saccades=0, revisit_percentage_by_saccades=revisit_percentage)

            print("Scanpath processing complete.")

            # Create a video showing the scanpath development
            if create_videos:
                video_output_dir = os.path.join(trial_output_dir, 'video')
                os.makedirs(video_output_dir, exist_ok=True)
                video_path = self.create_scanpath_video(stimulus_image_path, fixation_points, skipped_saccades, video_output_dir, show_annotations=show_annotations)
                print(f"Scanpath video saved at {video_path}")


if __name__ == '__main__':

    visualizer = ScanpathVisualizer()

    stimulus_images_dir = '../step5/data/gen_envs/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400/simulate'
    
    # Simulation data visualizations
    # ---------------------------------------------------------------------------------------------------------------
    simulation_home_dir = '/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s'   
    simulation_output_dir = os.path.join(simulation_home_dir, 'visualizations')
    # Create the output directory if it does not exist
    if not os.path.exists(simulation_output_dir):
        os.makedirs(simulation_output_dir)

    simulation_json_file_path = os.path.join(simulation_home_dir, 'sim_processed_scanpath_merge_filter.json')
    
    visualizer.process_scanpaths_in_batches(
        json_file_path=simulation_json_file_path, stimulus_dir=stimulus_images_dir, output_dir=simulation_output_dir, create_videos=True)
    
    # # Human data visualizations
    # # ---------------------------------------------------------------------------------------------------------------
    # human_home_dir = '/home/baiy4/reading-model/data_analysis/human_data/processed_data/11_05_19_13'
    # human_output_dir = os.path.join(human_home_dir, 'visualizations')
    # # Create the output directory if it does not exist
    # if not os.path.exists(human_output_dir):
    #     os.makedirs(human_output_dir)
    # human_json_file_path = os.path.join(human_home_dir, 'processed_human_scanpath_wo_p1_to_p4.json')
    # visualizer.process_scanpaths_in_batches(
    #     json_file_path=human_json_file_path, stimulus_dir=stimulus_images_dir, output_dir=human_output_dir, create_videos=True)

