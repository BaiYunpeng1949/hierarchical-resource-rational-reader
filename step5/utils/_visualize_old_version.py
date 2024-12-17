import ast
import json
import os
import shutil

import pandas as pd
import yaml
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from step5.generators import image_generator as gens
from step5.utils import constants as cons


def get_prefix(file_name: str):
    """
        Get the prefix of a file path.
        Spec: 0 - mode, 1 - ep_len_prefix, 2 - weight_prefix
    """
    mode = os.path.basename(file_name).split('.')[0].split('_')[0]
    ep_len_prefix = os.path.basename(file_name).split('.')[0].split('_')[1]

    if mode == cons.TEST or mode == cons.TRAIN:
        weight_prefix = os.path.basename(file_name).split('.')[0].split('_')[2]
        return mode, ep_len_prefix, weight_prefix
    else:
        return mode, ep_len_prefix


def convert_json_to_dataframe(json_file_path):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    fixed_info = json_data['fixed_info']
    dynamic_steps = json_data['dynamic_steps']

    columns = [
        'episode', 'word step', 'saccade step', 'image idx', 'words', 'word num', 'target word',
        'encoded target word', 'target word len', 'target word idx', 'norm fix x',
        'norm fix y', 'fix x', 'fix y', 'foveal seen letters', 'parafoveal seen letters',
        'internal word', 'word counters', 'recognize flag', 'completeness',
        # 'distances', 'flag on target', 'flag updated',
        'reward', 'done action', 'done'
    ]

    data_rows = []
    episode_index = fixed_info.get('episode_index', 'NA')
    image_idx = fixed_info['section_info']['section_metadata'].get('image index', 'NA')
    words_metadata = fixed_info['section_info']['section_metadata'].get('words metadata', [])
    words_list = [word['word'] for word in words_metadata]
    words_str = str(words_list)  # Convert list to string

    for step in dynamic_steps:
        word_step = step.get('steps', 'NA')
        oc_logs_batch = step.get('fixations_info', {}).get('oc_logs_batch', [])

        for saccade_step, log in enumerate(oc_logs_batch, start=1):
            row = [
                episode_index,
                word_step,
                saccade_step,
                image_idx,
                words_str,
                len(words_list),
                step.get('target_word_info', {}).get('target_word_in_section', 'NA'),
                step.get('fixations_info', {}).get('encoded internal word', 'NA'),
                step.get('target_word_info', {}).get('target_word_len_in_section', 'NA'),
                step.get('target_word_info', {}).get('target_word_index_in_section', 'NA'),
                log.get('norm_fix_x', 'NA'),
                log.get('norm_fix_y', 'NA'),
                log.get('fix_x', 'NA'),
                log.get('fix_y', 'NA'),
                log.get('foveal seen letters', 'NA'),
                log.get('parafoveal seen letters', 'NA'),
                log.get('encoded internal word', 'NA'),
                log.get('word counter', 'NA'),
                log.get('recognize flag', 'NA'),
                log.get('completeness', 'NA'),
                # log.get('distances', 'NA'),
                # log.get('flag on target', 'NA'),
                # log.get('flag updated', 'NA'),
                log.get('reward', 'NA'),
                log.get('done action', 'NA'),
                log.get('done', 'NA')
            ]
            data_rows.append(row)

    detailed_df = pd.DataFrame(data_rows, columns=columns)

    # Also save the dataframe to a csv file
    detailed_df.to_csv(os.path.join(os.path.dirname(json_file_path), cons.SIM_DATA_CSV_FILE_NAME), index=False)

    return detailed_df


def plot_metrics_from_folder(folder_path, mode='test'):
    # Prepare lists to store the aggregated metrics
    w_values = []
    correct_fixations = []
    average_completeness = []
    average_fixations = []
    fixations_per_word_length = []

    # Loop over files in the specified folder
    for file in sorted(os.listdir(folder_path)):
        if file.startswith(mode):
            if file.endswith("_stats.txt"):  # Ensure we only process the right files
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as f:
                    data = f.readlines()

                # Extract the W value from the filename
                try:
                    w_value = int(file.split('_')[2][1:])
                    w_values.append(w_value)
                except (IndexError, ValueError):
                    continue  # Skip the file if W value cannot be extracted

                # Extract the metrics from the file
                correct_fixations.append(float(data[0].strip().split(': ')[1]))
                average_completeness.append(float(data[1].strip().split(': ')[1]))
                average_fixations.append(float(data[2].strip().split(': ')[1]))
                fixations_per_word_length.append(float(data[3].strip().split(': ')[1]))

    # Sort by W values to ensure correct plotting
    sorted_indices = sorted(range(len(w_values)), key=lambda i: w_values[i])
    w_values = [w_values[i] for i in sorted_indices]
    correct_fixations = [correct_fixations[i] for i in sorted_indices]
    average_completeness = [average_completeness[i] for i in sorted_indices]
    average_fixations = [average_fixations[i] for i in sorted_indices]
    fixations_per_word_length = [fixations_per_word_length[i] for i in sorted_indices]

    # Plotting the data with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(w_values, average_fixations, label='Average Number of Fixations', marker='o', color='green')
    ax1.set_xlabel('W Value')
    ax1.set_ylabel('Average Number of Fixations', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.plot(w_values, correct_fixations, label='Correct Inferences', marker='o', color='blue')
    ax2.plot(w_values, average_completeness, label='Average Activation Level / Completeness', marker='o', color='orange')
    # ax2.plot(w_values, fixations_per_word_length, label='Fixations per Word Length', marker='o', color='red')
    ax2.set_ylabel('Other Metrics', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    fig.suptitle('Comparison of Metrics Across W Values')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.875))
    ax1.grid(True)

    # Save the figure in the same folder
    figure_path = os.path.join(folder_path, f'{mode}_metrics_comparison_plot.png')
    plt.savefig(figure_path)


class VisualizeScanPath:

    def __init__(
            self,
            df: pd.DataFrame,
            amp_ratio: float,
            input_dir: str,
            mode: str,
            plot: bool = False,
    ):
        self._df = df
        self._amp_ratio = amp_ratio
        self._sim_results_dir = input_dir
        self._metadata = None
        self._output_dir_path = None
        self._self_claimed_mode = mode
        self._plot = plot

    def _get_metadata(
            self,
            mode: str,
    ):
        """Get the metadata from the json file."""
        # Get the current root directory
        root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Get the mode from the config yaml file
        with open(os.path.join(root_directory, "config.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Generate the metadata file path
        metadata_file_path = os.path.join(root_directory, "data", "gen_envs", config["resources"]["img_env_dir"],
                                          mode, cons.MD_FILE_NAME)
        # Load the metadata from the json file
        with open(metadata_file_path, 'r') as file:
            self._metadata = json.load(file)

    def _count(
            self,
            output_file_path: str,
    ):
        """Count the number of correct fixations."""
        df_terminations = self._df[self._df['done'] == True]

        # Get rows that have termination information
        total_num = len(df_terminations)

        # Get the success rate
        num_correct = 0
        for i, row in df_terminations.iterrows():
            if int(row["reward"]) >= 0:
                if 'reward' in df_terminations.columns:
                    if row["reward"] >= 0:
                        num_correct += 1
                else:
                    num_correct += 1
        success_rate = round(num_correct / total_num, 3)

        # Get the average completeness
        # To Do this only when the 'completeness' column is present in the dataframe
        if 'completeness' in df_terminations.columns:
            average_completeness = df_terminations['completeness'].mean()
        else:
            average_completeness = None

        # Filter out the termination steps (where 'done' is True)
        df_non_terminations = self._df[self._df['done'] == False]
        # Group by episodes (assuming there is an 'episode' column)
        fixation_counts = df_non_terminations.groupby('episode').size()
        # Compute the average fixation number across episodes
        average_fixation_number = fixation_counts.mean()
        # Get the average fixation number over word length
        word_lengths = self._df.groupby('episode')['target word len'].first()
        # Calculate the fixation per word length for each episode
        fixation_per_word_len = fixation_counts / word_lengths
        # Compute the average fixation per word length across all episodes
        average_fixation_per_word_len = fixation_per_word_len.mean()

        # Write information to a text file
        # Store the success rate in a text file
        with open(output_file_path, 'w') as file:
            file.write(f"The percentage of correct fixations is: {success_rate}")
            file.write(f"\nThe average completeness is: {average_completeness}")
            file.write(f"\nThe average number of fixations over episode is: {average_fixation_number}")
            file.write(f"\nThe average number of fixations per word length is: {average_fixation_per_word_len}")

        return success_rate

    @staticmethod
    def _draw(
            image_index: int,
            fixation_index: int,
            terminate_step: bool,
            num_saccades_in_one_word_step: int,
            img: Image,
            draw: ImageDraw,
            font: ImageFont,
            word_color: str,
            fix_x: float,
            fix_y: float,
            radius: float,
            foveal_width: int,
            foveal_height: int,
            parafoveal_width: int,
            parafoveal_height: int,
            peripheral_width: int,
            peripheral_height: int,
            img_width: int,
            img_height: int,
            target_word: str,
            imgs_save_path: str,
            words_on_generated_img: str,
            norm_x: float = 0,
            norm_y: float = 0,
            word_step: int = 0,
    ):

        """Draw the scan path from drawing from words because I need amplification visualizations."""
        # Split the text into lines and get the top position to start drawing
        # lines, line_height, len(lines), total_height
        lines, line_height, num_lines, total_height = gens.ImageGenerator.split_text_into_lines(
            selected_words=words_on_generated_img,
            draw=draw,
            font=font,
            image_width=img_width,
            image_height=img_height,
        )

        # Redraw the image
        # Get the max line width to plot text lines evenly
        line_widths = [gens.getsize(font, line)[0] for line in lines]
        max_line_width = max(line_widths)

        # Get the x and y initial/start positions
        max_x = img_width - max_line_width
        max_y = img_height - total_height
        x_init = max_x * norm_x
        y_init = max_y * norm_y

        # Get the max offsets for the letters for both x and y directions across all words and letters
        max_font_offset_x = max([gens.getsize(font, word)[2] for word in words_on_generated_img])
        max_font_offset_y = max([gens.getsize(font, word)[3] for word in words_on_generated_img])
        max_font_height = max([gens.getsize(font, word)[1] for word in words_on_generated_img])

        # current_top = top
        for idx_line, line in enumerate(lines):
            words_in_line = line.split(' ')
            line_width, line_height, left, _, right, bottom = gens.getsize(font, line)
            x = x_init - max_font_offset_x
            # y = y_init + idx_line * max_font_height
            y = y_init + idx_line * (max_font_height + cons.LINE_SPACING)  # Initial y position for the line

            for word in words_in_line:
                word_width, word_height, left, top, right, bottom = gens.getsize(font, word)
                # Draw the word and update x position for the next word
                draw.text((x, y), word, fill=word_color, font=font)

                x += word_width + gens.getsize(font=font, text=" ")[0]  # Add a space after each word

            # current_top += line_height  # Move the top position down for the next line

        if terminate_step is False or num_saccades_in_one_word_step <= 1:   # Remove this condition to draw the last fixation
            # Draw a rectangle area with the same size as the parafoveal area using blue color
            # Calculate the top-left and bottom-right corners of the parafoveal rectangle
            parafoveal_rect_top_left = (fix_x - foveal_width / 2, fix_y - parafoveal_height / 2)
            parafoveal_rect_bottom_right = (fix_x - foveal_width / 2 + parafoveal_width, fix_y + parafoveal_height / 2)
            # Draw the parafoveal rectangle
            draw.rectangle([parafoveal_rect_top_left, parafoveal_rect_bottom_right], outline="blue", width=1)
            # Draw a rectangle area with the same size as the foveal area using red color
            # Draw it after the parafoveal area to make it more visible
            draw.rectangle([fix_x - foveal_width / 2, fix_y - foveal_height / 2,
                            fix_x + foveal_width / 2, fix_y + foveal_height / 2], outline="red", width=2)

        # Save the image
        # image_filename = f"img{image_index}_word_step{word_step}_saccade_step{fixation_index}_{target_word}.jpg"
        image_filename = f"img{image_index}_word_step{word_step}_saccade_step{fixation_index}.jpg"
        save_filename = os.path.join(imgs_save_path, image_filename)
        img.save(save_filename)

    def _process_and_draw(self):
        """Process the simulation results from the read dataframe."""
        img_width = int(self._metadata[cons.md['config']][cons.md['img size']][0])
        img_height = int(self._metadata[cons.md['config']][cons.md['img size']][1])
        word_size = int(self._metadata[cons.md['config']][cons.md['word size']])
        foveal_size = (self._metadata[cons.md['config']][cons.md['foveal size']])
        parafoveal_size = (self._metadata[cons.md['config']][cons.md['parafoveal size']])
        peripheral_size = (self._metadata[cons.md['config']][cons.md['peripheral size']])
        foveal_width = int(foveal_size[0])
        foveal_height = int(foveal_size[1])
        parafoveal_width = int(parafoveal_size[0])
        parafoveal_height = int(parafoveal_size[1])
        peripheral_width = int(peripheral_size[0])
        peripheral_height = int(peripheral_size[1])

        bg_color = self._metadata[cons.md['config']][cons.md['background color']]
        font = ImageFont.truetype(cons.FONT_PATH, word_size)

        for i, row in self._df.iterrows():
            img_idx = int(row['episode'])

            # Get the word step if there is such a column in the dataframe
            num_saccades_in_one_word_step = 0
            if 'word step' in self._df.columns:
                word_step = int(row['word step'])
                # Count the number of saccades in one word step -- number of rows with the same word step
                num_saccades_in_one_word_step = len(self._df[self._df['word step'] == word_step])
            else:
                word_step = 0
                num_saccades_in_one_word_step = len(self._df[self._df['episode'] == img_idx])

            words = ast.literal_eval(row['words'])

            target_word = row['target word']

            terminate = row['done']

            fix_x = float(row['fix x'])
            fix_y = float(row['fix y'])

            # Get positional information
            img_idx_in_metadata = int(row['image idx'])

            norm_x = float(self._metadata[cons.md['images']][img_idx_in_metadata][cons.md['words metadata']][0][
                               cons.md["position"]][cons.md["x_norm"]])
            norm_y = float(self._metadata[cons.md['images']][img_idx_in_metadata][cons.md['words metadata']][0][
                               cons.md["position"]][cons.md["y_norm"]])
            words = self._metadata[cons.md['images']][img_idx_in_metadata][cons.md['selected words']]

            # Draw the amplified word image and the scan path
            img = Image.new('RGB', (img_width, img_height), bg_color)
            draw = ImageDraw.Draw(img)
            self._draw(
                image_index=img_idx,
                fixation_index=int(i),
                terminate_step=terminate,
                num_saccades_in_one_word_step=num_saccades_in_one_word_step,
                img=img,
                draw=draw,
                font=font,
                word_color="black",
                fix_x=fix_x,
                fix_y=fix_y,
                radius=cons.FOVEA_FACTOR * word_size,
                foveal_width=foveal_width,
                foveal_height=foveal_height,
                parafoveal_width=parafoveal_width,
                parafoveal_height=parafoveal_height,
                peripheral_width=peripheral_width,
                peripheral_height=peripheral_height,
                img_width=img_width,
                img_height=img_height,
                target_word=target_word,
                imgs_save_path=self._output_dir_path,
                words_on_generated_img=words,
                norm_x=norm_x,
                norm_y=norm_y,
                word_step=word_step,
            )

    def plot(self):
        """Plot the scan paths."""
        # Get the prefix of the file
        if self._self_claimed_mode == cons.TEST or self._self_claimed_mode == cons.TRAIN:
            mode, prefix, weight = get_prefix(self._sim_results_dir)
            text_file_name = f'{mode}_{prefix}_{weight}_stats.txt'
        elif self._self_claimed_mode == cons.SIMULATE:
            mode, prefix = get_prefix(self._sim_results_dir)
            text_file_name = f'{mode}_{prefix}_stats.txt'
        else:
            raise ValueError(f"Invalid mode {self._self_claimed_mode}. Please specify a valid mode.")

        if self._plot:

            # Generate the output folder path
            output_dir_path = os.path.join(os.path.dirname(self._sim_results_dir), f'{mode}_{prefix}_vis')

            # Check if the folder exists
            if os.path.exists(output_dir_path):
                # Delete the folder and its contents
                shutil.rmtree(output_dir_path)

            # Create the folder again
            os.makedirs(output_dir_path)

            # if not os.path.exists(output_dir_path):
            #     os.makedirs(output_dir_path)

            self._output_dir_path = output_dir_path

            # Get the metadata
            self._get_metadata(mode=mode)

            # Process the simulation results
            self._process_and_draw()
        else:
            # Get the metadata
            self._get_metadata(mode=mode)

        # Calculate the count results
        self._count(output_file_path=os.path.join(os.path.dirname(self._sim_results_dir), text_file_name))


def visualize_rl_test_data(
        root_dir: str,
        mode: str,
        num_episodes: int,
        weights_spec: list,
        rl_sim_data_dir_name: str = '0812_oculomotor_controller_02_rl_model_31000000_steps',
):
    for weight in range(weights_spec[0], weights_spec[1]+1, weights_spec[2]):

        start_time = time.time()

        # Input the dir
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        rl_sim_data_dir_name = rl_sim_data_dir_name
        rl_sim_data_file_name = f'{mode}_{num_episodes}ep_w{weight}_logger.csv'
        rl_sim_data_file_path = os.path.join(root_dir, 'data', 'sim_results', rl_sim_data_dir_name, rl_sim_data_file_name)
        # Load the data from the csv file
        rl_sim_data_frame = pd.read_csv(rl_sim_data_file_path)

        vis_rl_sim = VisualizeScanPath(
            df=rl_sim_data_frame,   # rl_sim_data_frame, simulation_data_frame
            amp_ratio=1,
            input_dir=rl_sim_data_file_path,    # rl_sim_data_file_path, simulation_data_json_file_path,
            mode=cons.TEST,
            plot=False,
        )
        vis_rl_sim.plot()

        print(f"Weight: {weight}. Time taken: {round((time.time() - start_time), 2)} seconds")

    # Draw some figures to visualize the metrics
    folder_data_dir_name = rl_sim_data_dir_name
    folder_data_path = os.path.join(root_dir, 'data', 'sim_results', folder_data_dir_name)
    plot_metrics_from_folder(folder_data_path, mode=mode)


def visualize_simulation_data(
        root_dir: str,
        simulation_data_dir_name: str = "sim_raw_data_08_13_11_56_image_3",
):
    start_time = time.time()
    simulation_data_dir_name = simulation_data_dir_name
    simulation_data_json_file_name = cons.SIM_DATA_JS_FILE_NAME
    simulation_data_json_file_path = os.path.join(root_dir, 'data', 'sim_results', simulation_data_dir_name,
                                                  simulation_data_json_file_name)

    simulation_data_frame = convert_json_to_dataframe(simulation_data_json_file_path)

    # Plot the data
    vis_simulation = VisualizeScanPath(
        df=simulation_data_frame,  # rl_sim_data_frame, simulation_data_frame
        amp_ratio=1,
        input_dir=simulation_data_json_file_path,  # rl_sim_data_file_path, simulation_data_json_file_path,
        mode=cons.SIMULATE, # Usually we do not need to change this
        plot=True,
    )
    vis_simulation.plot()
    print(f"Time taken: {round((time.time() - start_time), 2)} seconds")


if __name__ == "__main__":

    # Input the dir
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ----------------------------------------------------------------------------------------------------------------------
    # # Test datasets one by one
    # visualize_rl_test_data(
    #     root_dir=root_dir,
    #     mode="test",
    #     num_episodes=500,
    #     weights_spec=[1, 20, 1],
    #     rl_sim_data_dir_name='0816_oculomotor_controller_02_rl_model_14000000_steps',
    # )

    # ----------------------------------------------------------------------------------------------------------------------
    # Simulation data visualization
    visualize_simulation_data(
        root_dir=root_dir,
        simulation_data_dir_name="sim_raw_data_08_18_13_54_image_0_time_penalty_weight_5"
    )
