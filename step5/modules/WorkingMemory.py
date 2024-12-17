import yaml

from step5.modules.llm_envs.LLMSupervisoryController import LLMSupervisoryController
from step5.modules.llm_envs._LLMShortTermMemory import LLMShortTermMemory

import step5.utils.constants as const


class WorkingMemory:
    def __init__(
            self,
            config: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\config.yaml',
            task_specification: str = "Read the text and answer the questions.",
    ):
        """
        Updated in 22 July 2024
        Will come back to here later, or just delete this one and directly implement everything in the simulator

        Initialize the Working Memory, which is responsible for three main parts:
            1) Short-Term Memory (STM): The STM is responsible for both phonological and visuospatial information.
            2) Central Executive (CE): The CE is responsible for the control of the attention and the flow of information.
                But we call it the Supervisory Controller here.
            3) Episodic Buffer (EB): The EB is responsible for the memory gists, i.e., the summarised content information.

        TODO maybe rename the modules later, e.g., STM, LTM, WM, etc.
            Figma Ref: https://www.figma.com/board/T9YEXqz2NukOfcLLmHUP4L/18-June-Architecture?node-id=0-1&t=6eo2ZwlExnvK4qnC-0
        TODO Implementation Roadmap:
            1) SC with only the comprehension only one-section reading ability: read, comprehend, and answer questions properly.
                Main function: Set up the architecture.
            2) Plus the memory loss while reading and regression, some content will be summarised as gists and some are forgotten.
                a) Implement the architecture as the memory structure: STM, LTM, WM, etc. Especially the memory gists -- episodic buffer
                b) Implement the memory loss and regression control.
            3) Plus the ability to interrupt the reading process.
            4) Plus the ability to reading multiple sections, and implement the image input,
                auto-segmentation (or manually is fine, or call it segmenting different text fragments), and auto blurry.
                The ability to do the "diet" planning.

        Ref: Memory gist paper:
            1) A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts, https://arxiv.org/pdf/2402.09727
            2）Cognitive Structures of Content for Controlled Summarization, https://era.ed.ac.uk/bitstream/handle/1842/41676/Cardenas%20AcostaR_2024.pdf?sequence=1&isAllowed=y

        When gisting, the paper https://arxiv.org/pdf/2402.09727 use page as a segmentation unit, called "episodic pagination".
            We could determine our own segmentation unit, e.g., paragraph, section, etc.
            I could determine the gisting segmentation using the "Compression Rate (CR)" concept from this paper;
            or set a default value and let the agent to adaptively adjust it.

        :param config: The configuration file path.
        :param task_specification: The task specification.
        """

        print(f"{const.LV_ONE_DASHES}Supervisory Controller -- Initialize the Supervisory Controller.")

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        self._config_wm = self._config['simulate']['working_memory']
        self._config_stm = self._config['simulate']['working_memory']['short_term_memory']

        # Initialize the Working Memory Components here
        self._stm = LLMShortTermMemory(
            stm_capacity=self._config_stm['capacity'],
        )
        self._sc = LLMSupervisoryController(
            config=config,
            # task_specification=self._config_wm['task_spec'],
        )
        # TODO change this to not only the comprehender, but more than that,
        #  a supervisory controller that could comprehend,
        #  reason, solve problem, and plan

    def reset(self):
        """
        Reset the Supervisory Controller and its related modules.
        :return:
        """
        pass

    def supervise(self):
        pass

    def interrupt(self):
        """
        Interrupt the reading process determined by the reading goal,
            current reading progress, reading strategy, and comprehension ability.
        :return: Interrupt or not.
        """
        interrupt = False
        # TODO Implement the interrupt logic here later
        return interrupt


if __name__ == '__main__':
    # Test the Supervisory Controller
    sc = WorkingMemory()
    sc.supervise()
