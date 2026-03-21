This is the experiment running for replicating effects from the paper: The impact of reading time constraints on text comprehension 1 and eye movements

No need to replicate the exact values, because their dataset is on French; but only the general trends.

Trends identified in the paper:

ariable	Direction with stronger time constraint	Magnitude	Calculation
Fixation duration	↓ shorter	–14 % (≈ 22 ms less) 

The impact of reading time cons…

	Mean duration of fixations (ms)
% time in fixation	↓ slightly	–5 percentage points 

The impact of reading time cons…

	Total fixation time / trial time
Saccade amplitude	↑ larger	+16 % (≈ +0.73 cm) 

The impact of reading time cons…

	Average horizontal distance (cm or °) per saccade
Saccade rate	↑ marginally	+0.2 saccades s⁻¹ 

The impact of reading time cons…

	# saccades / s per trial
Left-to-right gaze velocity	↑ linearly	+37 % (14.2 → 19.4 cm s⁻¹ ≈ 262 → 359 wpm) 

The impact of reading time cons…

	Total forward-saccade distance / trial duration
Regressive saccades (%)	↓ nearly halved	14.7 → 7.7 % 

The impact of reading time cons…

	# right→left saccades / total
Fixated content words (%)	↓ but still high	91 → 77 % 

The impact of reading time cons…

	# content words fixated ≥1× / total content words
Sentence wrap-up effect	↓ reduced but present	extra 60 ms (free) → 22 ms (6.3 s) 

The impact of reading time cons…

	Last-fixation duration – mean other fixations


The dataset we are using is the simulation eye movement data after **parameter inference**.

To process these data for targeted metrics, see `build_french_corpus_effects_metris.py` outside.

To plot, see `plot_french_corpus_effects.py` outside.

NOTE: the `lightweight_metadata.json` was trimed from original `metadata.json` by dumping unnecessary keys. See how was it done here: `/home/baiy4/reader-agent-zuco/step5/simulators/simulator_v20250604/assets/experiment/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400/simulate`.