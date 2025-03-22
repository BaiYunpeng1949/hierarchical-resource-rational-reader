import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict
import string
from scipy import stats
import statsmodels.api as sm

def clean_word(word):
    """Clean word by removing punctuation and converting to lowercase"""
    word = word.lower().strip()
    word = ''.join(char for char in word if char not in string.punctuation)
    return word

def collect_word_statistics():
    """Collect statistics for each unique word occurrence across all simulated trials"""
    input_dir = "/home/baiy4/reader-agent-zuco/results/section2/_raw_simulated_results"
    
    # Dictionary to store word statistics
    # Key will be (sentence_id, word_id, word) to track unique occurrences
    word_stats = defaultdict(lambda: {
        'total_occurrences': 0,
        'skip_count': 0,
        'regression_count': 0,
        'length': 0,
        'frequency': 0,
        'log_frequency': 0,
        'difficulty': 0,
        "predictability": 0,
        "logit_predictability": 0,
        'original_word': '',  # Original word with punctuation
        'word_clean': '',  # Cleaned word
        'sentence': '',  # Full sentence
        'position': 0  # Word position in sentence
    })
    
    # Process simulated data
    sim_file = os.path.join(input_dir, "raw_simulated_results.json")
    try:
        with open(sim_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Process each trial
            for trial in data:
                # Skip None trials
                if trial is None:
                    continue
                    
                # Skip trials without words
                if 'words' not in trial or not trial['words']:
                    continue
                    
                sentence_id = trial.get('sentence_id', '')
                sentence_text = trial.get('sentence_content', '')
                    
                for word_idx, word_info in enumerate(trial['words']):
                    # Get original and cleaned word
                    original_word = word_info['word']  # Original word with punctuation
                    clean_word_text = word_info['word_clean']  # Already cleaned word
                    word_id = word_info.get('word_id', '')
                    
                    # Create unique key for this word occurrence
                    word_key = (str(sentence_id), str(word_id), clean_word_text)
                    
                    # Update counts
                    word_stats[word_key]['total_occurrences'] += 1
                    word_stats[word_key]['skip_count'] += 1 if word_info['is_first_pass_skip'] else 0
                    word_stats[word_key]['regression_count'] += 1 if word_info['is_regression_target'] else 0
                    
                    # Update features (will average later)
                    word_stats[word_key]['length'] = len(clean_word_text)
                    word_stats[word_key]['frequency'] += word_info.get('frequency_per_million', 0)
                    word_stats[word_key]['log_frequency'] += word_info.get('log_frequency', 0)
                    word_stats[word_key]['difficulty'] += word_info.get('difficulty', 0)
                    word_stats[word_key]['predictability'] += word_info.get('predictability', 0)
                    word_stats[word_key]['logit_predictability'] += word_info.get('logit_predictability', 0)
                    word_stats[word_key]['original_word'] = original_word
                    word_stats[word_key]['word_clean'] = clean_word_text
                    word_stats[word_key]['sentence'] = sentence_text
                    word_stats[word_key]['position'] = word_idx
    
    except Exception as e:
        print(f"Error processing simulated data file {sim_file}: {str(e)}")
    
    # Calculate averages and probabilities
    word_data = []
    for (sentence_id, word_id, word), stats in word_stats.items():
        n = stats['total_occurrences']
        if n > 0:  # Avoid division by zero
            word_data.append({
                'sentence_id': sentence_id,
                'word_id': word_id,
                'word': word,
                'original_word': stats['original_word'],
                'position': stats['position'],
                'sentence': stats['sentence'],
                'length': stats['length'],
                'frequency': stats['frequency'] / n,
                'log_frequency': stats['log_frequency'] / n,
                'difficulty': stats['difficulty'] / n,
                'skip_probability': stats['skip_count'] / n,
                'regression_probability': stats['regression_count'] / n,
                'total_occurrences': n
            })
    
    df = pd.DataFrame(word_data)
    
    # Print some basic statistics
    print(f"\nProcessed {len(word_stats)} unique word occurrences")
    print(f"Found {sum(stats['skip_count'] for stats in word_stats.values())} total skips")
    print(f"Found {sum(stats['regression_count'] for stats in word_stats.values())} total regressions")
    
    # Save to CSV
    output_dir = "/home/baiy4/reader-agent-zuco/results/section2/_simulation_effects_analysis"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "simulated_word_analysis.csv"), index=False)
    
    # Save to JSON
    word_features = {}
    for _, row in df.iterrows():
        # Convert tuple key to string format
        word_key = f"{row['sentence_id']}_{row['word_id']}_{row['word']}"
        word_features[word_key] = {
            'length': row['length'],
            'frequency': row['frequency'],
            'log_frequency': row['log_frequency'],
            'difficulty': row['difficulty'],
            'skip_probability': row['skip_probability'],
            'regression_probability': row['regression_probability'],
            'total_occurrences': row['total_occurrences'],
            'position': row['position'],
            'sentence': row['sentence']
        }
    
    with open(os.path.join(output_dir, "word_features.json"), 'w', encoding='utf-8') as f:
        json.dump(word_features, f, indent=2, ensure_ascii=False)
    
    return df

def perform_regression_analysis(x, y, x_name, y_name):
    """Perform detailed regression analysis including statistical tests"""
    # Pearson correlation
    correlation, p_value = stats.pearsonr(x, y)
    
    # Linear regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'slope': model.params[1],
        'intercept': model.params[0],
        'slope_p_value': model.pvalues[1],
        'f_statistic': model.fvalue,
        'f_p_value': model.f_pvalue,
        'n_observations': len(x),
        'x_name': x_name,
        'y_name': y_name
    }

def analyze_and_plot_relationships(df, output_dir):
    """Analyze relationships and create plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for all plots
    plt.style.use('seaborn-whitegrid')
    
    # Common plot settings
    plot_settings = {
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    }
    plt.rcParams.update(plot_settings)
    
    # 1. Word Length Effect on Skipping
    plt.figure()
    sns.scatterplot(data=df, x='length', y='skip_probability', 
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    sns.regplot(data=df, x='length', y='skip_probability', 
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    corr = df['length'].corr(df['skip_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Length Effect on Skipping Probability (Simulated)')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Skipping Probability')
    plt.savefig(os.path.join(output_dir, 'length_skip_effect.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Word Frequency Effect on Skipping
    plt.figure()
    sns.scatterplot(data=df, x='log_frequency', y='skip_probability', 
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    sns.regplot(data=df, x='log_frequency', y='skip_probability', 
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    corr = df['log_frequency'].corr(df['skip_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Frequency Effect on Skipping Probability (Simulated)')
    plt.xlabel('Log Frequency (per million)')
    plt.ylabel('Skipping Probability')
    plt.savefig(os.path.join(output_dir, 'frequency_skip_effect.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word Difficulty Effect on Regression
    plt.figure()
    sns.scatterplot(data=df, x='difficulty', y='regression_probability', 
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    sns.regplot(data=df, x='difficulty', y='regression_probability', 
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    corr = df['difficulty'].corr(df['regression_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Difficulty Effect on Regression Probability (Simulated)')
    plt.xlabel('Word Difficulty')
    plt.ylabel('Regression Probability')
    plt.savefig(os.path.join(output_dir, 'difficulty_regression_effect.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add skip-regression relationship analysis
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with size based on occurrences
    sns.scatterplot(data=df, x='skip_probability', y='regression_probability',
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    
    # Add regression line with confidence band
    sns.regplot(data=df, x='skip_probability', y='regression_probability',
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    # Add correlation coefficient
    corr = df['skip_probability'].corr(df['regression_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    # Add mean lines
    plt.axvline(x=df['skip_probability'].mean(), color='blue', linestyle='--', alpha=0.3, label='Mean skip prob.')
    plt.axhline(y=df['regression_probability'].mean(), color='green', linestyle='--', alpha=0.3, label='Mean regression prob.')
    
    plt.title('Relationship between Word Skipping and Regression Probabilities (Simulated)')
    plt.xlabel('Skipping Probability')
    plt.ylabel('Regression Probability')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'skip_regression_relationship.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed CSV files with more information
    
    # 1. Skipping analysis
    skip_analysis = df[[
        'word', 'original_word', 'length', 'frequency', 'log_frequency', 
        'skip_probability', 'total_occurrences'
    ]].sort_values('skip_probability', ascending=False)
    skip_analysis.to_csv(os.path.join(output_dir, 'word_skipping_analysis.csv'), index=False)
    
    # 2. Regression analysis
    regression_analysis = df[[
        'word', 'original_word', 'difficulty', 'regression_probability', 
        'total_occurrences'
    ]].sort_values('regression_probability', ascending=False)
    regression_analysis.to_csv(os.path.join(output_dir, 'word_regression_analysis.csv'), index=False)
    
    # 3. Full dataset
    df.to_csv(os.path.join(output_dir, 'all_words_regression_and_skip_probabilities.csv'), index=False)
    
    # Calculate summary statistics
    summary_stats = {
        'skipping': {
            'mean_probability': df['skip_probability'].mean(),
            'std_probability': df['skip_probability'].std(),
            'correlation_with_length': df['skip_probability'].corr(df['length']),
            'correlation_with_log_freq': df['skip_probability'].corr(df['log_frequency'])
        },
        'regression': {
            'mean_probability': df['regression_probability'].mean(),
            'std_probability': df['regression_probability'].std(),
            'correlation_with_difficulty': df['regression_probability'].corr(df['difficulty'])
        }
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("SIMULATED DATA ANALYSIS:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total unique words analyzed: {len(df)}\n")
        f.write(f"\nSkipping Statistics:\n")
        f.write(f"Mean skip probability: {summary_stats['skipping']['mean_probability']:.3f}\n")
        f.write(f"Correlation with length: {summary_stats['skipping']['correlation_with_length']:.3f}\n")
        f.write(f"Correlation with log frequency: {summary_stats['skipping']['correlation_with_log_freq']:.3f}\n")
        f.write(f"\nRegression Statistics:\n")
        f.write(f"Mean regression probability: {summary_stats['regression']['mean_probability']:.3f}\n")
        f.write(f"Correlation with difficulty: {summary_stats['regression']['correlation_with_difficulty']:.3f}\n")
        f.write("\n")
    
    return summary_stats

def main():
    # Set up directories
    output_dir = "/home/baiy4/reader-agent-zuco/results/section2/_simulation_effects_analysis"
    plots_dir = output_dir
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect word statistics
    print("Collecting word statistics...")
    df = collect_word_statistics()
    
    # Analyze and create plots
    print("Analyzing relationships and creating plots...")
    summary_stats = analyze_and_plot_relationships(df, plots_dir)
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"Total unique words analyzed: {len(df)}")
    print(f"\nSkipping Statistics:")
    print(f"Mean skip probability: {summary_stats['skipping']['mean_probability']:.3f}")
    print(f"Correlation with length: {summary_stats['skipping']['correlation_with_length']:.3f}")
    print(f"Correlation with log frequency: {summary_stats['skipping']['correlation_with_log_freq']:.3f}")
    
    print(f"\nRegression Statistics:")
    print(f"Mean regression probability: {summary_stats['regression']['mean_probability']:.3f}")
    print(f"Correlation with difficulty: {summary_stats['regression']['correlation_with_difficulty']:.3f}")
    
    print(f"\nResults saved in: {output_dir}")

if __name__ == "__main__":
    main() 