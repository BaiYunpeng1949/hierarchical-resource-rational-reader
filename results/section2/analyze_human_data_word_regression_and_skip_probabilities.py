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
    """Collect statistics for each unique word occurrence across all participants and trials"""
    input_dir = "/home/baiy4/reader-agent-zuco/results/section2/_raw_human_data"
    
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
        'predictability': 0,
        'logit_predictability': 0,
        'original_word': '',  # Original word with punctuation
        'word_clean': '',  # Cleaned word
        'sentence': '',  # Full sentence
        'position': 0  # Word position in sentence
    })
    
    # Process each participant's data
    for file_path in glob(os.path.join(input_dir, "*_reading_patterns.json")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Process each sentence
                for sentence in data:
                    # Skip None sentences
                    if sentence is None:
                        continue
                        
                    # Skip sentences without words
                    if 'words' not in sentence or not sentence['words']:
                        continue
                        
                    sentence_id = sentence.get('sentence_id', '')
                    sentence_text = sentence.get('sentence', '')
                        
                    for word_idx, word_info in enumerate(sentence['words']):
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
                        word_stats[word_key]['frequency'] += word_info['frequency_per_million']
                        word_stats[word_key]['log_frequency'] += word_info['log_frequency']
                        word_stats[word_key]['difficulty'] += word_info['difficulty']
                        word_stats[word_key]['predictability'] += word_info['predictability']
                        word_stats[word_key]['logit_predictability'] += word_info['logit_predictability']
                        word_stats[word_key]['original_word'] = original_word
                        word_stats[word_key]['word_clean'] = clean_word_text
                        word_stats[word_key]['sentence'] = sentence_text
                        word_stats[word_key]['position'] = word_idx
        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
    
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
                'predictability': stats['predictability'] / n,
                'logit_predictability': stats['logit_predictability'] / n,
                'skip_probability': stats['skip_count'] / n,
                'regression_probability': stats['regression_count'] / n,
                'total_occurrences': n
            })
    
    df = pd.DataFrame(word_data)
    
    # Ensure predictability is between 0 and 1
    df['predictability'] = df['predictability'].clip(0, 1)
    
    # Add predictability class based on logit values
    df['pred_class'] = pd.cut(df['logit_predictability'],
                             bins=[-np.inf, -1.5, -1.0, -0.5, 0, np.inf],
                             labels=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    
    # Print some basic statistics
    print(f"\nProcessed {len(word_stats)} unique word occurrences")
    print(f"Found {sum(stats['skip_count'] for stats in word_stats.values())} total skips")
    print(f"Found {sum(stats['regression_count'] for stats in word_stats.values())} total regressions")
    print(f"Average predictability: {df['predictability'].mean():.3f}")
    print("\nPredictability class distribution:")
    print(df['pred_class'].value_counts().sort_index())
    
    # Save to CSV
    output_dir = "/home/baiy4/ScanDL/scripts/data/zuco/bai_word_probability_analysis"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "complete_word_analysis.csv"), index=False)
    
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
            'predictability': row['predictability'],
            'logit_predictability': row['logit_predictability'],
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
    # Create weighted scatter plot based on occurrence count
    sns.scatterplot(data=df, x='length', y='skip_probability', 
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    # Add regression line with confidence band
    sns.regplot(data=df, x='length', y='skip_probability', 
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    # Add correlation coefficient
    corr = df['length'].corr(df['skip_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Length Effect on Skipping Probability')
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
    
    # Add correlation coefficient
    corr = df['log_frequency'].corr(df['skip_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Frequency Effect on Skipping Probability')
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
    
    # Add correlation coefficient
    corr = df['difficulty'].corr(df['regression_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Difficulty Effect on Regression Probability')
    plt.xlabel('Word Difficulty')
    plt.ylabel('Regression Probability')
    plt.savefig(os.path.join(output_dir, 'difficulty_regression_effect.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Word Predictability Effect on Skipping
    plt.figure()
    sns.scatterplot(data=df, x='predictability', y='skip_probability', 
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    sns.regplot(data=df, x='predictability', y='skip_probability', 
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    # Add correlation coefficient
    corr = df['predictability'].corr(df['skip_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Word Predictability Effect on Skipping Probability')
    plt.xlabel('Word Predictability')
    plt.ylabel('Skipping Probability')
    plt.savefig(os.path.join(output_dir, 'predictability_skip_effect.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add new plot for logit predictability distribution and regression
    plt.figure(figsize=(12, 8))
    
    # Create main scatter plot with size based on occurrences
    sns.scatterplot(data=df, x='logit_predictability', y='skip_probability',
                    size='total_occurrences', sizes=(20, 200),
                    alpha=0.3, legend='brief')
    
    # Add regression line with confidence band
    sns.regplot(data=df, x='logit_predictability', y='skip_probability',
                scatter=False,
                line_kws={'linewidth': 2, 'color': 'red'},
                ci=95)
    
    # Add vertical lines for class boundaries
    plt.axvline(x=-1.5, color='r', linestyle='--', alpha=0.5, label='Class boundaries')
    plt.axvline(x=-1.0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.5, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Add correlation coefficient
    corr = df['logit_predictability'].corr(df['skip_probability'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('Logit Predictability Effect on Skipping Probability')
    plt.xlabel('Logit Predictability')
    plt.ylabel('Skip Probability')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'logit_predictability_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add distribution histogram as a separate plot
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='logit_predictability', bins=30)
    plt.axvline(x=-1.5, color='r', linestyle='--', alpha=0.5, label='Class boundaries')
    plt.axvline(x=-1.0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.5, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Distribution of Logit Predictability Values')
    plt.xlabel('Logit Predictability')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'logit_predictability_histogram.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add class-based skipping analysis
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='pred_class', y='skip_probability')
    plt.title('Skipping Probability by Predictability Class')
    plt.xlabel('Predictability Class')
    plt.ylabel('Skipping Probability')
    plt.savefig(os.path.join(output_dir, 'pred_class_skip_effect.png'),
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
    
    plt.title('Relationship between Word Skipping and Regression Probabilities')
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
        'predictability', 'skip_probability', 'total_occurrences'
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
            'correlation_with_log_freq': df['skip_probability'].corr(df['log_frequency']),
            'correlation_with_predictability': df['skip_probability'].corr(df['predictability'])
        },
        'regression': {
            'mean_probability': df['regression_probability'].mean(),
            'std_probability': df['regression_probability'].std(),
            'correlation_with_difficulty': df['regression_probability'].corr(df['difficulty'])
        }
    }
    
    # Save summary statistics with more detail
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("Word Skipping Analysis:\n")
        f.write(f"Total unique words analyzed: {len(df)}\n")
        f.write(f"Mean skip probability: {summary_stats['skipping']['mean_probability']:.3f}\n")
        f.write(f"Std skip probability: {summary_stats['skipping']['std_probability']:.3f}\n")
        f.write(f"Correlation with length: {summary_stats['skipping']['correlation_with_length']:.3f}\n")
        f.write(f"Correlation with log frequency: {summary_stats['skipping']['correlation_with_log_freq']:.3f}\n")
        f.write(f"Correlation with predictability: {summary_stats['skipping']['correlation_with_predictability']:.3f}\n\n")
        
        f.write("Word Regression Analysis:\n")
        f.write(f"Mean regression probability: {summary_stats['regression']['mean_probability']:.3f}\n")
        f.write(f"Std regression probability: {summary_stats['regression']['std_probability']:.3f}\n")
        f.write(f"Correlation with difficulty: {summary_stats['regression']['correlation_with_difficulty']:.3f}\n")
        
        # Add distribution statistics
        f.write("\nDistribution Statistics:\n")
        f.write("Word Lengths: min={:.0f}, max={:.0f}, mean={:.1f}\n".format(
            df['length'].min(), df['length'].max(), df['length'].mean()))
        f.write("Log Frequencies: min={:.2f}, max={:.2f}, mean={:.2f}\n".format(
            df['log_frequency'].min(), df['log_frequency'].max(), df['log_frequency'].mean()))
        f.write("Difficulties: min={:.2f}, max={:.2f}, mean={:.2f}\n".format(
            df['difficulty'].min(), df['difficulty'].max(), df['difficulty'].mean()))
        f.write("Predictabilities: min={:.2f}, max={:.2f}, mean={:.2f}\n".format(
            df['predictability'].min(), df['predictability'].max(), df['predictability'].mean()))
    
    # Perform statistical analyses for all reading effects
    effects_analyses = []
    
    # 1. Length vs Skip Probability
    length_skip = perform_regression_analysis(
        df['length'], df['skip_probability'],
        'Word Length', 'Skip Probability'
    )
    effects_analyses.append(length_skip)
    
    # 2. Log Frequency vs Skip Probability
    freq_skip = perform_regression_analysis(
        df['log_frequency'], df['skip_probability'],
        'Log Frequency', 'Skip Probability'
    )
    effects_analyses.append(freq_skip)
    
    # 3. Difficulty vs Regression Probability
    diff_reg = perform_regression_analysis(
        df['difficulty'], df['regression_probability'],
        'Word Difficulty', 'Regression Probability'
    )
    effects_analyses.append(diff_reg)
    
    # 4. Predictability vs Skip Probability
    pred_skip = perform_regression_analysis(
        df['predictability'], df['skip_probability'],
        'Predictability', 'Skip Probability'
    )
    effects_analyses.append(pred_skip)
    
    # 5. Logit Predictability vs Skip Probability
    logit_skip = perform_regression_analysis(
        df['logit_predictability'], df['skip_probability'],
        'Logit Predictability', 'Skip Probability'
    )
    effects_analyses.append(logit_skip)
    
    # 6. Skip Probability vs Regression Probability
    skip_reg = perform_regression_analysis(
        df['skip_probability'], df['regression_probability'],
        'Skip Probability', 'Regression Probability'
    )
    effects_analyses.append(skip_reg)
    
    # Save detailed effects analyses
    with open(os.path.join(output_dir, 'effects_analyses.txt'), 'w') as f:
        f.write("DETAILED WORD READING EFFECTS ANALYSES\n")
        f.write("====================================\n\n")
        
        f.write("STATISTICAL MEASURES EXPLAINED:\n")
        f.write("------------------------------\n")
        f.write("- Pearson r: Measures the strength and direction of the linear relationship between two variables.\n")
        f.write("  Values range from -1 to +1, where:\n")
        f.write("  • -1 indicates perfect negative correlation\n")
        f.write("  • 0 indicates no correlation\n")
        f.write("  • +1 indicates perfect positive correlation\n\n")
        
        f.write("- R-squared: The coefficient of determination, indicates how much variance in the dependent\n")
        f.write("  variable is explained by the independent variable (percentage when multiplied by 100).\n")
        f.write("  Values range from 0 to 1, where higher values indicate better fit.\n\n")
        
        f.write("- Adjusted R-squared: Similar to R-squared but adjusted for the number of predictors.\n")
        f.write("  More suitable for comparing models with different numbers of predictors.\n\n")
        
        f.write("- Slope: The change in Y for a one-unit change in X. Positive values indicate that Y\n")
        f.write("  increases as X increases; negative values indicate that Y decreases as X increases.\n\n")
        
        f.write("- F-statistic: Tests whether the model as a whole is statistically significant.\n")
        f.write("  Larger values suggest stronger evidence against the null hypothesis.\n\n")
        
        f.write("- P-values: The probability of obtaining test results at least as extreme as the observed\n")
        f.write("  results, assuming the null hypothesis is true. Values < 0.05 are typically\n")
        f.write("  considered statistically significant.\n\n")
        
        f.write("\nANALYSIS RESULTS BY EFFECT TYPE\n")
        f.write("============================\n")
        
        # Group analyses by type
        skipping_effects = [analysis for analysis in effects_analyses 
                          if 'Skip Probability' in analysis['y_name']]
        regression_effects = [analysis for analysis in effects_analyses 
                            if 'Regression Probability' in analysis['y_name']]
        
        # Write skipping effects
        f.write("\nWORD SKIPPING EFFECTS\n")
        f.write("--------------------\n")
        for analysis in skipping_effects:
            f.write(f"\n{analysis['x_name']} Effect on Word Skipping\n")
            f.write("----------------------------------------\n")
            f.write(f"Number of observations: {analysis['n_observations']}\n")
            
            f.write("\nCorrelation Analysis:\n")
            f.write(f"Pearson r: {analysis['correlation']:.3f}")
            f.write(" (Strong)" if abs(analysis['correlation']) > 0.5 else 
                    " (Moderate)" if abs(analysis['correlation']) > 0.3 else 
                    " (Weak)")
            f.write("\n")
            f.write(f"Correlation p-value: {analysis['p_value']:.6f}")
            f.write(" ***" if analysis['p_value'] < 0.001 else 
                    " **" if analysis['p_value'] < 0.01 else 
                    " *" if analysis['p_value'] < 0.05 else 
                    " (n.s.)")
            f.write("\n")
            
            f.write("\nRegression Analysis:\n")
            f.write(f"R-squared: {analysis['r_squared']:.3f}")
            f.write(f" ({analysis['r_squared']*100:.1f}% of variance explained)\n")
            f.write(f"Adjusted R-squared: {analysis['adj_r_squared']:.3f}\n")
            
            f.write(f"Slope: {analysis['slope']:.3f}")
            f.write(" (Positive relationship)" if analysis['slope'] > 0 else 
                    " (Negative relationship)" if analysis['slope'] < 0 else 
                    " (No relationship)")
            f.write("\n")
            f.write(f"Intercept: {analysis['intercept']:.3f}")
            f.write(f" (Predicted {analysis['y_name']} when {analysis['x_name']} = 0)\n")
            
            f.write(f"Slope p-value: {analysis['slope_p_value']:.6f}")
            f.write(" ***" if analysis['slope_p_value'] < 0.001 else 
                    " **" if analysis['slope_p_value'] < 0.01 else 
                    " *" if analysis['slope_p_value'] < 0.05 else 
                    " (n.s.)")
            f.write("\n")
            
            f.write(f"F-statistic: {analysis['f_statistic']:.3f}\n")
            f.write(f"F-test p-value: {analysis['f_p_value']:.6f}")
            f.write(" ***" if analysis['f_p_value'] < 0.001 else 
                    " **" if analysis['f_p_value'] < 0.01 else 
                    " *" if analysis['f_p_value'] < 0.05 else 
                    " (n.s.)")
            
            # Add interpretation summary
            f.write("\n\nKey Findings:\n")
            f.write("------------\n")
            # Correlation interpretation
            f.write("1. Relationship strength: ")
            if abs(analysis['correlation']) > 0.5:
                f.write("Strong ")
            elif abs(analysis['correlation']) > 0.3:
                f.write("Moderate ")
            else:
                f.write("Weak ")
            f.write("correlation between variables")
            f.write(" (statistically significant)" if analysis['p_value'] < 0.05 else " (not statistically significant)")
            f.write("\n")
            
            # Effect size interpretation
            f.write(f"2. Effect size: {analysis['r_squared']*100:.1f}% of the variation in {analysis['y_name']}\n")
            f.write(f"   can be explained by {analysis['x_name']}\n")
            
            # Direction interpretation
            f.write("3. Direction: ")
            if analysis['slope'] > 0:
                f.write(f"As {analysis['x_name']} increases, {analysis['y_name']} tends to increase\n")
            elif analysis['slope'] < 0:
                f.write(f"As {analysis['x_name']} increases, {analysis['y_name']} tends to decrease\n")
            else:
                f.write("No clear directional relationship\n")
            
            f.write("\n" + "="*50 + "\n")
        
        # Write regression effects
        f.write("\nWORD REGRESSION EFFECTS\n")
        f.write("----------------------\n")
        for analysis in regression_effects:
            f.write(f"\n{analysis['x_name']} Effect on Word Regression\n")
            f.write("----------------------------------------\n")
            f.write(f"Number of observations: {analysis['n_observations']}\n")
            
            f.write("\nCorrelation Analysis:\n")
            f.write(f"Pearson r: {analysis['correlation']:.3f}")
            f.write(" (Strong)" if abs(analysis['correlation']) > 0.5 else 
                    " (Moderate)" if abs(analysis['correlation']) > 0.3 else 
                    " (Weak)")
            f.write("\n")
            f.write(f"Correlation p-value: {analysis['p_value']:.6f}")
            f.write(" ***" if analysis['p_value'] < 0.001 else 
                    " **" if analysis['p_value'] < 0.01 else 
                    " *" if analysis['p_value'] < 0.05 else 
                    " (n.s.)")
            f.write("\n")
            
            f.write("\nRegression Analysis:\n")
            f.write(f"R-squared: {analysis['r_squared']:.3f}")
            f.write(f" ({analysis['r_squared']*100:.1f}% of variance explained)\n")
            f.write(f"Adjusted R-squared: {analysis['adj_r_squared']:.3f}\n")
            
            f.write(f"Slope: {analysis['slope']:.3f}")
            f.write(" (Positive relationship)" if analysis['slope'] > 0 else 
                    " (Negative relationship)" if analysis['slope'] < 0 else 
                    " (No relationship)")
            f.write("\n")
            f.write(f"Intercept: {analysis['intercept']:.3f}")
            f.write(f" (Predicted {analysis['y_name']} when {analysis['x_name']} = 0)\n")
            
            f.write(f"Slope p-value: {analysis['slope_p_value']:.6f}")
            f.write(" ***" if analysis['slope_p_value'] < 0.001 else 
                    " **" if analysis['slope_p_value'] < 0.01 else 
                    " *" if analysis['slope_p_value'] < 0.05 else 
                    " (n.s.)")
            f.write("\n")
            
            f.write(f"F-statistic: {analysis['f_statistic']:.3f}\n")
            f.write(f"F-test p-value: {analysis['f_p_value']:.6f}")
            f.write(" ***" if analysis['f_p_value'] < 0.001 else 
                    " **" if analysis['f_p_value'] < 0.01 else 
                    " *" if analysis['f_p_value'] < 0.05 else 
                    " (n.s.)")
            
            # Add interpretation summary
            f.write("\n\nKey Findings:\n")
            f.write("------------\n")
            # Correlation interpretation
            f.write("1. Relationship strength: ")
            if abs(analysis['correlation']) > 0.5:
                f.write("Strong ")
            elif abs(analysis['correlation']) > 0.3:
                f.write("Moderate ")
            else:
                f.write("Weak ")
            f.write("correlation between variables")
            f.write(" (statistically significant)" if analysis['p_value'] < 0.05 else " (not statistically significant)")
            f.write("\n")
            
            # Effect size interpretation
            f.write(f"2. Effect size: {analysis['r_squared']*100:.1f}% of the variation in {analysis['y_name']}\n")
            f.write(f"   can be explained by {analysis['x_name']}\n")
            
            # Direction interpretation
            f.write("3. Direction: ")
            if analysis['slope'] > 0:
                f.write(f"As {analysis['x_name']} increases, {analysis['y_name']} tends to increase\n")
            elif analysis['slope'] < 0:
                f.write(f"As {analysis['x_name']} increases, {analysis['y_name']} tends to decrease\n")
            else:
                f.write("No clear directional relationship\n")
            
            f.write("\n" + "="*50 + "\n")
        
        f.write("\nSignificance levels:\n")
        f.write("*** p < 0.001 (Strong evidence)\n")
        f.write("** p < 0.01 (Very good evidence)\n")
        f.write("* p < 0.05 (Good evidence)\n")
        f.write("n.s. = not significant (p ≥ 0.05, Insufficient evidence)\n")
    
    return summary_stats

def main():
    # Set up directories
    output_dir = "/home/baiy4/reader-agent-zuco/results/section2/_human_effects_analysis"
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