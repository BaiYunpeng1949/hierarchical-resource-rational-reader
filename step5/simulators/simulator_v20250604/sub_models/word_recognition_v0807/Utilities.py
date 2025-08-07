import random


class ApproximateWordGenerator:
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz", variation_prob=0.3):
        """
        Initialize the word generator with an alphabet and variation probability.
        - `variation_prob`: Controls how often random modifications happen.

        NOTE: this function is only called when letters are sampled.
        """
        self.alphabet = list(alphabet)
        self.variation_prob = variation_prob  # Adjusts randomness level

    def generate_similar_words(
        self,
        sampled_letters,
        original_word,
        top_k=5,
        chunk_pos_fuzziness=2,         # how far from the found position we can move the chunk
        chunk_skip_prob=0.1,           # sometimes skip placing the chunk entirely
        insertion_prob=0.1,            # probability of random insert
        deletion_prob=0.1,             # probability of random delete
        substitution_prob=0.2,         # probability of substituting a letter
        keep_original_letter_prob=0.75, # probability of using the original letter instead of a random one
        min_length=3,
        max_length=15,
    ):
        """
        Generate exactly `top_k` words that are loosely similar to the `original_word`.
        
        Relaxations introduced:
        - chunk_pos_fuzziness: how many positions away from the exact location we can place chunk
        - chunk_skip_prob: how often we skip placing a chunk altogether
        - insertion_prob, deletion_prob: how likely we insert/delete letters
        - substitution_prob: chance of randomly substituting a letter
        - keep_original_letter_prob: chance of keeping the original letter in that position
        - min_length, max_length: bounding the length of the word
        """
        word_length = len(original_word)
        generated_words = set()
        generated_words.add(original_word)  # always include the original word
        
        sampled_chunks = sampled_letters.split()  # splits by spaces
        chunk_positions = []
        
        # Identify approximate positions of each chunk
        for chunk in sampled_chunks:
            pos_in_original = original_word.find(chunk)
            
            if pos_in_original == -1:
                # approximate if not found
                pos_in_original = random.randint(0, max(1, word_length - len(chunk)))
            
            chunk_positions.append((chunk, pos_in_original))
        
        # Helper to place chunk into a new_word with possible fuzziness
        def place_chunk_with_fuzziness(new_word, chunk, original_pos, 
                               chunk_pos_fuzziness=2, chunk_skip_prob=0.1):
            # Possibly skip placing the chunk altogether
            if random.random() < chunk_skip_prob:
                return
            
            # Random shift in [-chunk_pos_fuzziness, chunk_pos_fuzziness]
            shift = random.randint(-chunk_pos_fuzziness, chunk_pos_fuzziness)
            pos = original_pos + shift
            
            # Clamp pos to valid range for new_word
            # (pos must be >= 0, but less than or equal to len(new_word))
            pos = max(0, min(pos, len(new_word)))
            
            # Check if chunk can fit. If not, skip it.
            if pos + len(chunk) > len(new_word):
                return
            
            # Place the chunk
            for i, c in enumerate(chunk):
                new_word[pos + i] = c

        # Keep generating until we have at least top_k unique words
        while len(generated_words) < top_k:
            # Start from either the original word or an empty slate
            new_word_length = word_length
            new_word = list(original_word)
            
            # Randomly alter the length: do multiple inserts/deletions
            # We'll do a single pass for possible insertions and deletions
            # and then clamp the final length to [min_length, max_length].
            
            # Deletions
            i = 0
            while i < len(new_word):
                if random.random() < deletion_prob and len(new_word) > min_length:
                    del new_word[i]
                else:
                    i += 1
            
            # Insertions
            i = 0
            while i < len(new_word):
                if random.random() < insertion_prob and len(new_word) < max_length:
                    new_word.insert(i, random.choice(self.alphabet))
                    i += 1  # skip over the inserted character
                i += 1
            
            # If after insertion/deletion, new_word is outside [min_length, max_length], clamp it:
            if len(new_word) < min_length:
                # pad
                new_word += [random.choice(self.alphabet) for _ in range(min_length - len(new_word))]
            elif len(new_word) > max_length:
                # truncate
                new_word = new_word[:max_length]
            
            # Now place chunks (with fuzziness) – do it on a second pass so length is somewhat final
            for (chunk, pos) in chunk_positions:
                place_chunk_with_fuzziness(new_word, chunk, pos)
            
            # Randomly substitute letters
            for i in range(len(new_word)):
                if random.random() < substitution_prob:
                    # either keep original or choose random
                    if random.random() < keep_original_letter_prob and i < len(original_word):
                        new_word[i] = original_word[i]
                    else:
                        new_word[i] = random.choice(self.alphabet)
            
            generated_words.add("".join(new_word))
        
        # If we still don’t have enough words, generate random filler words
        generated_words_list = list(generated_words)
        while len(generated_words_list) < top_k:
            # random word of random length in [min_length, max_length]
            rand_len = random.randint(min_length, max_length)
            filler_word = "".join(random.choices(self.alphabet, k=rand_len))
            if filler_word not in generated_words_list:
                generated_words_list.append(filler_word)
        
        # Return exactly top_k
        return generated_words_list[:top_k]
    
    def generate_a_random_word(self, word_length=8):
        """
        Generate a random word of a given length.
        """
        return "".join(random.choices(self.alphabet, k=word_length))


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein distance between two strings a and b.
    This is the minimal number of single-character edits (insert, delete, substitute)
    needed to transform a into b.
    """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]

    # Initialize boundaries
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # deletion
                dp[i][j-1] + 1,     # insertion
                dp[i-1][j-1] + cost # substitution
            )

    return dp[n][m]