
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

print("="*80)
print(" "*15 + "HANGMAN ML - OPTIMIZED SOLUTION")
print("="*80)
print()

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

print("SECTION 1: LOADING DATA")
print("-" * 80)

def load_words(filepath):
    """Load and clean words"""
    try:
        with open(filepath, 'r') as f:
            words = [line.strip().upper() for line in f if line.strip()]
        
        cleaned_words = []
        for w in words:
            if re.match(r'^[A-Z]+$', w) and 3 <= len(w) <= 20:
                cleaned_words.append(w)
        
        cleaned_words = list(set(cleaned_words))
        print(f"✓ Loaded {len(cleaned_words)} words from {filepath}")
        return cleaned_words
    except FileNotFoundError:
        print(f"✗ Error: {filepath} not found!")
        return []

corpus_words = load_words('/kaggle/input/hackathonml/cleaned_corpus.txt')
test_words = load_words('/kaggle/input/hackathonml/cleaned_test.txt')

if not corpus_words or not test_words:
    print("\n⚠ Please ensure corpus and test files exist!")
    exit()

# Build word database
word_database = defaultdict(list)
for word in corpus_words:
    word_database[len(word)].append(word)

print(f"✓ Word database: {len(word_database)} length categories\n")

# ============================================================================
# SECTION 2: ENHANCED HMM
# ============================================================================

print("SECTION 2: BUILDING ENHANCED HMM")
print("-" * 80)

class EnhancedHMM:
    def __init__(self, word_database):
        self.word_database = word_database
        self.position_freq = defaultdict(lambda: defaultdict(int))
        self.bigram_freq = defaultdict(lambda: defaultdict(int))
        self.trigram_freq = defaultdict(lambda: defaultdict(int))
        self.fourgram_freq = defaultdict(lambda: defaultdict(int))
        self.overall_freq = defaultdict(int)
        
    def train(self, words):
        print("Training HMM...")
        total_letters = 0
        
        for word in tqdm(words, desc="Processing", leave=False):
            word_len = len(word)
            
            for pos, letter in enumerate(word):
                pos_key = f"{word_len}_{pos}"
                self.position_freq[pos_key][letter] += 1
                self.overall_freq[letter] += 1
                total_letters += 1
                
                if pos > 0:
                    self.bigram_freq[word[pos-1]][letter] += 1
                
                if pos > 1:
                    context = word[pos-2:pos]
                    self.trigram_freq[context][letter] += 1
                
                if pos > 2:
                    context = word[pos-3:pos]
                    self.fourgram_freq[context][letter] += 1
        
        # Normalize
        for letter in self.overall_freq:
            self.overall_freq[letter] /= total_letters
        
        for prev in self.bigram_freq:
            total = sum(self.bigram_freq[prev].values())
            if total > 0:
                for next_letter in self.bigram_freq[prev]:
                    self.bigram_freq[prev][next_letter] /= total
        
        for context in self.trigram_freq:
            total = sum(self.trigram_freq[context].values())
            if total > 0:
                for letter in self.trigram_freq[context]:
                    self.trigram_freq[context][letter] /= total
        
        for context in self.fourgram_freq:
            total = sum(self.fourgram_freq[context].values())
            if total > 0:
                for letter in self.fourgram_freq[context]:
                    self.fourgram_freq[context][letter] /= total
        
        print(f"✓ HMM trained successfully\n")
    
    def filter_words_by_pattern(self, masked_word, guessed_letters):
        word_len = len(masked_word)
        candidates = []
        
        for word in self.word_database[word_len]:
            matches = True
            for i, char in enumerate(masked_word):
                if char != '_' and word[i] != char:
                    matches = False
                    break
            
            if not matches:
                continue
            
            wrong_letters = guessed_letters - set(masked_word.replace('_', ''))
            if set(word) & wrong_letters:
                continue
            
            candidates.append(word)
        
        return candidates
    
    def predict(self, masked_word, guessed_letters):
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        word_len = len(masked_word)
        
        probs = {letter: 0.0 for letter in alphabet if letter not in guessed_letters}
        
        if not probs:
            return {}
        
        # STRATEGY 1: PATTERN MATCHING (STRONGEST)
        candidates = self.filter_words_by_pattern(masked_word, guessed_letters)
        
        if candidates:
            letter_counts = defaultdict(int)
            for word in candidates:
                for i, char in enumerate(masked_word):
                    if char == '_':
                        letter_counts[word[i]] += 1
            
            if letter_counts:
                total = sum(letter_counts.values())
                for letter in probs:
                    if letter in letter_counts:
                        probs[letter] += (letter_counts[letter] / total) * 150.0
        
        # STRATEGY 2: 4-GRAM CONTEXT
        for pos, char in enumerate(masked_word):
            if char == '_' and pos > 2:
                context = masked_word[pos-3:pos]
                if '_' not in context:
                    fourgrams = self.fourgram_freq.get(context, {})
                    for letter in probs:
                        if letter in fourgrams:
                            probs[letter] += fourgrams[letter] * 8.0
        
        # STRATEGY 3: TRIGRAM CONTEXT
        for pos, char in enumerate(masked_word):
            if char == '_' and pos > 1:
                context = masked_word[pos-2:pos]
                if '_' not in context:
                    trigrams = self.trigram_freq.get(context, {})
                    for letter in probs:
                        if letter in trigrams:
                            probs[letter] += trigrams[letter] * 6.0
        
        # STRATEGY 4: BIGRAM CONTEXT
        for pos, char in enumerate(masked_word):
            if char == '_':
                if pos > 0 and masked_word[pos-1] != '_':
                    prev = masked_word[pos-1]
                    bigrams = self.bigram_freq.get(prev, {})
                    for letter in probs:
                        if letter in bigrams:
                            probs[letter] += bigrams[letter] * 4.0
                
                if pos < len(masked_word) - 1 and masked_word[pos+1] != '_':
                    next_char = masked_word[pos+1]
                    for letter in probs:
                        if letter in self.bigram_freq:
                            probs[letter] += self.bigram_freq[letter].get(next_char, 0) * 4.0
        
        # STRATEGY 5: POSITION-SPECIFIC FREQUENCY
        for pos, char in enumerate(masked_word):
            if char == '_':
                pos_key = f"{word_len}_{pos}"
                pos_freqs = self.position_freq.get(pos_key, {})
                
                for letter in probs:
                    if letter in pos_freqs:
                        probs[letter] += pos_freqs[letter] / 500.0
        
        # STRATEGY 6: VOWEL/CONSONANT BALANCE
        vowels = 'AEIOU'
        revealed = [c for c in masked_word if c != '_']
        unknown = masked_word.count('_')
        
        if revealed and unknown > 0:
            vowel_count = sum(1 for c in revealed if c in vowels)
            vowel_ratio = vowel_count / len(revealed)
            
            if vowel_ratio < 0.15:
                for letter in probs:
                    if letter in vowels:
                        probs[letter] *= 3.5
            elif vowel_ratio < 0.25:
                for letter in probs:
                    if letter in vowels:
                        probs[letter] *= 2.5
            elif vowel_ratio < 0.35:
                for letter in probs:
                    if letter in vowels:
                        probs[letter] *= 1.8
            elif vowel_ratio > 0.65:
                for letter in probs:
                    if letter not in vowels:
                        probs[letter] *= 2.0
        
        # STRATEGY 7: OVERALL FREQUENCY
        for letter in probs:
            probs[letter] += self.overall_freq.get(letter, 0.001) * 0.2
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            for letter in probs:
                probs[letter] /= total
        else:
            for letter in probs:
                probs[letter] = 1.0 / len(probs)
        
        return probs

hmm = EnhancedHMM(word_database)
hmm.train(corpus_words)

# ============================================================================
# SECTION 3: HANGMAN ENVIRONMENT
# ============================================================================

class HangmanEnvironment:
    def __init__(self, word, max_wrong=6):
        self.word = word.upper()
        self.word_length = len(word)
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        self.masked_word = '_' * self.word_length
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.done = False
        self.won = False
        self.turn = 0
        return self.get_state()
    
    def get_state(self):
        return {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'lives': self.max_wrong - self.wrong_guesses,
            'word_length': self.word_length,
            'turn': self.turn,
            'completion': (self.word_length - self.masked_word.count('_')) / self.word_length
        }
    
    def step(self, letter):
        letter = letter.upper()
        self.turn += 1
        
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return self.get_state(), -5, self.done, {'repeated': True}
        
        self.guessed_letters.add(letter)
        
        if letter in self.word:
            new_masked = ''
            letters_revealed = 0
            for i, char in enumerate(self.word):
                if char == letter:
                    new_masked += letter
                    letters_revealed += 1
                else:
                    new_masked += self.masked_word[i]
            self.masked_word = new_masked
            
            if '_' not in self.masked_word:
                self.done = True
                self.won = True
                efficiency_bonus = max(0, (26 - len(self.guessed_letters)) * 3)
                reward = 200 + efficiency_bonus
            else:
                progress_bonus = letters_revealed * 10
                completion = (self.word_length - self.masked_word.count('_')) / self.word_length
                reward = progress_bonus + (completion * 8)
            
            return self.get_state(), reward, self.done, {'correct': True, 'revealed': letters_revealed}
        else:
            self.wrong_guesses += 1
            
            if self.wrong_guesses >= self.max_wrong:
                self.done = True
                self.won = False
                reward = -150
            else:
                penalty = -12 - (self.wrong_guesses * 3)
                reward = penalty
            
            return self.get_state(), reward, self.done, {'correct': False}
    
    def get_available_actions(self):
        alphabet = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return list(alphabet - self.guessed_letters)

# ============================================================================
# SECTION 4: SIMPLE AGENT
# ============================================================================

print("SECTION 4: CREATING AGENT")
print("-" * 80)

class SimpleAgent:
    def __init__(self, hmm):
        self.hmm = hmm
    
    def predict(self, state, available_actions):
        if not available_actions:
            return None
        
        hmm_probs = self.hmm.predict(state['masked_word'], state['guessed_letters'])
        
        votes = defaultdict(float)
        
        # HMM voting
        candidates = self.hmm.filter_words_by_pattern(state['masked_word'], state['guessed_letters'])
        hmm_weight = 100.0 if len(candidates) > 0 else 50.0
        
        for letter in available_actions:
            if letter in hmm_probs:
                votes[letter] += hmm_probs[letter] * hmm_weight
        
        # Vowel balance
        vowels = 'AEIOU'
        revealed = state['masked_word'].replace('_', '')
        if revealed:
            vowel_count = sum(1 for c in revealed if c in vowels)
            vowel_ratio = vowel_count / len(revealed)
            
            if vowel_ratio < 0.25:
                for letter in available_actions:
                    if letter in vowels:
                        votes[letter] *= 2.0
            elif vowel_ratio > 0.6:
                for letter in available_actions:
                    if letter not in vowels:
                        votes[letter] *= 1.5
        
        # Frequency baseline
        freq_order = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
        for i, letter in enumerate(freq_order):
            if letter in available_actions:
                votes[letter] += (26 - i) * 0.3
        
        if votes:
            best_letter = max(votes.items(), key=lambda x: x[1])[0]
            return best_letter
        
        return available_actions[0] if available_actions else None

agent = SimpleAgent(hmm)
print("✓ Agent created\n")

# ============================================================================
# SECTION 5: EVALUATION
# ============================================================================

print("SECTION 5: EVALUATION")
print("-" * 80)

def evaluate_agent(agent, test_words, max_games=None):
    results = []
    eval_words = test_words if max_games is None else test_words[:max_games]
    
    print(f"Evaluating on {len(eval_words)} games...")
    
    for word in tqdm(eval_words, desc="Evaluating"):
        env = HangmanEnvironment(word)
        state = env.reset()
        
        while not env.done:
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            
            action = agent.predict(state, available_actions)
            if action is None:
                break
            
            state, reward, done, info = env.step(action)
        
        results.append({
            'word': word,
            'length': len(word),
            'won': env.won,
            'wrong_guesses': env.wrong_guesses,
            'repeated_guesses': env.repeated_guesses,
            'total_guesses': env.turn,
        })
    
    return pd.DataFrame(results)

results_df = evaluate_agent(agent, test_words)

# ============================================================================
# SECTION 6: RESULTS
# ============================================================================

total_games = len(results_df)
wins = results_df['won'].sum()
success_rate = wins / total_games if total_games > 0 else 0
total_wrong = results_df['wrong_guesses'].sum()
total_repeated = results_df['repeated_guesses'].sum()
avg_wrong = results_df['wrong_guesses'].mean()
avg_repeated = results_df['repeated_guesses'].mean()

final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)

print("\n" + "="*80)
print(" "*25 + "FINAL RESULTS")
print("="*80)
print(f"\n📊 PERFORMANCE:")
print(f"  Games: {total_games}")
print(f"  Won: {wins} ({success_rate*100:.2f}%)")
print(f"  Lost: {total_games - wins}")
print(f"\n  Avg Wrong: {avg_wrong:.2f}")
print(f"  Avg Repeated: {avg_repeated:.2f}")
print(f"\n🏆 FINAL SCORE: {final_score:.2f}")
print("="*80)

print("\n📈 BY LENGTH:")
length_stats = results_df.groupby('length')['won'].agg(['count', 'sum'])
length_stats.columns = ['Total', 'Won']
length_stats['Win%'] = (length_stats['Won'] / length_stats['Total'] * 100).round(1)
print(length_stats)

# Save
results_df.to_csv('results.csv', index=False)
print("\n✓ Saved to results.csv")

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================

print("\nSECTION 7: CREATING VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax = axes[0, 0]
ax.bar(['Lost', 'Won'], [total_games-wins, wins], color=['#EF476F', '#06D6A0'], width=0.6)
ax.set_title('Win/Loss Distribution', fontweight='bold', fontsize=14)
ax.set_ylabel('Count')
for i, v in enumerate([total_games-wins, wins]):
    ax.text(i, v, str(v), ha='center', fontweight='bold')

ax = axes[0, 1]
length_success = results_df.groupby('length')['won'].mean() * 100
ax.bar(length_success.index, length_success.values, color='#118AB2', alpha=0.8)
ax.axhline(success_rate*100, color='red', linestyle='--', linewidth=2, label=f'Avg: {success_rate*100:.1f}%')
ax.set_title('Success Rate by Length', fontweight='bold', fontsize=14)
ax.set_xlabel('Word Length')
ax.set_ylabel('Success Rate (%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
ax.hist(results_df['wrong_guesses'], bins=range(8), color='#EF476F', edgecolor='black', alpha=0.8)
ax.set_title('Wrong Guesses Distribution', fontweight='bold', fontsize=14)
ax.set_xlabel('Number of Wrong Guesses')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
ax.hist(results_df['total_guesses'], bins=20, color='#06D6A0', edgecolor='black', alpha=0.8)
ax.set_title('Total Guesses Distribution', fontweight='bold', fontsize=14)
ax.set_xlabel('Total Guesses')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha=0.3)

plt.suptitle(f'HANGMAN ML - Final Score: {final_score:.2f} | Win Rate: {success_rate*100:.2f}%', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results.png', dpi=300, bbox_inches='tight')
print("✓ Saved to results.png")
plt.show()

print("\n" + "="*80)
print("✨ ANALYSIS COMPLETE! ✨")
print("="*80)