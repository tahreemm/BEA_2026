"""
Complete Pedagogical Metrics Computation System (Merged)
Evaluates Teacher, Judge, and Ours feedback quality across all dimensions
INCLUDES: All original metrics + NEW metrics (Question Diversity, Second-Person, Verb Ratio)
NO API COSTS - All metrics run locally
"""

import re
import json
import spacy
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
import textstat
import statsmodels.stats.multitest as smm
import scipy.stats as stats

# Required installations:
# pip install bert-score sentence-transformers spacy textstat lexicalrichness evaluate torch tqdm pandas
# python -m spacy download en_core_web_sm

from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import textstat
from lexicalrichness import LexicalRichness
import evaluate


class CompletePedagogicalMetrics:
    """
    Complete implementation of all pedagogical quality metrics.
    All methods run locally without API calls.
    """
    
    def __init__(self, device='cuda', use_bertscore=True):
        """
        Initialize all required models.
        
        Args:
            device: 'cuda' or 'cpu'
            use_bertscore: Set False to skip BERTScore (saves memory)
        """
        print("Initializing models...")
        
        # Load spaCy for linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load Sentence-BERT for semantic similarity (fast, lightweight)
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Load BERTScore (slower, more memory, but higher quality)
        self.use_bertscore = use_bertscore
        if use_bertscore:
            self.bert_scorer = BERTScorer(
                model_type='microsoft/deberta-large-mnli',
                device=device
            )
        
        # Load perplexity evaluator
        self.perplexity = evaluate.load("perplexity", module_type="metric")
        
        # Define scaffolding markers with weights
        self.scaffolding_markers = {
            # High-value scaffolding cues
            'notice that': 2.0,
            'think about': 2.0,
            'consider': 2.0,
            'what if': 2.0,
            
            # Questions prompting reasoning
            'why': 1.5,
            'how': 1.5,
            'what': 1.5,
            'which': 1.5,
            'can you': 1.5,
            'could you': 1.5,
            
            # Reasoning prompts
            'because': 1.0,
            'therefore': 1.0,
            'this means': 1.0,
            
            # Elaboration cues
            'for example': 1.0,
            'specifically': 1.0,
            'in other words': 1.0,
            
            # Metacognitive prompts
            'remember': 1.0,
            'recall': 1.0,
            "let's think": 1.0,
        }
        
        # Define direct answer markers (anti-scaffolding)
        self.direct_markers = {
            'the answer is': 3.0,
            'the solution is': 3.0,
            'you should': 2.0,
            'you must': 2.0,
            'you need to': 2.0,
            'apply': 1.5,
            'use': 1.5,
        }
        
        # Define vague vs. specific markers
        self.vague_markers = ['some', 'many', 'often', 'might', 'could', 
                             'perhaps', 'around', 'maybe']
        self.specific_markers = ['specifically', 'exactly', 'for example', 
                                'in particular', 'precisely']
        
        # NEW: Question diversity patterns
        self.question_patterns = {
            'what': r'\bwhat\b',
            'why': r'\bwhy\b',
            'how': r'\bhow\b',
            'which': r'\bwhich\b',
            'consider': r'\bconsider\b',
            'think': r'\bthink\b',
            'notice': r'\bnotice\b',
            'remember': r'\bremember\b',
            'can you': r'\bcan you\b',
            'could you': r'\bcould you\b',
        }
        
        # NEW: Second-person pronouns
        self.second_person = ['you', 'your', 'yourself', "you're", "you've", "you'll"]
        
        print("✓ Models loaded successfully")
    
    # ==================== ORIGINAL METRICS ====================
    
    def count_questions(self, text: str) -> int:
        """Count explicit and implicit questions."""
        if not text or not text.strip():
            return 0
        # Explicit questions (with ?)
        explicit = text.count('?')
        
        # Implicit questions (question words at sentence start)
        doc = self.nlp(text)
        implicit = 0
        for sent in doc.sents:
            if sent and sent[0].tag_ in ['WDT', 'WP', 'WRB', 'WP$']:
                implicit += 1
        
        return explicit + implicit
    
    def question_ratio(self, text: str) -> float:
        """Compute ratio of questions to total sentences."""
        if not text or not text.strip():
            return 0.0
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if len(sentences) == 0:
            return 0.0
        
        questions = sum(1 for sent in sentences 
                       if sent.text.strip().endswith('?') 
                       or (sent and sent[0].tag_ in ['WDT','WP','WRB']))
        
        return questions / len(sentences)
    
    def scaffolding_score(self, text: str) -> float:
        """Compute scaffolding score from discourse markers."""
        if not text or not text.strip():
            return 0.0
        text_lower = text.lower()
        
        # Count scaffolding markers
        scaffolding = sum(
            text_lower.count(marker) * weight 
            for marker, weight in self.scaffolding_markers.items()
        )
        
        # Normalize by word count
        word_count = len(text.split())
        return scaffolding / max(word_count, 1)
    
    def count_hints(self, text: str) -> int:
        """Count total hint markers."""
        if not text or not text.strip():
            return 0
        text_lower = text.lower()
        return sum(text_lower.count(marker) for marker in self.scaffolding_markers.keys())
    
    def count_direct_answers(self, text: str) -> int:
        """Count direct answer-giving statements."""
        if not text or not text.strip():
            return 0
        text_lower = text.lower()
        return sum(text_lower.count(marker) for marker in self.direct_markers.keys())
    
    def compute_scaffolding_ratio(self, text: str) -> float:
        """
        Scaffolding ratio = hints / (hints + direct answers)
        Range: 0.0 (all direct) to 1.0 (all scaffolding)
        """
        hints = self.count_hints(text)
        direct = self.count_direct_answers(text)
        
        if hints + direct == 0:
            return 0.5  # Neutral if neither detected
        
        return hints / (hints + direct)
    
    def specificity_score(self, text: str) -> float:
        """Measure feedback specificity vs. vagueness."""
        if not text or not text.strip():
            return 0.0
        doc = self.nlp(text)
        text_lower = text.lower()
        
        # Count vague markers
        vague = sum(1 for word in text_lower.split() if word in self.vague_markers)
        
        # Count specific markers
        specific = sum(1 for marker in self.specific_markers if marker in text_lower)
        
        # Count numbers (indicate specificity)
        numbers = sum(1 for token in doc if token.like_num)
        
        # Specificity score: (specific + numbers) / (vague + 1)
        return (specific + numbers) / (vague + 1)
    
    def compute_perplexity(self, texts: List[str]) -> float:
        """
        Compute GPT-2 perplexity (lower = more natural).
        
        Interpretation:
        - 20-80: Natural educational discourse
        - <20: Potentially formulaic
        - >150: Potentially garbled
        """
        # Filter empty/short texts
        texts = [t for t in texts if t and len(t.split()) > 5]
        if not texts:
            return None
        try:
            results = self.perplexity.compute(
                model_id='gpt2',
                predictions=texts,
                batch_size=8
            )
            return results['mean_perplexity']
        except Exception as e:
            print(f"Warning: Perplexity computation failed: {e}")
            return None
    
    def semantic_coherence(self, feedback: str, student_utterance: str) -> float:
        """
        Compute semantic coherence between feedback and student's work.
        Uses fast SBERT embeddings.
        """
        if not feedback or not feedback.strip() or not student_utterance or not student_utterance.strip():
            return 0.0
        emb_feedback = self.sbert.encode(feedback, convert_to_tensor=True)
        emb_student = self.sbert.encode(student_utterance, convert_to_tensor=True)
        
        similarity = util.cos_sim(emb_feedback, emb_student)
        return float(similarity)
    
    def bertscore_quality(self, feedback: str, reference: str) -> Dict[str, float]:
        """
        Compute BERTScore comparing feedback to reference.
        Returns precision, recall, F1.
        """
        if not self.use_bertscore:
            return {'precision': None, 'recall': None, 'f1': None}
        
        if not feedback or not feedback.strip() or not reference or not reference.strip():
            return {'precision': None, 'recall': None, 'f1': None}
        
        P, R, F1 = self.bert_scorer.score([feedback], [reference])
        return {
            'precision': float(P),
            'recall': float(R),
            'f1': float(F1)
        }
    
    def informativeness_index(self, feedback: str, correct_answer: str) -> float:
        """
        Compute Informativeness Index I² (Liermann et al., EMNLP 2024).
        
        Measures whether feedback guides discovery WITHOUT revealing the answer.
        I² = 1 - cosine_similarity(feedback, correct_answer)
        
        Interpretation:
        - Higher I² (closer to 1.0) = Better scaffolding (doesn't reveal answer)
        - Lower I² (closer to 0.0) = Poor scaffolding (reveals answer)
        
        Args:
            feedback: The generated feedback text
            correct_answer: The correct answer/solution that should NOT be revealed
        
        Returns:
            Float between 0 and 1 (higher = better pedagogical quality)
        """
        if not feedback or not feedback.strip() or not correct_answer or not correct_answer.strip():
            return 0.5  # Neutral if either is empty
        
        emb_feedback = self.sbert.encode(feedback, convert_to_tensor=True)
        emb_answer = self.sbert.encode(correct_answer, convert_to_tensor=True)
        
        similarity = float(util.cos_sim(emb_feedback, emb_answer))
        
        # I² = 1 - similarity (higher means feedback doesn't reveal answer)
        return 1.0 - similarity
    
    def lexical_diversity_mtld(self, text: str) -> float:
        """
        MTLD (Measure of Textual Lexical Diversity).
        Higher = more diverse vocabulary.
        """
        if not text or not text.strip():
            return 0.0
        try:
            if len(text.split()) < 10:
                return 0.0
            lex = LexicalRichness(text)
            return lex.mtld(threshold=0.72)
        except:
            return 0.0
    
    # ==================== NEW METRICS ====================
    
    def question_diversity(self, text: str) -> float:
        """
        Count variety of question/prompt types used.
        Range: 0.0 to 1.0 (higher = more diverse questioning)
        """
        if not text or not text.strip():
            return 0.0
        text_lower = text.lower()
        types_used = sum(1 for pattern in self.question_patterns.values() 
                        if re.search(pattern, text_lower))
        return types_used / len(self.question_patterns)
    
    def question_types_used(self, text: str) -> List[str]:
        """Return list of question types found in text."""
        if not text or not text.strip():
            return []
        text_lower = text.lower()
        return [qtype for qtype, pattern in self.question_patterns.items() 
                if re.search(pattern, text_lower)]
    
    def second_person_ratio(self, text: str) -> float:
        """
        Ratio of second-person pronouns (student-directed language).
        Higher = more student-focused feedback.
        """
        if not text or not text.strip():
            return 0.0
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        count = sum(1 for w in words if w in self.second_person)
        return count / len(words)
    
    def second_person_count(self, text: str) -> int:
        """Count of second-person pronouns."""
        if not text or not text.strip():
            return 0
        words = text.lower().split()
        return sum(1 for w in words if w in self.second_person)
    
    def verb_count(self, text: str) -> int:
        """Count verbs (action orientation)."""
        if not text or not text.strip():
            return 0
        doc = self.nlp(text)
        return sum(1 for token in doc if token.pos_ == 'VERB')
    
    def verb_ratio(self, text: str) -> float:
        """Ratio of verbs to total words."""
        if not text or not text.strip():
            return 0.0
        doc = self.nlp(text)
        words = [token for token in doc if token.is_alpha]
        if len(words) == 0:
            return 0.0
        verbs = sum(1 for token in doc if token.pos_ == 'VERB')
        return verbs / len(words)
    
    # ==================== COMPLETE EVALUATION ====================
    
    def evaluate_single_feedback(
        self, 
        feedback: str, 
        student_utterance: str,
        reference: Optional[str] = None,
        correct_answer: Optional[str] = None
    ) -> Dict:
        """
        Compute all metrics for a single feedback instance.
        
        Args:
            feedback: The feedback text to evaluate
            student_utterance: Student's work (for coherence)
            reference: Optional reference feedback for BERTScore
            correct_answer: Optional correct answer for Informativeness Index
        
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}
        
        # Ensure feedback is string
        if not isinstance(feedback, str):
            feedback = str(feedback) if feedback else ""
        
        # ==================== ORIGINAL METRICS ====================
        # Core pedagogical metrics
        metrics['question_count'] = self.count_questions(feedback)
        metrics['question_ratio'] = self.question_ratio(feedback)
        metrics['hint_count'] = self.count_hints(feedback)
        metrics['direct_count'] = self.count_direct_answers(feedback)
        metrics['scaffolding_ratio'] = self.compute_scaffolding_ratio(feedback)
        metrics['scaffolding_score'] = self.scaffolding_score(feedback)
        metrics['specificity'] = self.specificity_score(feedback)
        
        # Informativeness Index
        if correct_answer:
            metrics['informativeness'] = self.informativeness_index(feedback, correct_answer)
        else:
            metrics['informativeness'] = None
        
        # Semantic metrics
        metrics['coherence'] = self.semantic_coherence(feedback, student_utterance)
        
        # BERTScore (if reference provided)
        if reference and self.use_bertscore:
            bert_scores = self.bertscore_quality(feedback, reference)
            metrics['bertscore_f1'] = bert_scores['f1']
            metrics['bertscore_precision'] = bert_scores['precision']
            metrics['bertscore_recall'] = bert_scores['recall']
        else:
            metrics['bertscore_f1'] = None
            metrics['bertscore_precision'] = None
            metrics['bertscore_recall'] = None
        
        # Readability
        if feedback and feedback.strip():
            metrics['flesch_ease'] = textstat.flesch_reading_ease(feedback)
        else:
            metrics['flesch_ease'] = 0
        
        # Lexical diversity (MTLD)
        metrics['lexical_diversity'] = self.lexical_diversity_mtld(feedback)
        
        # Text properties
        metrics['word_count'] = len(feedback.split()) if feedback else 0
        if feedback and feedback.strip():
            doc = self.nlp(feedback)
            metrics['sentence_count'] = len(list(doc.sents))
        else:
            metrics['sentence_count'] = 0
        
        # ==================== NEW METRICS ====================
        # Question diversity
        metrics['question_diversity'] = self.question_diversity(feedback)
        metrics['question_types'] = self.question_types_used(feedback)
        
        # Second-person (student-directed)
        metrics['second_person_count'] = self.second_person_count(feedback)
        metrics['second_person_ratio'] = self.second_person_ratio(feedback)
        
        # Verb count (action orientation)
        metrics['verb_count'] = self.verb_count(feedback)
        metrics['verb_ratio'] = self.verb_ratio(feedback)
        
        return metrics


# ==================== STATISTICAL ANALYSIS FUNCTIONS ====================

def compute_cliffs_delta(x: List[float], y: List[float]) -> tuple:
    """
    Compute Cliff's delta effect size with 95% confidence interval.
    
    Cliff's delta is a non-parametric effect size measure.
    
    Interpretation thresholds:
    - |δ| < 0.147: Negligible
    - |δ| < 0.33: Small
    - |δ| < 0.474: Medium
    - |δ| >= 0.474: Large
    
    Returns:
        (delta, ci_lower, ci_upper)
    """
    n1, n2 = len(x), len(y)
    
    if n1 == 0 or n2 == 0:
        return 0, 0, 0
    
    # Count dominance
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    
    delta = (more - less) / (n1 * n2)
    
    # Compute confidence interval using normal approximation
    var_delta = (1 - delta**2) / (n1 * n2 - 1) if (n1 * n2 - 1) > 0 else 0
    se = np.sqrt(var_delta) if var_delta > 0 else 0
    
    # 95% CI
    z = 1.96
    ci_lower = max(-1, delta - z * se)
    ci_upper = min(1, delta + z * se)
    
    return delta, ci_lower, ci_upper


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def run_three_way_statistical_analysis(
    teacher_scores: List[float],
    judge_scores: List[float],
    ours_scores: List[float],
    metric_name: str
) -> Dict:
    """
    Run proper 3-way statistical comparison following ACL standards.
    
    1. Friedman test (omnibus) to check if any differences exist
    2. If significant, Wilcoxon signed-rank pairwise tests
    3. Holm-Bonferroni correction for multiple comparisons
    4. Cliff's delta effect sizes with CIs
    """
    from scipy.stats import friedmanchisquare, wilcoxon
    from statsmodels.stats.multitest import multipletests
    
    results = {'metric': metric_name}
    
    # Convert to numpy arrays
    teacher = np.array(teacher_scores)
    judge = np.array(judge_scores)
    ours = np.array(ours_scores)
    
    # Descriptive statistics
    results['teacher_mean'] = np.mean(teacher)
    results['teacher_std'] = np.std(teacher)
    results['judge_mean'] = np.mean(judge)
    results['judge_std'] = np.std(judge)
    results['ours_mean'] = np.mean(ours)
    results['ours_std'] = np.std(ours)
    
    # Step 1: Friedman test (omnibus)
    try:
        friedman_stat, friedman_p = friedmanchisquare(teacher, judge, ours)
        results['friedman_stat'] = friedman_stat
        results['friedman_p'] = friedman_p
        results['friedman_significant'] = friedman_p < 0.05
    except Exception as e:
        print(f"Warning: Friedman test failed for {metric_name}: {e}")
        results['friedman_stat'] = None
        results['friedman_p'] = None
        results['friedman_significant'] = False
    
    # Step 2: Pairwise Wilcoxon tests
    pairs = [
        ('teacher_vs_judge', teacher, judge),
        ('judge_vs_ours', judge, ours),
        ('teacher_vs_ours', teacher, ours)
    ]
    
    p_values = []
    for pair_name, x, y in pairs:
        try:
            stat, p = wilcoxon(x, y, alternative='two-sided')
            results[f'{pair_name}_wilcoxon_stat'] = stat
            results[f'{pair_name}_p_uncorrected'] = p
            p_values.append(p)
        except Exception as e:
            print(f"Warning: Wilcoxon test failed for {pair_name}: {e}")
            results[f'{pair_name}_wilcoxon_stat'] = None
            results[f'{pair_name}_p_uncorrected'] = 1.0
            p_values.append(1.0)
    
    # Step 3: Holm-Bonferroni correction
    reject, p_corrected, _, _ = multipletests(p_values, method='holm')
    
    for i, (pair_name, _, _) in enumerate(pairs):
        results[f'{pair_name}_p_corrected'] = p_corrected[i]
        results[f'{pair_name}_significant'] = reject[i]
    
    # Step 4: Effect sizes (Cliff's delta)
    for pair_name, x, y in pairs:
        delta, ci_low, ci_high = compute_cliffs_delta(list(x), list(y))
        results[f'{pair_name}_cliffs_delta'] = delta
        results[f'{pair_name}_delta_ci_lower'] = ci_low
        results[f'{pair_name}_delta_ci_upper'] = ci_high
        results[f'{pair_name}_effect_size'] = interpret_cliffs_delta(delta)
    
    return results


def print_statistical_results(results: Dict):
    """Pretty print statistical analysis results."""
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS: {results['metric']}")
    print('='*70)
    
    # Descriptive stats
    print(f"\nDescriptive Statistics:")
    print(f"  Teacher (S→T):     {results['teacher_mean']:.3f} ± {results['teacher_std']:.3f}")
    print(f"  Judge (S→J):       {results['judge_mean']:.3f} ± {results['judge_std']:.3f}")
    print(f"  Ours (S→T→J):      {results['ours_mean']:.3f} ± {results['ours_std']:.3f}")
    
    # Friedman test
    print(f"\nFriedman Test (omnibus):")
    if results['friedman_stat'] is not None:
        sig = "***" if results['friedman_p'] < 0.001 else "**" if results['friedman_p'] < 0.01 else "*" if results['friedman_p'] < 0.05 else "ns"
        print(f"  χ² = {results['friedman_stat']:.2f}, p = {results['friedman_p']:.4f} {sig}")
    
    # Pairwise comparisons
    print(f"\nPairwise Comparisons (Holm-Bonferroni corrected):")
    pair_labels = {
        'teacher_vs_judge': 'S→T vs S→J',
        'judge_vs_ours': 'S→J vs S→T→J',
        'teacher_vs_ours': 'S→T vs S→T→J'
    }
    for pair in ['teacher_vs_judge', 'judge_vs_ours', 'teacher_vs_ours']:
        p_corr = results[f'{pair}_p_corrected']
        delta = results[f'{pair}_cliffs_delta']
        ci_low = results[f'{pair}_delta_ci_lower']
        ci_high = results[f'{pair}_delta_ci_upper']
        effect = results[f'{pair}_effect_size']
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        
        print(f"  {pair_labels[pair]}:")
        print(f"    p = {p_corr:.4f} {sig}")
        print(f"    Cliff's δ = {delta:.3f} [{ci_low:.3f}, {ci_high:.3f}] ({effect})")


# ==================== DATA LOADING ====================

def load_jsonl(path):
    """Load JSONL file with error handling."""
    records = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            original = line
            line = line.strip()
            
            if not line:
                continue
            if set(line) == {"-"}:
                continue
            if line.lower().startswith("record"):
                continue
            if not line.startswith("{"):
                continue
            
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {original[:80]!r}...")
    
    print(f"Loaded {len(records)} valid records from {path}")
    return records


def parse_response(raw_resp, key):
    """Parse response field from various formats."""
    if isinstance(raw_resp, str):
        try:
            resp = json.loads(raw_resp)
        except json.JSONDecodeError:
            resp = {}
    elif isinstance(raw_resp, dict):
        resp = raw_resp
    else:
        resp = {}
    return resp.get(key, "")


# ==================== MAIN PROCESSING ====================

def process_all_instances(
    data_path_ST: str,
    data_path_JUDGE: str,
    data_path_OURS: str,
    output_path: str = 'pedagogical_metrics_results.csv',
    use_bertscore: bool = True,
    device: str = 'cuda'
):
    """
    Process all instances and compute metrics for Teacher, Judge, and Ours feedback.
    """
    # Initialize evaluator
    evaluator = CompletePedagogicalMetrics(device=device, use_bertscore=use_bertscore)
    
    # Load data
    print(f"Loading data...")
    data_ST = load_jsonl(data_path_ST)
    data_JUDGE = load_jsonl(data_path_JUDGE)
    data_OURS = load_jsonl(data_path_OURS)
    
    min_len = min(len(data_ST), len(data_JUDGE), len(data_OURS))
    print(f"Processing {min_len} instances...")
    
    results = []
    
    for idx, (instance_ST, instance_JUDGE, instance_OURS) in enumerate(tqdm(
        zip(data_ST, data_JUDGE, data_OURS), 
        desc="Computing metrics",
        total=min_len
    )):
        # Extract student response
        student_resp_raw = instance_ST.get('student_response', {})
        if isinstance(student_resp_raw, str):
            try:
                student_resp = json.loads(student_resp_raw)
            except:
                student_resp = {}
        else:
            student_resp = student_resp_raw if isinstance(student_resp_raw, dict) else {}
        student_utterance = student_resp.get('REASONING', '') or json.dumps(student_resp_raw)
        
        # Extract Teacher feedback
        teacher_resp_raw = instance_ST.get('teacher_response', {})
        if isinstance(teacher_resp_raw, str):
            try:
                teacher_resp = json.loads(teacher_resp_raw)
            except:
                teacher_resp = {}
        else:
            teacher_resp = teacher_resp_raw if isinstance(teacher_resp_raw, dict) else {}
        teacher_feedback = teacher_resp.get('TEACHER_FEEDBACK', '')
        
        # Extract Judge feedback
        judge_resp_raw = instance_JUDGE.get('judge_response', {})
        if isinstance(judge_resp_raw, str):
            try:
                judge_resp = json.loads(judge_resp_raw)
            except:
                judge_resp = {}
        else:
            judge_resp = judge_resp_raw if isinstance(judge_resp_raw, dict) else {}
        judge_feedback = judge_resp.get('JUDGE_FEEDBACK', '')
        
        # Extract Ours feedback (try FINAL_FEEDBACK first, then JUDGE_FEEDBACK)
        ours_resp_raw = instance_OURS.get('judge_response', {})
        if isinstance(ours_resp_raw, str):
            try:
                ours_resp = json.loads(ours_resp_raw)
            except:
                ours_resp = {}
        else:
            ours_resp = ours_resp_raw if isinstance(ours_resp_raw, dict) else {}
        ours_feedback = ours_resp.get('FINAL_FEEDBACK', '') or ours_resp.get('JUDGE_FEEDBACK', '')
        
        # Extract correct answer for I²
        KG_correct_steps = instance_JUDGE.get('KG_correct_steps', [])
        if isinstance(KG_correct_steps, str):
            KG_correct_steps = [KG_correct_steps]
        correct_answer = KG_correct_steps[0] if KG_correct_steps else None
        reference = correct_answer
        
        # Evaluate all three pipelines
        teacher_metrics = evaluator.evaluate_single_feedback(
            teacher_feedback, student_utterance, reference, correct_answer
        )
        judge_metrics = evaluator.evaluate_single_feedback(
            judge_feedback, student_utterance, reference, correct_answer
        )
        ours_metrics = evaluator.evaluate_single_feedback(
            ours_feedback, student_utterance, reference, correct_answer
        )
        
        # Store ALL metrics
        result = {'problem_id': instance_ST.get('problem_id', idx)}
        
        # All metrics to store
        all_metric_keys = [
            'question_count', 'question_ratio', 'hint_count', 'direct_count',
            'scaffolding_ratio', 'scaffolding_score', 'specificity',
            'informativeness', 'coherence', 'flesch_ease', 'word_count',
            'sentence_count', 'lexical_diversity', 'bertscore_f1',
            'bertscore_precision', 'bertscore_recall',
            # NEW metrics
            'question_diversity', 'second_person_count', 'second_person_ratio',
            'verb_count', 'verb_ratio'
        ]
        
        for metric in all_metric_keys:
            result[f'teacher_{metric}'] = teacher_metrics.get(metric)
            result[f'judge_{metric}'] = judge_metrics.get(metric)
            result[f'ours_{metric}'] = ours_metrics.get(metric)
        
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # ==================== AGGREGATE RESULTS ====================
    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    
    metrics_to_print = [
        ('scaffolding_ratio', 'Scaffolding Ratio'),
        ('question_count', 'Question Count'),
        ('question_ratio', 'Question Ratio'),
        ('question_diversity', 'Question Diversity'),  # NEW
        ('hint_count', 'Hint Count'),
        ('direct_count', 'Direct Statements'),
        ('specificity', 'Specificity Score'),
        ('second_person_ratio', 'Second-Person Ratio'),  # NEW
        ('verb_ratio', 'Verb Ratio'),  # NEW
        ('lexical_diversity', 'Lexical Diversity (MTLD)'),
        ('informativeness', 'Informativeness Index (I²)'),
        ('coherence', 'Coherence with Student'),
        ('flesch_ease', 'Flesch Reading Ease'),
        ('word_count', 'Word Count'),
        ('bertscore_f1', 'BERTScore F1'),
    ]
    
    for pipeline, prefix in [('TEACHER (S→T)', 'teacher'), 
                              ('JUDGE (S→J)', 'judge'), 
                              ('OURS (S→T→J)', 'ours')]:
        print(f"\n--- {pipeline} ---")
        for metric_key, metric_name in metrics_to_print:
            col = f"{prefix}_{metric_key}"
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if pd.notna(mean_val):
                    print(f"{metric_name}: {mean_val:.3f} (±{std_val:.3f})")
    
    # ==================== STATISTICAL ANALYSIS ====================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS (ACL Standards)")
    print("="*80)
    
    metrics_to_analyze = [
        ('scaffolding_ratio', 'Scaffolding Ratio'),
        ('question_count', 'Question Count'),
        ('question_diversity', 'Question Diversity'),  # NEW
        ('informativeness', 'Informativeness Index (I²)'),
        ('coherence', 'Coherence with Student'),
        ('specificity', 'Specificity Score'),
        ('second_person_ratio', 'Second-Person Ratio'),  # NEW
        ('verb_ratio', 'Verb Ratio'),  # NEW
        ('lexical_diversity', 'Lexical Diversity (MTLD)'),
        ('flesch_ease', 'Flesch Reading Ease'),
    ]
    
    all_stats_results = []
    
    for metric_col, metric_name in metrics_to_analyze:
        teacher_col = f'teacher_{metric_col}'
        judge_col = f'judge_{metric_col}'
        ours_col = f'ours_{metric_col}'
        
        if teacher_col not in df.columns:
            continue
        
        # Get scores, handling NaN
        valid_mask = df[teacher_col].notna() & df[judge_col].notna() & df[ours_col].notna()
        teacher_scores = df.loc[valid_mask, teacher_col].tolist()
        judge_scores = df.loc[valid_mask, judge_col].tolist()
        ours_scores = df.loc[valid_mask, ours_col].tolist()
        
        if len(teacher_scores) < 10:
            print(f"Warning: Skipping {metric_name} - insufficient data ({len(teacher_scores)} samples)")
            continue
        
        # Run 3-way statistical analysis
        stats_results = run_three_way_statistical_analysis(
            teacher_scores, judge_scores, ours_scores, metric_name
        )
        
        print_statistical_results(stats_results)
        all_stats_results.append(stats_results)
    
    # ==================== SUMMARY OF SIGNIFICANT DIFFERENCES ====================
    print("\n" + "="*80)
    print("SUMMARY: SIGNIFICANT DIFFERENCES (p < 0.05, Holm-Bonferroni corrected)")
    print("="*80)
    
    comparisons = [
        ('teacher_vs_judge', 'S→T vs S→J'),
        ('judge_vs_ours', 'S→J vs S→T→J'),
        ('teacher_vs_ours', 'S→T vs S→T→J')
    ]
    
    for comp_key, comp_name in comparisons:
        print(f"\n{comp_name}:")
        sig_found = False
        for stats in all_stats_results:
            if stats[f'{comp_key}_significant']:
                delta = stats[f'{comp_key}_cliffs_delta']
                effect = stats[f'{comp_key}_effect_size']
                p = stats[f'{comp_key}_p_corrected']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                
                # Determine winner
                if comp_key == 'teacher_vs_judge':
                    winner = "Teacher" if delta > 0 else "Judge"
                elif comp_key == 'judge_vs_ours':
                    winner = "Judge" if delta > 0 else "Ours"
                else:
                    winner = "Teacher" if delta > 0 else "Ours"
                
                print(f"  • {stats['metric']}: {winner} wins ({sig}, δ={delta:.2f}, {effect})")
                sig_found = True
        
        if not sig_found:
            print("  No significant differences found")
    
    # ==================== PERPLEXITY ====================
    print("\n" + "="*80)
    print("PERPLEXITY ANALYSIS")
    print("="*80)
    
    teacher_texts = [parse_response(inst.get("teacher_response", {}), "TEACHER_FEEDBACK") 
                     for inst in data_ST]
    judge_texts = [parse_response(inst.get("judge_response", {}), "JUDGE_FEEDBACK") 
                   for inst in data_JUDGE]
    ours_texts = [parse_response(inst.get("judge_response", {}), "FINAL_FEEDBACK") or 
                  parse_response(inst.get("judge_response", {}), "JUDGE_FEEDBACK")
                  for inst in data_OURS]
    
    teacher_perplexity = evaluator.compute_perplexity(teacher_texts)
    judge_perplexity = evaluator.compute_perplexity(judge_texts)
    ours_perplexity = evaluator.compute_perplexity(ours_texts)
    
    if teacher_perplexity:
        print(f"Teacher Perplexity: {teacher_perplexity:.2f}")
    if judge_perplexity:
        print(f"Judge Perplexity: {judge_perplexity:.2f}")
    if ours_perplexity:
        print(f"Ours Perplexity: {ours_perplexity:.2f}")
    
    # ==================== SAVE RESULTS ====================
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Save statistical results
    if all_stats_results:
        stats_df = pd.DataFrame(all_stats_results)
        stats_output = output_path.replace('.csv', '_statistics.csv')
        stats_df.to_csv(stats_output, index=False)
        print(f"✓ Statistical results saved to {stats_output}")
    
    return df


if __name__ == "__main__":
    process_all_instances(
        data_path_ST='Data/llm_output/gpt-o3/gpt_baseline_1.jsonl',
        data_path_JUDGE='Data/llm_output/gpt-o3/gpt_baseline_2.jsonl',
        data_path_OURS='Data/llm_output/gpt-o3/gpt_ours.jsonl',
        output_path='Data/llm_output/gpt-o3/metrics_results.csv',
        use_bertscore=True,
        device='cpu'
    )