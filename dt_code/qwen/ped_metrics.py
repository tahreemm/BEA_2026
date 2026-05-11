"""
Complete Pedagogical Metrics Computation System
Evaluates Teacher and Judge feedback quality across all dimensions
NO API COSTS - All metrics run locally
"""

import re
import json
import spacy
spacy.load('en_core_web_sm')
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np

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
        
        print("✓ Models loaded successfully")
    
    def count_questions(self, text: str) -> int:
        """Count explicit and implicit questions."""
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
        text_lower = text.lower()
        return sum(text_lower.count(marker) for marker in self.scaffolding_markers.keys())
    
    def count_direct_answers(self, text: str) -> int:
        """Count direct answer-giving statements."""
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
        
        P, R, F1 = self.bert_scorer.score([feedback], [reference])
        return {
            'precision': float(P),
            'recall': float(R),
            'f1': float(F1)
        }
    
    def evaluate_single_feedback(
        self, 
        feedback: str, 
        student_utterance: str,
        reference: Optional[str] = None
    ) -> Dict:
        """
        Compute all metrics for a single feedback instance.
        
        Args:
            feedback: The feedback text to evaluate
            student_utterance: Student's work (for coherence)
            reference: Optional reference feedback for BERTScore
        
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}
        
        # Core pedagogical metrics
        metrics['question_count'] = self.count_questions(feedback)
        metrics['question_ratio'] = self.question_ratio(feedback)
        metrics['hint_count'] = self.count_hints(feedback)
        metrics['direct_count'] = self.count_direct_answers(feedback)
        metrics['scaffolding_ratio'] = self.compute_scaffolding_ratio(feedback)
        metrics['scaffolding_score'] = self.scaffolding_score(feedback)
        metrics['specificity'] = self.specificity_score(feedback)
        
        # Semantic metrics
        metrics['coherence_with_student'] = self.semantic_coherence(
            feedback, student_utterance
        )
        
        # BERTScore (if reference provided)
        if reference and self.use_bertscore:
            bert_scores = self.bertscore_quality(feedback, reference)
            metrics['bertscore_f1'] = bert_scores['f1']
            metrics['bertscore_precision'] = bert_scores['precision']
            metrics['bertscore_recall'] = bert_scores['recall']
        
        # Readability
        metrics['flesch_ease'] = textstat.flesch_reading_ease(feedback)
        
        # Lexical diversity
        try:
            lex = LexicalRichness(feedback)
            metrics['lexical_diversity'] = lex.mtld(threshold=0.72) if len(feedback.split()) > 10 else 0
        except:
            metrics['lexical_diversity'] = 0
        
        # Text properties
        metrics['word_count'] = len(feedback.split())
        doc = self.nlp(feedback)
        metrics['sentence_count'] = len(list(doc.sents))
        
        return metrics


def process_all_instances(
    data_path_ST: str,
    data_path_JUDGE: str,
    data_path_OURS: str,
    output_path: str = 'pedagogical_metrics_results.csv',
    use_bertscore: bool = True,
    device: str = 'cuda'
):
    """
    Process all 490 instances and compute metrics for Teacher and Judge feedback.
    
    Args:
        data_path_ST: Path to JSON file with format:
                   [{"student_response": "...", 
                     "teacher_feedback": "...",
                     "reference_feedback": "..." (optional)}, ...]
        data_path_JUDGE: Path to JSON file with format:
                   [{"student_response": "...", 
                     "judge_feedback": "...",
                     "reference_feedback": "..." (optional)}, ...]
        data_path_OURS: Path to JSON file with format:
                   [{"student_response": "...", 
                     "judge_feedback": "...",
                     "reference_feedback": "..." (optional)}, ...]
        output_path: Path to save results CSV
        use_bertscore: Whether to compute BERTScore (slower)
        device: 'cuda' or 'cpu'
    """
    # Initialize evaluator
    evaluator = CompletePedagogicalMetrics(device=device, use_bertscore=use_bertscore)
    
    # Load data
    print(f"Loading data from {data_path_ST}...")
    print(f"Loading data from {data_path_JUDGE}...")
    def load_jsonl(path):
        records = []
        with open(path, "r", errors="replace") as f:
            for line in f:
                original = line  # keep raw for debugging
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip separator lines like "-----"
                if set(line) == {"-"}:
                    continue

                # Skip metadata like "record: 1"
                if line.lower().startswith("record"):
                    continue

                # Skip anything that is not JSON
                if not line.startswith("{"):
                    # print("Skipping non-JSON line:", repr(original))
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # print("⚠️ Skipping invalid JSON line:", repr(original))
                    print("Error:", e)
                    continue

        print(f"Loaded {len(records)} valid records.")  
        return records


    # Load data from JSONL
    data_ST = load_jsonl(data_path_ST)
    data_JUDGE = load_jsonl(data_path_JUDGE)
    data_OURS = load_jsonl(data_path_OURS)
    print(f"Loaded {len(data_ST)} records from JSONL.")
    print(f"Loaded {len(data_JUDGE)} records from JSONL.")
    print(f"Loaded {len(data_OURS)} records from JSONL.")
    
    print(f"Processing {len(data_ST)} instances...")
    print(f"Processing {len(data_JUDGE)} instances...")
    print(f"Processing {len(data_OURS)} instances...")
    results = []
    
    for idx, (instance_ST, instance_JUDGE, instance_OURS) in enumerate(
    tqdm(zip(data_ST, data_JUDGE, data_OURS), total=min(len(data_ST), len(data_JUDGE), len(data_OURS)), desc="Computing metrics")):
        
        # ---------STUDENT RESPONSE--------- #
        raw_resp = instance_ST.get("student_response", {})
        # Normalize to a dict
        if isinstance(raw_resp, str):
            # It's a JSON string: try to parse
            try:
                student_resp = json.loads(raw_resp)
            except json.JSONDecodeError:
                # It's just plain text or bad JSON
                student_resp = {}
        elif isinstance(raw_resp, dict):
            # Already a dict: use as is
            student_resp = raw_resp
        else:
            # Anything else (None, list, etc.)
            student_resp = {}
        student_utterance = student_resp.get("REASONING", "")
        print("student_utterance: ", student_utterance)
    
        # ---------TEACHER BASELINE 1 RESPONSE--------- #
        raw_resp = instance_ST.get("teacher_response", {})
        # Normalize to a dict
        if isinstance(raw_resp, str):
            # It's a JSON string: try to parse
            try:
                teacher_resp = json.loads(raw_resp)
            except json.JSONDecodeError:
                # It's just plain text or bad JSON
                teacher_resp = {}
        elif isinstance(raw_resp, dict):
            # Already a dict: use as is
            teacher_resp = raw_resp
        else:
            # Anything else (None, list, etc.)
            teacher_resp = {}
        teacher_feedback = teacher_resp.get("TEACHER_FEEDBACK", "")
        print("teacher_feedback: ", teacher_feedback)
        
        # ---------JUDGE BASELINE 2 RESPONSE--------- #
        raw_resp = instance_JUDGE.get("judge_response", {})
        # print("raw_resp: ", raw_resp)
        # Normalize to a dict
        if isinstance(raw_resp, str):
            # It's a JSON string: try to parse
            try:
                judge_resp = json.loads(raw_resp)
            except json.JSONDecodeError:
                # It's just plain text or bad JSON
                judge_resp = {}
        elif isinstance(raw_resp, dict):
            # Already a dict: use as is
            judge_resp = raw_resp
        else:
            # Anything else (None, list, etc.)
            judge_resp = {}
        judge_feedback = judge_resp.get("JUDGE_FEEDBACK", "")
        print("judge_feedback: ", judge_feedback)
        
        KG_correct_steps = instance_JUDGE.get("KG_correct_steps", [])
        
        
         # ---------JUDGE OURS RESPONSE--------- #
        raw_resp = instance_OURS.get("judge_response", {})
        # print("raw_resp: ", raw_resp)
        # Normalize to a dict
        if isinstance(raw_resp, str):
            # It's a JSON string: try to parse
            try:
                ours_resp = json.loads(raw_resp)
            except json.JSONDecodeError:
                # It's just plain text or bad JSON
                ours_resp = {}
        elif isinstance(raw_resp, dict):
            # Already a dict: use as is
            ours_resp = raw_resp
        else:
            # Anything else (None, list, etc.)
            ours_resp = {}
        ours_feedback = ours_resp.get("FINAL_FEEDBACK", "")
        print("ours_feedback: ", ours_feedback)

# Normalize: if it's not a list, wrap or fallback
        if isinstance(KG_correct_steps, str):
            KG_correct_steps = [KG_correct_steps]
        elif KG_correct_steps is None:
            KG_correct_steps = []

        if not KG_correct_steps:
            reference = None
        else:
            reference = KG_correct_steps[0]
            print("reference: ", reference)
        
        
        # Evaluate Teacher feedback
        teacher_metrics = evaluator.evaluate_single_feedback(
            teacher_feedback,
            student_utterance,
            reference
        )
        
        # Evaluate Judge feedback
        judge_metrics = evaluator.evaluate_single_feedback(
            judge_feedback,
            student_utterance,
            reference
        )
        
        # Evaluate Ours feedback
        ours_metrics = evaluator.evaluate_single_feedback(
            ours_feedback,
            student_utterance,
            reference
        )
        
        # Store results
        result = {
            'problem_id': instance_ST.get('problem_id', idx),
            
            # Teacher metrics
            'teacher_question_count': teacher_metrics['question_count'],
            'teacher_question_ratio': teacher_metrics['question_ratio'],
            'teacher_hint_count': teacher_metrics['hint_count'],
            'teacher_direct_count': teacher_metrics['direct_count'],
            'teacher_scaffolding_ratio': teacher_metrics['scaffolding_ratio'],
            'teacher_scaffolding_score': teacher_metrics['scaffolding_score'],
            'teacher_specificity': teacher_metrics['specificity'],
            'teacher_coherence': teacher_metrics['coherence_with_student'],
            'teacher_flesch_ease': teacher_metrics['flesch_ease'],
            'teacher_word_count': teacher_metrics['word_count'],
            
            # Judge metrics
            'judge_question_count': judge_metrics['question_count'],
            'judge_question_ratio': judge_metrics['question_ratio'],
            'judge_hint_count': judge_metrics['hint_count'],
            'judge_direct_count': judge_metrics['direct_count'],
            'judge_scaffolding_ratio': judge_metrics['scaffolding_ratio'],
            'judge_scaffolding_score': judge_metrics['scaffolding_score'],
            'judge_specificity': judge_metrics['specificity'],
            'judge_coherence': judge_metrics['coherence_with_student'],
            'judge_flesch_ease': judge_metrics['flesch_ease'],
            'judge_word_count': judge_metrics['word_count'],
            
            # Ours metrics
            'ours_question_count': ours_metrics['question_count'],
            'ours_question_ratio': ours_metrics['question_ratio'],
            'ours_hint_count': ours_metrics['hint_count'],
            'ours_direct_count': ours_metrics['direct_count'],
            'ours_scaffolding_ratio': ours_metrics['scaffolding_ratio'],
            'ours_scaffolding_score': ours_metrics['scaffolding_score'],
            'ours_specificity': ours_metrics['specificity'],
            'ours_coherence': ours_metrics['coherence_with_student'],
            'ours_flesch_ease': ours_metrics['flesch_ease'],
            'ours_word_count': ours_metrics['word_count'],
        }

        # Add BERTScore if computed
        if use_bertscore and reference:
            result['teacher_bertscore_f1'] = teacher_metrics.get('bertscore_f1')
            result['judge_bertscore_f1'] = judge_metrics.get('bertscore_f1')
            result['ours_bertscore_f1'] = ours_metrics.get('bertscore_f1')
        
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Compute aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    
    print("\n--- TEACHER FEEDBACK ---")
    print(f"Average Scaffolding Ratio: {df['teacher_scaffolding_ratio'].mean():.3f}")
    print(f"Average Questions per Feedback: {df['teacher_question_count'].mean():.2f}")
    print(f"Average Hints per Feedback: {df['teacher_hint_count'].mean():.2f}")
    print(f"Average Direct Statements: {df['teacher_direct_count'].mean():.2f}")
    print(f"Average Specificity Score: {df['teacher_specificity'].mean():.3f}")
    print(f"Average Word Count: {df['teacher_word_count'].mean():.1f}")
    print(f"Average BERTScore F1: {df['teacher_bertscore_f1'].mean():.3f}")
    print(f"flesch_ease: {df['teacher_flesch_ease'].mean():.3f}")
    print(f"coherence_with_student: {df['teacher_coherence'].mean():.3f}")
    
    print("\n--- JUDGE FEEDBACK ---")
    print(f"Average Scaffolding Ratio: {df['judge_scaffolding_ratio'].mean():.3f}")
    print(f"Average Questions per Feedback: {df['judge_question_count'].mean():.2f}")
    print(f"Average Hints per Feedback: {df['judge_hint_count'].mean():.2f}")
    print(f"Average Direct Statements: {df['judge_direct_count'].mean():.2f}")
    print(f"Average Specificity Score: {df['judge_specificity'].mean():.3f}")
    print(f"Average Word Count: {df['judge_word_count'].mean():.1f}")
    print(f"Average BERTScore F1: {df['judge_bertscore_f1'].mean():.3f}")
    print(f"flesch_ease: {df['judge_flesch_ease'].mean():.3f}")
    print(f"coherence_with_student: {df['judge_coherence'].mean():.3f}")
    
    print("\n--- OURS FEEDBACK ---")
    print(f"Average Scaffolding Ratio: {df['ours_scaffolding_ratio'].mean():.3f}")
    print(f"Average Questions per Feedback: {df['ours_question_count'].mean():.2f}")
    print(f"Average Hints per Feedback: {df['ours_hint_count'].mean():.2f}")
    print(f"Average Direct Statements: {df['ours_direct_count'].mean():.2f}")
    print(f"Average Specificity Score: {df['ours_specificity'].mean():.3f}")
    print(f"Average Word Count: {df['ours_word_count'].mean():.1f}")
    print(f"Average BERTScore F1: {df['ours_bertscore_f1'].mean():.3f}")
    print(f"flesch_ease: {df['ours_flesch_ease'].mean():.3f}")
    print(f"coherence_with_student: {df['ours_coherence'].mean():.3f}")
    
    print("\n--- COMPARISON (Judge vs. Teacher) ---")
    scaff_diff = df['judge_scaffolding_ratio'].mean() - df['teacher_scaffolding_ratio'].mean()
    q_diff = df['judge_question_count'].mean() - df['teacher_question_count'].mean()
    dir_diff = df['judge_direct_count'].mean() - df['teacher_direct_count'].mean()
    bertscore_diff = df['judge_bertscore_f1'].mean() - df['teacher_bertscore_f1'].mean()
    flesch_ease_diff = df['judge_flesch_ease'].mean() - df['teacher_flesch_ease'].mean()
    coherence_with_student_diff = df['judge_coherence'].mean() - df['teacher_coherence'].mean()
    print(f"Scaffolding Ratio Difference: {scaff_diff:+.3f} ({'Judge better' if scaff_diff > 0 else 'Teacher better'})")
    print(f"Question Count Difference: {q_diff:+.2f} ({'Judge more' if q_diff > 0 else 'Teacher more'})")
    print(f"Direct Statements Difference: {dir_diff:+.2f} ({'Judge more' if dir_diff > 0 else 'Teacher less'})")
    print(f"BERTScore F1 Difference: {bertscore_diff:+.3f} ({'Judge better' if bertscore_diff > 0 else 'Teacher better'})")
    print(f"Flesch Ease Difference: {flesch_ease_diff:+.3f} ({'Judge better' if flesch_ease_diff > 0 else 'Teacher better'})")
    print(f"Coherence with Student Difference: {coherence_with_student_diff:+.3f} ({'Judge better' if coherence_with_student_diff > 0 else 'Teacher better'})")
    
    print("\n--- COMPARISON (Ours vs. Judge) ---")
    scaff_diff = df['ours_scaffolding_ratio'].mean() - df['judge_scaffolding_ratio'].mean()
    q_diff = df['ours_question_count'].mean() - df['judge_question_count'].mean()
    dir_diff = df['ours_direct_count'].mean() - df['judge_direct_count'].mean()
    bertscore_diff = df['ours_bertscore_f1'].mean() - df['judge_bertscore_f1'].mean()
    flesch_ease_diff = df['ours_flesch_ease'].mean() - df['judge_flesch_ease'].mean()
    coherence_with_student_diff = df['ours_coherence'].mean() - df['judge_coherence'].mean()
    print(f"Scaffolding Ratio Difference: {scaff_diff:+.3f} ({'Ours better' if scaff_diff > 0 else 'Judge better'})")
    print(f"Question Count Difference: {q_diff:+.2f} ({'Ours more' if q_diff > 0 else 'Judge more'})")
    print(f"Direct Statements Difference: {dir_diff:+.2f} ({'Ours more' if dir_diff > 0 else 'Judge less'})")
    print(f"BERTScore F1 Difference: {bertscore_diff:+.3f} ({'Ours better' if bertscore_diff > 0 else 'Judge better'})")
    print(f"Flesch Ease Difference: {flesch_ease_diff:+.3f} ({'Ours better' if flesch_ease_diff > 0 else 'Judge better'})")
    print(f"Coherence with Student Difference: {coherence_with_student_diff:+.3f} ({'Ours better' if coherence_with_student_diff > 0 else 'Judge better'})")
    
    
    # Statistical tests between Judge and Teacher
    from scipy import stats
    
    print("\n--- STATISTICAL SIGNIFICANCE ---")
    
    # Paired t-test for scaffolding ratio
    t_scaff, p_scaff = stats.ttest_rel(
        df['judge_scaffolding_ratio'],
        df['teacher_scaffolding_ratio']
    )
    print(f"Scaffolding Ratio: t={t_scaff:.2f}, p={p_scaff:.4f} {'***' if p_scaff < 0.001 else '**' if p_scaff < 0.01 else '*' if p_scaff < 0.05 else 'ns'}")
    
    # Paired t-test for question count
    t_q, p_q = stats.ttest_rel(
        df['judge_question_count'],
        df['teacher_question_count']
    )
    print(f"Question Count: t={t_q:.2f}, p={p_q:.4f} {'***' if p_q < 0.001 else '**' if p_q < 0.01 else '*' if p_q < 0.05 else 'ns'}")
    
    # Paired t-test for direct statements
    t_dir, p_dir = stats.ttest_rel(
        df['judge_direct_count'],
        df['teacher_direct_count']
    )
    print(f"Direct Statements: t={t_dir:.2f}, p={p_dir:.4f} {'***' if p_dir < 0.001 else '**' if p_dir < 0.01 else '*' if p_dir < 0.05 else 'ns'}")
    
    
    # Statistical tests between Judge and Ours
    t_ours, p_ours = stats.ttest_rel(
        df['ours_scaffolding_ratio'],
        df['judge_scaffolding_ratio']
    )
    print(f"Scaffolding Ratio: t={t_ours:.2f}, p={p_ours:.4f} {'***' if p_ours < 0.001 else '**' if p_ours < 0.01 else '*' if p_ours < 0.05 else 'ns'}")
    t_q_ours, p_q_ours = stats.ttest_rel(
        df['ours_question_count'],
        df['judge_question_count']
    )
    print(f"Question Count: t={t_q_ours:.2f}, p={p_q_ours:.4f} {'***' if p_q_ours < 0.001 else '**' if p_q_ours < 0.01 else '*' if p_q_ours < 0.05 else 'ns'}")
    t_dir_ours, p_dir_ours = stats.ttest_rel(
        df['ours_direct_count'],
        df['judge_direct_count']
    )
    print(f"Direct Statements: t={t_dir_ours:.2f}, p={p_dir_ours:.4f} {'***' if p_dir_ours < 0.001 else '**' if p_dir_ours < 0.01 else '*' if p_dir_ours < 0.05 else 'ns'}")
    
    
    # Compute perplexity for all texts (batch processing) between Judge and Teacher and Ours
    print("\n--- COMPUTING PERPLEXITY (may take a few minutes) ---")
    teacher_texts = [
    instance_ST.get("teacher_response", {}).get("TEACHER_FEEDBACK", "")
    for instance_ST in data_ST
    ]

    judge_texts = [
        instance_JUDGE.get("judge_response", {}).get("JUDGE_FEEDBACK", "")
        for instance_JUDGE in data_JUDGE
    ]
    ours_texts = [
        instance_OURS.get("judge_response", {}).get("JUDGE_FEEDBACK", "")
        for instance_OURS in data_OURS
    ]
    teacher_texts = [t for t in teacher_texts if t.strip()]
    judge_texts   = [t for t in judge_texts if t.strip()]
    ours_texts = [t for t in ours_texts if t.strip()]
    
    teacher_perplexity = evaluator.compute_perplexity(teacher_texts)
    judge_perplexity = evaluator.compute_perplexity(judge_texts)
    ours_perplexity = evaluator.compute_perplexity(ours_texts)
    if teacher_perplexity and judge_perplexity and ours_perplexity:
        print(f"Teacher Perplexity: {teacher_perplexity:.2f}")
        print(f"Judge Perplexity: {judge_perplexity:.2f}")
        print(f"Ours Perplexity: {ours_perplexity:.2f}")
    # Save results
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    return df


    # Process all instances
    


if __name__ == "__main__":
    
    process_all_instances(
        data_path_ST='Data/llm_output/gpt_baseline_1.jsonl',
        data_path_JUDGE='Data/llm_output/gpt_baseline_2.jsonl',
        data_path_OURS='Data/llm_output/gpt_ours.jsonl',
        output_path='Data/llm_output/metrics_results.csv',
        use_bertscore=True,  # Set True if you have reference feedback and GPU
        device='cpu'  # Use 'cuda' if available
    )
    