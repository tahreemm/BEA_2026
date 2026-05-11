"""
Complete Pedagogical Metrics Computation System (Updated)
Evaluates Tutor (S→Tu), Teacher (S→T), and Judge (S→Tu→J) feedback quality
INCLUDES: All original metrics + Leakage Detection + Comprehensive Composite Score
NO API COSTS - All metrics run locally

Pipeline Definitions:
- S→Tu: Student → Tutor (partial hints access)
- S→T: Student → Teacher (complete solution access)
- S→Tu→J: Student → Tutor → Judge (verifier with complete solution)
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
def convert_rule_to_short_name(rule_name: str) -> str:
    """
    Convert full rule name to short name.
    Returns the short name if found, otherwise returns the original rule_name.
    """
    if not rule_name or rule_name == "N/A":
        return rule_name
    
    # Normalize the input: strip whitespace and convert to lowercase for matching
    rule_lower = rule_name.strip().lower()
    
    # Mapping of full rule names to short names
    rule_mapping = {
        'modus ponens': 'MP',
        'modus tollens': 'MT',
        'conjunction': 'Conj',
        'addition': 'Add',
        'disjunctive syllogism': 'DS',
        'hypothetical syllogism': 'HS',
        "de morgan's laws": 'DeM',
        "de morgan's law": 'DeM',
        "de morgan": 'DeM',
        'implication': 'Impl',
        'simplification': 'Simp',
        'distribution': 'Dist',
        'associativity': 'Assoc',
        'contraposition': 'CP',
        'contrapositive': 'CP',
        'contrapositive rule': 'CP',
        'commutation': 'Com',
        'commutativity': 'Com',
        'equivalence': 'Equiv',
        'equivalence elimination': 'Equiv',
        'equivalence elimination rule': 'Equiv',
        'biconditional elimination': 'Equiv',
        'biconditional': 'Equiv',
        'constructive dilemma': 'CD',
        'double negation': 'DN',
        'double negation rule': 'DN',
        'double negation elimination': 'DN',
        'double negation elimination rule': 'DN',
    }
    
    # Try exact match first
    if rule_lower in rule_mapping:
        return rule_mapping[rule_lower]
    
    # Try partial match (e.g., "Modus Ponens rule" -> "modus ponens")
    for full_name, short_name in rule_mapping.items():
        if full_name in rule_lower:
            return short_name
    
    # If already a short name (2-4 characters, uppercase), return as is
    if len(rule_name.strip()) <= 4 and rule_name.strip().isupper():
        return rule_name.strip()
    
    # Return original if no match found
    return rule_name

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
        
        # Question diversity patterns
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
        
        # Second-person pronouns
        self.second_person = ['you', 'your', 'yourself', "you're", "you've", "you'll"]
        
        # Propositional logic rule patterns (exact match for leakage detection)
        self.rule_patterns = {
            'MP': r'\b(MP|Modus\s*Ponens)\b',
            'MT': r'\b(MT|Modus\s*Tollens)\b',
            'Simp': r'\b(Simp|Simplification)\b',
            'Add': r'\b(Add|Addition)\b',
            'DS': r'\b(DS|Disjunctive\s*Syllogism)\b',
            'HS': r'\b(HS|Hypothetical\s*Syllogism)\b',
            'CD': r'\b(CD|Constructive\s*Dilemma)\b',
            'DeM': r'\b(DeM|De\s*Morgan)\b',
            'DN': r'\b(DN|Double\s*Negation)\b',
            'Impl': r'\b(Impl|Implication)\b',
            'CP': r'\b(CP|Contraposition)\b',
            'Com': r'\b(Com|Commutation)\b',
            'Assoc': r'\b(Assoc|Associativity)\b',
            'Dist': r'\b(Dist|Distribution)\b',
            'Equiv': r'\b(Equiv|Equivalence)\b',
            'Conj': r'\b(Conj|Conjunction)\b',
        }
        
        # Logical expression pattern
        self.expression_pattern = r'[\(\)A-Z\-\>\*\+\=\~]+'
        
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
        
        Reference:
        Liermann, W., Huang, J.-X., Lee, Y., & Lee, K.J. (2024). 
        More Insightful Feedback for Tutoring: Enhancing Generation Mechanisms 
        and Automatic Evaluation. EMNLP 2024, pp. 10838–10851.
        
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
    
    # ==================== QUESTION & ENGAGEMENT METRICS ====================
    
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
    
    # ==================== LEAKAGE DETECTION METRICS ====================
    
    def rule_leakage_count(self, text: str) -> int:
        """
        Count number of explicit rule mentions in feedback.
        
        Detects both abbreviations (MP, Simp) and full names (Modus Ponens).
        Higher count = more answer leakage.
        """
        if not text or not text.strip():
            return 0
        count = 0
        for pattern in self.rule_patterns.values():
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def rule_leakage_binary(self, text: str) -> int:
        """
        Binary indicator: 1 if any rule is mentioned, 0 otherwise.
        
        Use for computing leakage rate across dataset.
        """
        return 1 if self.rule_leakage_count(text) > 0 else 0
    
    def rules_leaked(self, text: str) -> List[str]:
        """
        Return list of specific rules mentioned in feedback.
        
        Useful for error analysis and identifying patterns.
        """
        if not text or not text.strip():
            return []
        leaked = []
        for rule_name, pattern in self.rule_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                leaked.append(rule_name)
        return leaked
    
    def expression_overlap(self, feedback: str, kg_step: str, n: int = 2) -> float:
        """
        N-gram overlap between feedback and ground-truth step.
        
        Higher = more leakage of the actual step/expression.
        
        Args:
            feedback: The feedback text
            kg_step: The correct step from knowledge graph (e.g., "(A>B)")
            n: N-gram size (default 2)
        
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not feedback or not kg_step:
            return 0.0
        
        # Tokenize (keep logical operators)
        def tokenize(text):
            # Split on whitespace but keep logical expressions intact
            tokens = re.findall(r'[\w\-\>\*\+\=\~\(\)]+', text)
            return [t.upper() for t in tokens]  # Case-insensitive
        
        fb_tokens = tokenize(feedback)
        kg_tokens = tokenize(kg_step)
        
        if len(kg_tokens) < n:
            # For short expressions, check exact containment
            kg_str = ''.join(kg_tokens)
            fb_str = ''.join(fb_tokens)
            return 1.0 if kg_str in fb_str else 0.0
        
        # N-gram overlap
        def get_ngrams(tokens, n):
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        
        fb_ngrams = get_ngrams(fb_tokens, n)
        kg_ngrams = get_ngrams(kg_tokens, n)
        
        if not kg_ngrams:
            return 0.0
        
        return len(fb_ngrams & kg_ngrams) / len(kg_ngrams)
    
    def expression_containment(self, feedback: str, kg_step: str) -> int:
        """
        Binary: 1 if feedback contains the exact KG step expression, 0 otherwise.
        
        Strict check for direct step revelation.
        """
        if not feedback or not kg_step:
            return 0
        
        # Normalize both (remove whitespace, uppercase)
        fb_norm = re.sub(r'\s+', '', feedback.upper())
        kg_norm = re.sub(r'\s+', '', kg_step.upper())
        
        return 1 if kg_norm in fb_norm else 0
    
    def specific_rule_leaked(self, feedback: str, target_rule: str) -> int:
        """
        Check if a specific target rule is mentioned in feedback.
        
        Args:
            feedback: The feedback text
            target_rule: The rule to check for (e.g., "MP", "Simp")
        
        Returns:
            1 if target rule is mentioned, 0 otherwise
        """
        if not feedback or not target_rule:
            return 0
        
        pattern = self.rule_patterns.get(target_rule, rf'\b{target_rule}\b')
        return 1 if re.search(pattern, feedback, re.IGNORECASE) else 0
    
    def comprehensive_leakage_score(
        self, 
        feedback: str, 
        kg_step: str, 
        kg_rule: str,
        correct_answer: str = None
    ) -> float:
        """
        Comprehensive leakage score combining ALL approaches:
        1. Rule leakage (pattern-based)
        2. Expression overlap (n-gram)
        3. Semantic similarity (embedding-based)
        4. Direct answer markers (keyword-based)
        
        Range: 0.0 (no leakage) to 1.0 (full leakage)
        
        Weights:
        - Rule leakage: 0.35 (most harmful - direct answer)
        - Expression overlap: 0.25 (reveals specific step)
        - Semantic similarity: 0.25 (captures paraphrased leakage)
        - Direct markers: 0.15 (stylistic indicators)
        """
        if not feedback:
            return 0.0
        
        scores = []
        weights = []
        
        # 1. Rule leakage (pattern-based) - weight 0.35
        if kg_rule:
            rule_pattern = self.rule_patterns.get(kg_rule, rf'\b{kg_rule}\b')
            rule_leaked = 1.0 if re.search(rule_pattern, feedback, re.IGNORECASE) else 0.0
        else:
            # Check if ANY rule is leaked
            rule_leaked = min(self.rule_leakage_count(feedback) / 2.0, 1.0)
        scores.append(rule_leaked)
        weights.append(0.35)
        
        # 2. Expression overlap (n-gram) - weight 0.25
        if kg_step:
            expr_overlap = self.expression_overlap(feedback, kg_step)
            scores.append(expr_overlap)
            weights.append(0.25)
        
        # 3. Semantic similarity (embedding-based) - weight 0.25
        # Inverse of informativeness: higher similarity = more leakage
        answer_text = correct_answer or kg_step
        if answer_text:
            informativeness = self.informativeness_index(feedback, answer_text)
            semantic_leakage = 1.0 - informativeness  # Invert: high similarity = high leakage
            scores.append(semantic_leakage)
            weights.append(0.25)
        
        # 4. Direct answer markers - weight 0.15
        direct_count = self.count_direct_answers(feedback)
        word_count = max(len(feedback.split()), 1)
        direct_ratio = min(direct_count / word_count * 10, 1.0)
        scores.append(direct_ratio)
        weights.append(0.15)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight
    
    # ==================== COMPLETE EVALUATION ====================
    
    def evaluate_single_feedback(
        self, 
        feedback: str, 
        student_utterance: str,
        reference: Optional[str] = None,
        correct_answer: Optional[str] = None,
        kg_step: Optional[str] = None,
        kg_rule: Optional[str] = None
    ) -> Dict:
        """
        Compute all metrics for a single feedback instance.
        
        Args:
            feedback: The feedback text to evaluate
            student_utterance: Student's work (for coherence)
            reference: Optional reference feedback for BERTScore
            correct_answer: Optional correct answer for Informativeness Index
            kg_step: Optional correct step for leakage detection
            kg_rule: Optional correct rule for leakage detection
        
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
        
        # Informativeness Index (Liermann et al., EMNLP 2024)
        if correct_answer or kg_step:
            answer_text = correct_answer or kg_step
            metrics['informativeness'] = self.informativeness_index(feedback, answer_text)
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
        
        # ==================== QUESTION & ENGAGEMENT METRICS ====================
        metrics['question_diversity'] = self.question_diversity(feedback)
        metrics['question_types'] = self.question_types_used(feedback)
        
        metrics['second_person_count'] = self.second_person_count(feedback)
        metrics['second_person_ratio'] = self.second_person_ratio(feedback)
        
        metrics['verb_count'] = self.verb_count(feedback)
        metrics['verb_ratio'] = self.verb_ratio(feedback)
        
        # ==================== LEAKAGE DETECTION METRICS ====================
        metrics['rule_leakage_count'] = self.rule_leakage_count(feedback)
        metrics['rule_leakage_binary'] = self.rule_leakage_binary(feedback)
        metrics['rules_leaked'] = self.rules_leaked(feedback)
        
        if kg_step:
            metrics['expression_overlap'] = self.expression_overlap(feedback, kg_step)
            metrics['expression_containment'] = self.expression_containment(feedback, kg_step)
        else:
            metrics['expression_overlap'] = None
            metrics['expression_containment'] = None
        
        if kg_rule:
            metrics['target_rule_leaked'] = self.specific_rule_leaked(feedback, kg_rule)
        else:
            metrics['target_rule_leaked'] = None
        
        # Comprehensive leakage score (combines rule + expression + semantic + direct)
        if kg_step or kg_rule or correct_answer:
            metrics['comprehensive_leakage'] = self.comprehensive_leakage_score(
                feedback, kg_step, kg_rule, correct_answer
            )
        else:
            metrics['comprehensive_leakage'] = None
        
        return metrics


# ==================== STATISTICAL ANALYSIS FUNCTIONS ====================

def compute_cliffs_delta(x: List[float], y: List[float]) -> tuple:
    """
    Compute Cliff's delta effect size with 95% confidence interval.
    
    Interpretation thresholds:
    - |δ| < 0.147: Negligible
    - |δ| < 0.33: Small
    - |δ| < 0.474: Medium
    - |δ| >= 0.474: Large
    """
    n1, n2 = len(x), len(y)
    
    if n1 == 0 or n2 == 0:
        return 0, 0, 0
    
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    
    delta = (more - less) / (n1 * n2)
    
    var_delta = (1 - delta**2) / (n1 * n2 - 1) if (n1 * n2 - 1) > 0 else 0
    se = np.sqrt(var_delta) if var_delta > 0 else 0
    
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
    tutor_scores: List[float],
    teacher_scores: List[float],
    judge_scores: List[float],
    metric_name: str
) -> Dict:
    """
    Run proper 3-way statistical comparison following ACL standards.
    
    Pipelines:
    - Tutor: S→Tu (partial access)
    - Teacher: S→T (complete access)
    - Judge: S→Tu→J (verifier)
    """
    from scipy.stats import friedmanchisquare, wilcoxon
    from statsmodels.stats.multitest import multipletests
    
    results = {'metric': metric_name}
    
    tutor = np.array(tutor_scores)
    teacher = np.array(teacher_scores)
    judge = np.array(judge_scores)
    
    # Descriptive statistics
    results['tutor_mean'] = np.mean(tutor)
    results['tutor_std'] = np.std(tutor)
    results['teacher_mean'] = np.mean(teacher)
    results['teacher_std'] = np.std(teacher)
    results['judge_mean'] = np.mean(judge)
    results['judge_std'] = np.std(judge)
    
    # Step 1: Friedman test (omnibus)
    try:
        friedman_stat, friedman_p = friedmanchisquare(tutor, teacher, judge)
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
        ('tutor_vs_teacher', tutor, teacher),
        ('teacher_vs_judge', teacher, judge),
        ('tutor_vs_judge', tutor, judge)
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
    
    print(f"\nDescriptive Statistics:")
    print(f"  Tutor (S→Tu):      {results['tutor_mean']:.3f} ± {results['tutor_std']:.3f}")
    print(f"  Teacher (S→T):     {results['teacher_mean']:.3f} ± {results['teacher_std']:.3f}")
    print(f"  Judge (S→Tu→J):    {results['judge_mean']:.3f} ± {results['judge_std']:.3f}")
    
    print(f"\nFriedman Test (omnibus):")
    if results['friedman_stat'] is not None:
        sig = "***" if results['friedman_p'] < 0.001 else "**" if results['friedman_p'] < 0.01 else "*" if results['friedman_p'] < 0.05 else "ns"
        print(f"  χ² = {results['friedman_stat']:.2f}, p = {results['friedman_p']:.4f} {sig}")
    
    print(f"\nPairwise Comparisons (Holm-Bonferroni corrected):")
    pair_labels = {
        'tutor_vs_teacher': 'S→Tu vs S→T',
        'teacher_vs_judge': 'S→T vs S→Tu→J',
        'tutor_vs_judge': 'S→Tu vs S→Tu→J'
    }
    for pair in ['tutor_vs_teacher', 'teacher_vs_judge', 'tutor_vs_judge']:
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
    data_path_TUTOR: str,
    data_path_TEACHER: str,
    data_path_JUDGE: str,
    output_path: str = 'pedagogical_metrics_results.csv',
    use_bertscore: bool = True,
    device: str = 'cuda'
):
    """
    Process all instances and compute metrics for all three pipelines.
    
    Args:
        data_path_TUTOR: Path to S→Tu (Tutor) feedback data
        data_path_TEACHER: Path to S→T (Teacher) feedback data
        data_path_JUDGE: Path to S→Tu→J (Judge) feedback data
        output_path: Path for output CSV
        use_bertscore: Whether to compute BERTScore
        device: 'cuda' or 'cpu'
    """
    # Initialize evaluator
    evaluator = CompletePedagogicalMetrics(device=device, use_bertscore=use_bertscore)
    
    # Load data
    print(f"Loading data...")
    data_tutor = load_jsonl(data_path_TUTOR)
    data_teacher = load_jsonl(data_path_TEACHER)
    data_judge = load_jsonl(data_path_JUDGE)
    
    min_len = min(len(data_tutor), len(data_teacher), len(data_judge))
    print(f"Processing {min_len} instances...")
    
    results = []
    
    for idx, (instance_tutor, instance_teacher, instance_judge) in enumerate(tqdm(
        zip(data_tutor, data_teacher, data_judge), 
        desc="Computing metrics",
        total=min_len
    )):
        # Extract student response
        student_resp_raw = instance_tutor.get('student_response', {})
        if isinstance(student_resp_raw, str):
            try:
                student_resp = json.loads(student_resp_raw)
            except:
                student_resp = {}
        else:
            student_resp = student_resp_raw if isinstance(student_resp_raw, dict) else {}
        student_utterance = student_resp.get('REASONING', '') or json.dumps(student_resp_raw)
        
        # Extract Tutor feedback (S→Tu)
        tutor_resp_raw = instance_tutor.get('teacher_response', {})
        if isinstance(tutor_resp_raw, str):
            try:
                tutor_resp = json.loads(tutor_resp_raw)
            except:
                tutor_resp = {}
        else:
            tutor_resp = tutor_resp_raw if isinstance(tutor_resp_raw, dict) else {}
        tutor_feedback = tutor_resp.get('TEACHER_FEEDBACK', '')
        
        # Extract Teacher feedback (S→T)
        teacher_resp_raw = instance_teacher.get('judge_response', {})
        if isinstance(teacher_resp_raw, str):
            try:
                teacher_resp = json.loads(teacher_resp_raw)
            except:
                teacher_resp = {}
        else:
            teacher_resp = teacher_resp_raw if isinstance(teacher_resp_raw, dict) else {}
        teacher_feedback = teacher_resp.get('JUDGE_FEEDBACK', '')
        
        # Extract Judge feedback (S→Tu→J)
        judge_resp_raw = instance_judge.get('judge_response', {})
        if isinstance(judge_resp_raw, str):
            try:
                judge_resp = json.loads(judge_resp_raw)
            except:
                judge_resp = {}
        else:
            judge_resp = judge_resp_raw if isinstance(judge_resp_raw, dict) else {}
        judge_feedback = judge_resp.get('FINAL_FEEDBACK', '') 
        
        # Extract KG step and rule for leakage detection
        KG_text = (instance_tutor.get("KG_correct_steps") or [""])[0]
        KG_plan = re.sub(r'^Step\s*\d+\s*:\s*', '', KG_text)
        # Pattern: "Derive (expression) from ... using the RuleName rule."
        plan_match = re.search(r'Derive\s+([^from]+)\s+from', KG_plan)
        rule_match = re.search(r'using\s+the\s+([^.]*?)\s+rule', KG_plan)
        kg_step = plan_match.group(1).strip() if plan_match else "N/A"
        KG_rule = rule_match.group(1).strip() if rule_match else "N/A"
        # Convert full rule name to short name
        kg_rule = convert_rule_to_short_name(KG_rule)

        
        correct_answer = kg_step
        reference = correct_answer
        
        # Evaluate all three pipelines
        tutor_metrics = evaluator.evaluate_single_feedback(
            tutor_feedback, student_utterance, reference, correct_answer,
            kg_step=kg_step, kg_rule=kg_rule
        )
        teacher_metrics = evaluator.evaluate_single_feedback(
            teacher_feedback, student_utterance, reference, correct_answer,
            kg_step=kg_step, kg_rule=kg_rule
        )
        judge_metrics = evaluator.evaluate_single_feedback(
            judge_feedback, student_utterance, reference, correct_answer,
            kg_step=kg_step, kg_rule=kg_rule
        )
        
        # Store ALL metrics
        result = {
            'problem_id': instance_tutor.get('problem_id', idx),
            'kg_step': kg_step,
            'kg_rule': kg_rule
        }
        
        # All metrics to store
        all_metric_keys = [
            # Original metrics
            'question_count', 'question_ratio', 'hint_count', 'direct_count',
            'scaffolding_ratio', 'scaffolding_score', 'specificity',
            'informativeness', 'coherence', 'flesch_ease', 'word_count',
            'sentence_count', 'lexical_diversity', 'bertscore_f1',
            'bertscore_precision', 'bertscore_recall',
            # Question & engagement metrics
            'question_diversity', 'second_person_count', 'second_person_ratio',
            'verb_count', 'verb_ratio',
            # Leakage metrics
            'rule_leakage_count', 'rule_leakage_binary', 
            'expression_overlap', 'expression_containment',
            'target_rule_leaked', 'comprehensive_leakage'
        ]
        
        for metric in all_metric_keys:
            result[f'tutor_{metric}'] = tutor_metrics.get(metric)
            result[f'teacher_{metric}'] = teacher_metrics.get(metric)
            result[f'judge_{metric}'] = judge_metrics.get(metric)
        
        # Store rules leaked (as string for CSV)
        result['tutor_rules_leaked'] = ','.join(tutor_metrics.get('rules_leaked', []))
        result['teacher_rules_leaked'] = ','.join(teacher_metrics.get('rules_leaked', []))
        result['judge_rules_leaked'] = ','.join(judge_metrics.get('rules_leaked', []))
        
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
        ('question_diversity', 'Question Diversity'),
        ('hint_count', 'Hint Count'),
        ('direct_count', 'Direct Statements'),
        ('specificity', 'Specificity Score'),
        ('second_person_ratio', 'Second-Person Ratio'),
        ('verb_ratio', 'Verb Ratio'),
        ('lexical_diversity', 'Lexical Diversity (MTLD)'),
        ('informativeness', 'Informativeness Index (I²)'),
        ('coherence', 'Coherence with Student'),
        ('flesch_ease', 'Flesch Reading Ease'),
        ('word_count', 'Word Count'),
        ('bertscore_f1', 'BERTScore F1'),
        # Leakage metrics
        ('rule_leakage_count', 'Rule Leakage Count'),
        ('rule_leakage_binary', 'Rule Leakage Rate'),
        ('expression_overlap', 'Expression Overlap'),
        ('expression_containment', 'Expression Containment'),
        ('comprehensive_leakage', 'Comprehensive Leakage'),
        ('target_rule_leaked', 'Target Rule Leaked'),
    ]
    
    for pipeline, prefix in [('TUTOR (S→Tu)', 'tutor'), 
                              ('TEACHER (S→T)', 'teacher'), 
                              ('JUDGE (S→Tu→J)', 'judge')]:
        print(f"\n--- {pipeline} ---")
        for metric_key, metric_name in metrics_to_print:
            col = f"{prefix}_{metric_key}"
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if pd.notna(mean_val):
                    print(f"{metric_name}: {mean_val:.3f} (±{std_val:.3f})")
    
    # ==================== LEAKAGE SUMMARY ====================
    print("\n" + "="*80)
    print("LEAKAGE DETECTION SUMMARY")
    print("="*80)
    
    for pipeline, prefix in [('TUTOR (S→Tu)', 'tutor'), 
                              ('TEACHER (S→T)', 'teacher'), 
                              ('JUDGE (S→Tu→J)', 'judge')]:
        print(f"\n--- {pipeline} ---")
        
        # Rule leakage rate
        leak_col = f'{prefix}_rule_leakage_binary'
        if leak_col in df.columns:
            leak_rate = df[leak_col].mean() * 100
            print(f"Rule Leakage Rate: {leak_rate:.1f}%")
        
        # Target rule leaked rate
        target_col = f'{prefix}_target_rule_leaked'
        if target_col in df.columns and df[target_col].notna().any():
            target_rate = df[target_col].mean() * 100
            print(f"Target Rule Leaked Rate: {target_rate:.1f}%")
        
        # Comprehensive leakage score
        comp_col = f'{prefix}_comprehensive_leakage'
        if comp_col in df.columns:
            comp_mean = df[comp_col].mean()
            comp_std = df[comp_col].std()
            if pd.notna(comp_mean):
                print(f"Comprehensive Leakage Score: {comp_mean:.3f} (±{comp_std:.3f})")
        
        # Most commonly leaked rules
        rules_col = f'{prefix}_rules_leaked'
        if rules_col in df.columns:
            all_rules = []
            for rules_str in df[rules_col].dropna():
                if rules_str:
                    all_rules.extend(rules_str.split(','))
            if all_rules:
                from collections import Counter
                rule_counts = Counter(all_rules)
                top_rules = rule_counts.most_common(5)
                print(f"Most Leaked Rules: {', '.join(f'{r}({c})' for r, c in top_rules)}")
    
    # ==================== STATISTICAL ANALYSIS ====================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS (ACL Standards)")
    print("="*80)
    
    metrics_to_analyze = [
        ('scaffolding_ratio', 'Scaffolding Ratio'),
        ('question_count', 'Question Count'),
        ('question_diversity', 'Question Diversity'),
        ('informativeness', 'Informativeness Index (I²)'),
        ('coherence', 'Coherence with Student'),
        ('specificity', 'Specificity Score'),
        ('second_person_ratio', 'Second-Person Ratio'),
        ('verb_ratio', 'Verb Ratio'),
        ('lexical_diversity', 'Lexical Diversity (MTLD)'),
        ('flesch_ease', 'Flesch Reading Ease'),
        # Leakage metrics
        ('rule_leakage_count', 'Rule Leakage Count'),
        ('comprehensive_leakage', 'Comprehensive Leakage'),
        ('expression_overlap', 'Expression Overlap'),
    ]
    
    all_stats_results = []
    
    for metric_col, metric_name in metrics_to_analyze:
        tutor_col = f'tutor_{metric_col}'
        teacher_col = f'teacher_{metric_col}'
        judge_col = f'judge_{metric_col}'
        
        if tutor_col not in df.columns:
            continue
        
        # Get scores, handling NaN
        valid_mask = df[tutor_col].notna() & df[teacher_col].notna() & df[judge_col].notna()
        tutor_scores = df.loc[valid_mask, tutor_col].tolist()
        teacher_scores = df.loc[valid_mask, teacher_col].tolist()
        judge_scores = df.loc[valid_mask, judge_col].tolist()
        
        if len(tutor_scores) < 10:
            print(f"Warning: Skipping {metric_name} - insufficient data ({len(tutor_scores)} samples)")
            continue
        
        # Run 3-way statistical analysis
        stats_results = run_three_way_statistical_analysis(
            tutor_scores, teacher_scores, judge_scores, metric_name
        )
        
        print_statistical_results(stats_results)
        all_stats_results.append(stats_results)
    
    # ==================== SUMMARY OF SIGNIFICANT DIFFERENCES ====================
    print("\n" + "="*80)
    print("SUMMARY: SIGNIFICANT DIFFERENCES (p < 0.05, Holm-Bonferroni corrected)")
    print("="*80)
    
    comparisons = [
        ('tutor_vs_teacher', 'S→Tu vs S→T'),
        ('teacher_vs_judge', 'S→T vs S→Tu→J'),
        ('tutor_vs_judge', 'S→Tu vs S→Tu→J')
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
                if comp_key == 'tutor_vs_teacher':
                    winner = "Tutor" if delta > 0 else "Teacher"
                elif comp_key == 'teacher_vs_judge':
                    winner = "Teacher" if delta > 0 else "Judge"
                else:
                    winner = "Tutor" if delta > 0 else "Judge"
                
                print(f"  • {stats['metric']}: {winner} wins ({sig}, δ={delta:.2f}, {effect})")
                sig_found = True
        
        if not sig_found:
            print("  No significant differences found")
    
    # ==================== PERPLEXITY ====================
    print("\n" + "="*80)
    print("PERPLEXITY ANALYSIS")
    print("="*80)
    
    tutor_texts = [parse_response(inst.get("teacher_feedback", {}), "TEACHER_FEEDBACK") 
                   for inst in data_tutor]
    teacher_texts = [parse_response(inst.get("judge_feedback", {}), "JUDGE_FEEDBACK") 
                     for inst in data_teacher]
    judge_texts = [parse_response(inst.get("final_feedback", {}), "FINAL_FEEDBACK") 
                   for inst in data_judge]
    
    tutor_perplexity = evaluator.compute_perplexity(tutor_texts)
    teacher_perplexity = evaluator.compute_perplexity(teacher_texts)
    judge_perplexity = evaluator.compute_perplexity(judge_texts)
    
    if tutor_perplexity:
        print(f"Tutor (S→Tu) Perplexity: {tutor_perplexity:.2f}")
    if teacher_perplexity:
        print(f"Teacher (S→T) Perplexity: {teacher_perplexity:.2f}")
    if judge_perplexity:
        print(f"Judge (S→Tu→J) Perplexity: {judge_perplexity:.2f}")
    
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


# ==================== STANDALONE LEAKAGE ANALYSIS ====================

def analyze_leakage_only(
    data_path: str,
    feedback_key: str = 'tutor_response',
    feedback_field: str = 'TUTOR_FEEDBACK',
    output_path: str = 'leakage_analysis.csv'
):
    """
    Standalone function to analyze leakage in a single feedback file.
    """
    evaluator = CompletePedagogicalMetrics(device='cpu', use_bertscore=False)
    
    data = load_jsonl(data_path)
    results = []
    
    for idx, instance in enumerate(tqdm(data, desc="Analyzing leakage")):
        # Extract feedback
        resp_raw = instance.get(feedback_key, {})
        if isinstance(resp_raw, str):
            try:
                resp = json.loads(resp_raw)
            except:
                resp = {}
        else:
            resp = resp_raw if isinstance(resp_raw, dict) else {}
        feedback = resp.get(feedback_field, '')
        
        # Extract KG info
        KG_correct_steps = instance.get('KG_correct_steps', [])
        if isinstance(KG_correct_steps, str):
            KG_correct_steps = [KG_correct_steps]
        kg_step = KG_correct_steps[0] if KG_correct_steps else None
        
        KG_correct_rules = instance.get('KG_correct_rules', [])
        if isinstance(KG_correct_rules, str):
            KG_correct_rules = [KG_correct_rules]
        kg_rule = KG_correct_rules[0] if KG_correct_rules else None
        
        # Compute leakage metrics
        result = {
            'problem_id': instance.get('problem_id', idx),
            'kg_step': kg_step,
            'kg_rule': kg_rule,
            'feedback': feedback[:200] + '...' if len(feedback) > 200 else feedback,
            'rule_leakage_count': evaluator.rule_leakage_count(feedback),
            'rule_leakage_binary': evaluator.rule_leakage_binary(feedback),
            'rules_leaked': ','.join(evaluator.rules_leaked(feedback)),
            'expression_overlap': evaluator.expression_overlap(feedback, kg_step) if kg_step else None,
            'expression_containment': evaluator.expression_containment(feedback, kg_step) if kg_step else None,
            'comprehensive_leakage': evaluator.comprehensive_leakage_score(feedback, kg_step, kg_rule, kg_step),
            'target_rule_leaked': evaluator.specific_rule_leaked(feedback, kg_rule) if kg_rule else None,
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("LEAKAGE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total instances: {len(df)}")
    print(f"Rule Leakage Rate: {df['rule_leakage_binary'].mean()*100:.1f}%")
    if df['target_rule_leaked'].notna().any():
        print(f"Target Rule Leaked Rate: {df['target_rule_leaked'].mean()*100:.1f}%")
    print(f"Mean Comprehensive Leakage: {df['comprehensive_leakage'].mean():.3f}")
    
    # Most leaked rules
    all_rules = []
    for rules_str in df['rules_leaked'].dropna():
        if rules_str:
            all_rules.extend(rules_str.split(','))
    if all_rules:
        from collections import Counter
        rule_counts = Counter(all_rules)
        print(f"\nMost Leaked Rules:")
        for rule, count in rule_counts.most_common(10):
            print(f"  {rule}: {count} ({count/len(df)*100:.1f}%)")
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    process_all_instances(
        data_path_TUTOR='Data/llm_output/qwen/qwen_baseline_1.jsonl',      # S→Tu
        data_path_TEACHER='Data/llm_output/qwen/qwen_baseline_2.jsonl',  # S→T
        data_path_JUDGE='Data/llm_output/qwen/qwen_ours.jsonl',      # S→Tu→J
        output_path='Data/llm_output/qwen/qwen_metrics_results.csv',
        use_bertscore=True,
        device='cpu'
    )