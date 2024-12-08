import json
from datetime import datetime
from typing import Dict, List
import os
from pinecone import Pinecone
import numpy as np
from sentence_transformers import SentenceTransformer

class UserProfile:
    def __init__(self, username: str):
        self.username = username
        self.profile_dir = f"profiles/{username}"
        self.interactions_file = f"{self.profile_dir}/interactions.json"
        self.stats_file = f"{self.profile_dir}/stats.json"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_profile()
        
    def _initialize_profile(self):
        """Initialize profile directory and files if they don't exist"""
        os.makedirs(self.profile_dir, exist_ok=True)
        
        if not os.path.exists(self.interactions_file):
            self._save_interactions([])
            
        if not os.path.exists(self.stats_file):
            self._save_stats({
                "total_questions": 0,
                "topics": {},
                "last_active": str(datetime.now()),
                "understanding_score": 0
            })
    
    def _save_interactions(self, interactions: List[Dict]):
        with open(self.interactions_file, 'w') as f:
            json.dump(interactions, f, indent=2)
    
    def _save_stats(self, stats: Dict):
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _load_interactions(self) -> List[Dict]:
        with open(self.interactions_file, 'r') as f:
            return json.load(f)
    
    def _load_stats(self) -> Dict:
        with open(self.stats_file, 'r') as f:
            return json.load(f)
    
    def log_interaction(self, question: str, answer: str, context_used: str, topic: str):
        """Log a new interaction and update stats"""
        # Load current data
        interactions = self._load_interactions()
        stats = self._load_stats()
        
        # Create interaction entry
        interaction = {
            "timestamp": str(datetime.now()),
            "question": question,
            "answer": answer,
            "context_used": context_used,
            "topic": topic,
            "embedding": self.model.encode(question).tolist()
        }
        
        # Update interactions
        interactions.append(interaction)
        self._save_interactions(interactions)
        
        # Update stats
        stats["total_questions"] += 1
        stats["last_active"] = str(datetime.now())
        if topic not in stats["topics"]:
            stats["topics"][topic] = {
                "questions_asked": 0,
                "last_interaction": None
            }
        stats["topics"][topic]["questions_asked"] += 1
        stats["topics"][topic]["last_interaction"] = str(datetime.now())
        
        # Calculate understanding score based on interaction patterns
        self._update_understanding_score(stats)
        self._save_stats(stats)
    
    def _update_understanding_score(self, stats: Dict):
        """Update understanding score based on interaction patterns"""
        # This is a simple scoring mechanism - can be made more sophisticated
        total_questions = stats["total_questions"]
        topic_diversity = len(stats["topics"])
        
        # Basic score calculation
        base_score = min(total_questions / 100, 1.0)  # Max out at 100 questions
        diversity_bonus = min(topic_diversity / 5, 1.0)  # Max out at 5 topics
        
        stats["understanding_score"] = (base_score * 0.7 + diversity_bonus * 0.3) * 100
    
    def get_progress_stats(self) -> Dict:
        """Get user progress statistics"""
        return self._load_stats()
    
    def get_topic_progress(self, topic: str) -> Dict:
        """Get progress for a specific topic"""
        stats = self._load_stats()
        return stats["topics"].get(topic, {
            "questions_asked": 0,
            "last_interaction": None
        })
