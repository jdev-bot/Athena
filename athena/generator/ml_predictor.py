"""ML fitness predictor for seeding promising candidates."""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from athena.common.models import StrategyTemplate
from athena.generator.dna import DNAEncoder
from athena.generator.ga_engine import Individual


class MLPredictor:
    """Predict strategy fitness from DNA using ML."""
    
    def __init__(self, template: StrategyTemplate):
        self.template = template
        self.encoder = DNAEncoder()
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.spec = self.encoder.get_spec(template)
        self.param_names = [s.name for s in self.spec]
        
    def _dna_to_features(self, dna: Dict[str, Any]) -> np.ndarray:
        """Convert DNA dict to feature vector."""
        features = []
        for name in self.param_names:
            val = dna.get(name, 0)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            features.append(float(val))
        return np.array(features)
    
    def _population_to_arrays(self, population: List[Individual]) -> tuple:
        """Convert population to X, y arrays."""
        X = []
        y = []
        for ind in population:
            X.append(self._dna_to_features(ind.dna))
            y.append(ind.fitness)
        return np.array(X), np.array(y)
    
    def train(self, population: List[Individual]) -> None:
        """Train predictor on evaluated population."""
        if len(population) < 10:
            self.is_trained = False
            return
        
        X, y = self._population_to_arrays(population)
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Validate
        val_score = self.model.score(X_val, y_val)
        print(f"ML Predictor R²: {val_score:.3f}")
        
        self.is_trained = True
    
    def predict(self, dna: Dict[str, Any]) -> float:
        """Predict fitness for a DNA vector."""
        if not self.is_trained:
            return 0.5  # Neutral prediction
        
        features = self._dna_to_features(dna).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        pred = self.model.predict(features_scaled)[0]
        return float(pred)
    
    def generate_promising_candidates(self, template: StrategyTemplate,
                                     n_candidates: int = 10) -> List[Dict[str, Any]]:
        """Generate DNA vectors predicted to have high fitness."""
        if not self.is_trained:
            # Return random candidates
            return [self.encoder.random_dna(template) for _ in range(n_candidates)]
        
        candidates = []
        best_score = -1
        best_dna = None
        
        # Random search guided by predictor
        for _ in range(n_candidates * 10):
            dna = self.encoder.random_dna(template)
            score = self.predict(dna)
            if score > best_score:
                best_score = score
                best_dna = dna
            
            if len(candidates) < n_candidates and score > 0.5:
                candidates.append(dna)
        
        # Ensure we have enough
        while len(candidates) < n_candidates:
            if best_dna:
                candidates.append(best_dna)
            else:
                candidates.append(self.encoder.random_dna(template))
        
        return candidates[:n_candidates]
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'template': self.template.value,
                'param_names': self.param_names,
            }, f)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.param_names = data['param_names']
