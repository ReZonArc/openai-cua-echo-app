"""
Deep Tree Echo Computer Use ML Agent

This module implements a machine learning enhanced agent that organizes interactions
in a hierarchical tree structure and learns patterns from user interactions to provide
intelligent echoing and prediction capabilities.
"""

import json
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import hashlib
from datetime import datetime

from .agent import Agent
from computers import Computer


class TreeNode:
    """Represents a node in the deep tree structure for organizing interactions."""
    
    def __init__(self, action_type: str = None, context: Dict[str, Any] = None):
        self.action_type = action_type
        self.context = context or {}
        self.children: Dict[str, 'TreeNode'] = {}
        self.frequency = 0
        self.success_rate = 0.0
        self.total_attempts = 0
        self.successful_attempts = 0
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        
    def add_child(self, key: str, child: 'TreeNode'):
        """Add a child node to this node."""
        self.children[key] = child
        
    def get_child(self, key: str) -> Optional['TreeNode']:
        """Get a child node by key."""
        return self.children.get(key)
        
    def update_stats(self, success: bool):
        """Update node statistics based on action outcome."""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1
        self.success_rate = self.successful_attempts / self.total_attempts
        self.frequency += 1
        self.last_accessed = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            'action_type': self.action_type,
            'context': self.context,
            'frequency': self.frequency,
            'success_rate': self.success_rate,
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'children': {k: v.to_dict() for k, v in self.children.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeNode':
        """Create node from dictionary."""
        node = cls(data.get('action_type'), data.get('context', {}))
        node.frequency = data.get('frequency', 0)
        node.success_rate = data.get('success_rate', 0.0)
        node.total_attempts = data.get('total_attempts', 0)
        node.successful_attempts = data.get('successful_attempts', 0)
        node.created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        node.last_accessed = datetime.fromisoformat(data.get('last_accessed', datetime.now().isoformat()))
        
        # Recursively create children
        for k, v in data.get('children', {}).items():
            node.children[k] = cls.from_dict(v)
        
        return node


class MLPatternLearner:
    """Simple ML pattern learner for action sequences and outcomes."""
    
    def __init__(self):
        self.pattern_frequencies = defaultdict(int)
        self.action_sequences = deque(maxlen=1000)  # Store recent sequences
        self.success_patterns = defaultdict(list)
        
    def add_sequence(self, sequence: List[str], success: bool):
        """Add an action sequence with its outcome."""
        seq_key = "->".join(sequence)
        self.pattern_frequencies[seq_key] += 1
        self.success_patterns[seq_key].append(success)
        self.action_sequences.append((sequence, success))
        
    def predict_success(self, sequence: List[str]) -> float:
        """Predict success probability for a given sequence."""
        seq_key = "->".join(sequence)
        if seq_key in self.success_patterns:
            successes = self.success_patterns[seq_key]
            return sum(successes) / len(successes)
        
        # Try partial matches
        for i in range(len(sequence) - 1, 0, -1):
            partial_key = "->".join(sequence[-i:])
            if partial_key in self.success_patterns:
                successes = self.success_patterns[partial_key]
                return sum(successes) / len(successes) * 0.8  # Reduce confidence for partial match
                
        return 0.5  # Default probability
        
    def get_common_patterns(self, min_frequency: int = 3) -> List[Tuple[str, int, float]]:
        """Get common patterns with their frequency and success rate."""
        patterns = []
        for pattern, freq in self.pattern_frequencies.items():
            if freq >= min_frequency:
                successes = self.success_patterns[pattern]
                success_rate = sum(successes) / len(successes)
                patterns.append((pattern, freq, success_rate))
        
        return sorted(patterns, key=lambda x: x[1], reverse=True)


class TreeEchoMLAgent(Agent):
    """
    Deep Tree Echo Computer Use ML Agent
    
    Extends the base Agent with:
    - Hierarchical tree organization of interactions
    - ML-based pattern learning and prediction
    - Echo functionality that learns from user patterns
    - Deep navigation through interaction trees
    """
    
    def __init__(self, 
                 model="computer-use-preview",
                 computer: Computer = None,
                 tools: list[dict] = [],
                 acknowledge_safety_check_callback=lambda: False,
                 tree_file_path: str = None):
        
        super().__init__(model, computer, tools, acknowledge_safety_check_callback)
        
        # Tree and ML components
        self.interaction_tree = TreeNode("root")
        self.ml_learner = MLPatternLearner()
        self.current_path: List[str] = []
        self.action_history: List[str] = []
        
        # Persistence
        self.tree_file_path = tree_file_path or "/tmp/tree_echo_ml_data.json"
        self.load_tree_data()
        
        # Echo configuration
        self.echo_enabled = True
        self.echo_threshold = 0.7  # Confidence threshold for auto-echo
        self.max_tree_depth = 10
        
        print("🌳 Deep Tree Echo ML Agent initialized")
        
    def load_tree_data(self):
        """Load tree and ML data from file."""
        if os.path.exists(self.tree_file_path):
            try:
                with open(self.tree_file_path, 'r') as f:
                    data = json.load(f)
                    if 'tree' in data:
                        self.interaction_tree = TreeNode.from_dict(data['tree'])
                    if 'ml_patterns' in data:
                        ml_data = data['ml_patterns']
                        self.ml_learner.pattern_frequencies = defaultdict(int, ml_data.get('frequencies', {}))
                        self.ml_learner.success_patterns = defaultdict(list, ml_data.get('success_patterns', {}))
                print(f"📁 Loaded tree data from {self.tree_file_path}")
            except Exception as e:
                print(f"⚠️  Error loading tree data: {e}")
                
    def save_tree_data(self):
        """Save tree and ML data to file."""
        try:
            data = {
                'tree': self.interaction_tree.to_dict(),
                'ml_patterns': {
                    'frequencies': dict(self.ml_learner.pattern_frequencies),
                    'success_patterns': dict(self.ml_learner.success_patterns)
                }
            }
            with open(self.tree_file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved tree data to {self.tree_file_path}")
        except Exception as e:
            print(f"⚠️  Error saving tree data: {e}")
            
    def _generate_action_key(self, item: Dict[str, Any]) -> str:
        """Generate a unique key for an action based on its type and context."""
        if item.get("type") == "computer_call":
            action = item.get("action", {})
            action_type = action.get("type", "unknown")
            
            # Create context-aware key
            context_parts = [action_type]
            
            # Add relevant context based on action type
            if action_type == "click":
                x, y = action.get("x", 0), action.get("y", 0)
                # Normalize coordinates to screen regions
                screen_region = f"region_{x//100}_{y//100}"
                context_parts.append(screen_region)
            elif action_type == "type":
                text_length = len(action.get("text", ""))
                text_type = "short" if text_length < 10 else "medium" if text_length < 50 else "long"
                context_parts.append(text_type)
            elif action_type in ["scroll", "drag"]:
                direction = "up" if action.get("scroll_y", 0) < 0 else "down" if action.get("scroll_y", 0) > 0 else "horizontal"
                context_parts.append(direction)
                
            return "_".join(context_parts)
        
        elif item.get("type") == "function_call":
            return f"function_{item.get('name', 'unknown')}"
        
        return f"{item.get('type', 'unknown')}"
        
    def _navigate_to_tree_node(self, action_key: str) -> TreeNode:
        """Navigate to or create tree node for given action."""
        current_node = self.interaction_tree
        
        # Build path through tree
        path_parts = self.current_path + [action_key]
        
        for i, part in enumerate(path_parts):
            if part not in current_node.children:
                # Create new node
                node_context = {
                    'depth': i,
                    'path': path_parts[:i+1],
                    'parent_type': current_node.action_type
                }
                current_node.add_child(part, TreeNode(part, node_context))
                
            current_node = current_node.get_child(part)
            
        return current_node
        
    def _predict_and_echo(self, action_key: str) -> Optional[str]:
        """Predict and potentially echo based on learned patterns."""
        if not self.echo_enabled:
            return None
            
        # Get recent action sequence
        recent_actions = self.action_history[-5:] + [action_key]
        
        # Predict success probability
        success_prob = self.ml_learner.predict_success(recent_actions)
        
        if success_prob > self.echo_threshold:
            # Find similar successful patterns
            common_patterns = self.ml_learner.get_common_patterns()
            
            for pattern, freq, success_rate in common_patterns[:3]:
                if action_key in pattern and success_rate > 0.8:
                    echo_msg = f"🔮 Echo: Similar pattern '{pattern}' succeeded {success_rate:.1%} of the time (n={freq})"
                    return echo_msg
                    
        return None
        
    def handle_item(self, item):
        """Enhanced item handling with tree organization and ML learning."""
        # Generate action key
        action_key = self._generate_action_key(item)
        
        # Navigate to tree node
        tree_node = self._navigate_to_tree_node(action_key)
        
        # Predict and echo if relevant
        echo_message = self._predict_and_echo(action_key)
        if echo_message:
            print(echo_message)
            
        # Add to action history
        self.action_history.append(action_key)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]  # Keep recent history
            
        # Call parent handler
        result = super().handle_item(item)
        
        # Update tree node statistics (assume success if no exception)
        success = True
        try:
            # Check if action was successful based on result
            if result and len(result) > 0:
                success = True
            tree_node.update_stats(success)
            
            # Update ML learner
            if len(self.action_history) >= 2:
                sequence = self.action_history[-2:]  # Last 2 actions
                self.ml_learner.add_sequence(sequence, success)
                
        except Exception as e:
            success = False
            tree_node.update_stats(success)
            print(f"⚠️  Action failed: {e}")
            
        return result
        
    def run_full_turn(self, input_items, print_steps=True, debug=False, show_images=False):
        """Enhanced turn execution with deep tree navigation and learning."""
        print("🚀 Starting deep tree echo ML turn")
        
        # Update current path based on input
        if input_items:
            user_input = input_items[-1].get("content", "")
            # Create a simple hash-based context for the user input
            input_hash = hashlib.md5(user_input.encode()).hexdigest()[:8]
            self.current_path = [f"input_{input_hash}"]
        
        # Run parent implementation
        result = super().run_full_turn(input_items, print_steps, debug, show_images)
        
        # Analyze and learn from the complete turn
        self._analyze_turn_patterns()
        
        # Periodically save data
        if len(self.action_history) % 10 == 0:
            self.save_tree_data()
            
        print("🏁 Deep tree echo ML turn completed")
        return result
        
    def _analyze_turn_patterns(self):
        """Analyze patterns from the completed turn."""
        if len(self.action_history) >= 3:
            # Look for recurring patterns
            recent_sequence = self.action_history[-3:]
            prediction = self.ml_learner.predict_success(recent_sequence)
            
            if prediction > 0.8:
                print(f"✨ High success pattern detected: {' -> '.join(recent_sequence)} (confidence: {prediction:.1%})")
            elif prediction < 0.3:
                print(f"⚡ Low success pattern detected: {' -> '.join(recent_sequence)} (confidence: {prediction:.1%})")
                
    def get_tree_summary(self) -> Dict[str, Any]:
        """Get summary of the interaction tree and learned patterns."""
        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
            
        tree_size = count_nodes(self.interaction_tree)
        common_patterns = self.ml_learner.get_common_patterns()
        
        return {
            'tree_size': tree_size,
            'total_actions': len(self.action_history),
            'learned_patterns': len(self.ml_learner.pattern_frequencies),
            'common_patterns': common_patterns[:5],  # Top 5 patterns
            'echo_enabled': self.echo_enabled,
            'tree_depth': len(self.current_path)
        }
        
    def print_tree_summary(self):
        """Print a formatted summary of the tree and learning data."""
        summary = self.get_tree_summary()
        
        print("\n🌳 Deep Tree Echo ML Agent Summary:")
        print(f"   Tree Size: {summary['tree_size']} nodes")
        print(f"   Total Actions: {summary['total_actions']}")
        print(f"   Learned Patterns: {summary['learned_patterns']}")
        print(f"   Current Depth: {summary['tree_depth']}")
        print(f"   Echo Enabled: {summary['echo_enabled']}")
        
        if summary['common_patterns']:
            print("\n📊 Top Patterns:")
            for pattern, freq, success_rate in summary['common_patterns']:
                print(f"   {pattern} (n={freq}, success={success_rate:.1%})")
        
        print()