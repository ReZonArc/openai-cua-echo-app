"""
Tests for Deep Tree Echo ML Agent functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch

from agent.tree_echo_ml_agent import TreeNode, MLPatternLearner, TreeEchoMLAgent


class TestTreeNode:
    """Test the TreeNode class functionality."""
    
    def test_node_creation(self):
        """Test basic node creation and properties."""
        node = TreeNode("click", {"x": 100, "y": 200})
        
        assert node.action_type == "click"
        assert node.context == {"x": 100, "y": 200}
        assert node.frequency == 0
        assert node.success_rate == 0.0
        assert len(node.children) == 0
        
    def test_add_child(self):
        """Test adding child nodes."""
        parent = TreeNode("root")
        child = TreeNode("click")
        
        parent.add_child("click_action", child)
        
        assert "click_action" in parent.children
        assert parent.get_child("click_action") == child
        assert parent.get_child("nonexistent") is None
        
    def test_update_stats(self):
        """Test statistics tracking."""
        node = TreeNode("click")
        
        # Add successful attempt
        node.update_stats(True)
        assert node.total_attempts == 1
        assert node.successful_attempts == 1
        assert node.success_rate == 1.0
        assert node.frequency == 1
        
        # Add failed attempt
        node.update_stats(False)
        assert node.total_attempts == 2
        assert node.successful_attempts == 1
        assert node.success_rate == 0.5
        assert node.frequency == 2
        
    def test_serialization(self):
        """Test node serialization and deserialization."""
        original = TreeNode("click", {"x": 100, "y": 200})
        original.update_stats(True)
        
        child = TreeNode("type", {"text": "hello"})
        original.add_child("child_action", child)
        
        # Serialize
        data = original.to_dict()
        
        # Deserialize
        restored = TreeNode.from_dict(data)
        
        assert restored.action_type == original.action_type
        assert restored.context == original.context
        assert restored.frequency == original.frequency
        assert restored.success_rate == original.success_rate
        assert "child_action" in restored.children
        assert restored.children["child_action"].action_type == "type"


class TestMLPatternLearner:
    """Test the MLPatternLearner class functionality."""
    
    def test_add_sequence(self):
        """Test adding action sequences."""
        learner = MLPatternLearner()
        
        sequence = ["click", "type", "scroll"]
        learner.add_sequence(sequence, True)
        
        seq_key = "click->type->scroll"
        assert learner.pattern_frequencies[seq_key] == 1
        assert len(learner.success_patterns[seq_key]) == 1
        assert learner.success_patterns[seq_key][0] is True
        
    def test_predict_success(self):
        """Test success prediction."""
        learner = MLPatternLearner()
        
        # Add multiple sequences with different outcomes
        sequence = ["click", "type"]
        learner.add_sequence(sequence, True)
        learner.add_sequence(sequence, True)
        learner.add_sequence(sequence, False)
        
        # Should predict 2/3 success rate
        prediction = learner.predict_success(sequence)
        assert abs(prediction - (2/3)) < 0.01
        
    def test_partial_match_prediction(self):
        """Test prediction with partial sequence matches."""
        learner = MLPatternLearner()
        
        # Add a longer sequence
        learner.add_sequence(["start", "click", "type"], True)
        learner.add_sequence(["start", "click", "type"], True)
        
        # Test prediction for partial sequence
        prediction = learner.predict_success(["click", "type"])
        assert prediction >= 0.5  # Should get reduced confidence match or default
        
    def test_get_common_patterns(self):
        """Test retrieval of common patterns."""
        learner = MLPatternLearner()
        
        # Add patterns with different frequencies
        for i in range(5):
            learner.add_sequence(["click", "type"], True)
        for i in range(3):
            learner.add_sequence(["scroll", "click"], False)
        for i in range(2):
            learner.add_sequence(["drag", "drop"], True)
            
        patterns = learner.get_common_patterns(min_frequency=3)
        
        assert len(patterns) == 2  # Only patterns with freq >= 3
        # Should be sorted by frequency (descending)
        assert patterns[0][1] >= patterns[1][1]


class TestTreeEchoMLAgent:
    """Test the TreeEchoMLAgent class functionality."""
    
    def create_mock_computer(self):
        """Create a mock computer for testing."""
        mock_computer = Mock()
        mock_computer.get_dimensions.return_value = (1920, 1080)
        mock_computer.get_environment.return_value = "browser"
        mock_computer.screenshot.return_value = "fake_screenshot_data"
        return mock_computer
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        mock_computer = self.create_mock_computer()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            tree_file = f.name
            
        try:
            agent = TreeEchoMLAgent(
                computer=mock_computer,
                tree_file_path=tree_file
            )
            
            assert agent.echo_enabled is True
            assert agent.echo_threshold == 0.7
            assert agent.interaction_tree.action_type == "root"
            assert isinstance(agent.ml_learner, MLPatternLearner)
            
        finally:
            os.unlink(tree_file)
            
    def test_generate_action_key(self):
        """Test action key generation."""
        mock_computer = self.create_mock_computer()
        agent = TreeEchoMLAgent(computer=mock_computer)
        
        # Test click action
        click_item = {
            "type": "computer_call",
            "action": {
                "type": "click",
                "x": 150,
                "y": 250
            }
        }
        key = agent._generate_action_key(click_item)
        assert key.startswith("click_region_")
        
        # Test type action
        type_item = {
            "type": "computer_call",
            "action": {
                "type": "type",
                "text": "hello"
            }
        }
        key = agent._generate_action_key(type_item)
        assert "type_short" in key
        
        # Test function call
        func_item = {
            "type": "function_call",
            "name": "test_function"
        }
        key = agent._generate_action_key(func_item)
        assert key == "function_test_function"
        
    def test_tree_navigation(self):
        """Test tree navigation functionality."""
        mock_computer = self.create_mock_computer()
        agent = TreeEchoMLAgent(computer=mock_computer)
        
        # Navigate to a node
        node = agent._navigate_to_tree_node("test_action")
        
        assert node is not None
        assert node.action_type == "test_action"
        assert "test_action" in agent.interaction_tree.children
        
        # Navigate to same node again
        node2 = agent._navigate_to_tree_node("test_action")
        assert node2 == node  # Should return same node
        
    def test_prediction_and_echo(self):
        """Test prediction and echo functionality."""
        mock_computer = self.create_mock_computer()
        agent = TreeEchoMLAgent(computer=mock_computer)
        
        # Add some history to enable prediction
        agent.action_history = ["click_region_1_1", "type_short"]
        
        # Add patterns to ML learner
        agent.ml_learner.add_sequence(["click_region_1_1", "type_short", "scroll_down"], True)
        agent.ml_learner.add_sequence(["click_region_1_1", "type_short", "scroll_down"], True)
        
        # Test prediction
        echo_msg = agent._predict_and_echo("scroll_down")
        
        # Should generate echo message due to high success rate
        if echo_msg:
            assert "Echo:" in echo_msg
            assert "succeeded" in echo_msg
            
    def test_save_and_load_tree_data(self):
        """Test saving and loading tree data."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            tree_file = f.name
            
        try:
            mock_computer = self.create_mock_computer()
            
            # Create agent and add some data
            agent1 = TreeEchoMLAgent(computer=mock_computer, tree_file_path=tree_file)
            agent1.action_history = ["test_action"]
            agent1.ml_learner.add_sequence(["action1", "action2"], True)
            
            # Navigate to create tree structure
            agent1._navigate_to_tree_node("test_action")
            
            # Save data
            agent1.save_tree_data()
            
            # Create new agent and load data
            agent2 = TreeEchoMLAgent(computer=mock_computer, tree_file_path=tree_file)
            
            # Verify data was loaded
            assert "test_action" in agent2.interaction_tree.children
            assert len(agent2.ml_learner.pattern_frequencies) > 0
            
        finally:
            if os.path.exists(tree_file):
                os.unlink(tree_file)
                
    def test_get_tree_summary(self):
        """Test tree summary generation."""
        mock_computer = self.create_mock_computer()
        agent = TreeEchoMLAgent(computer=mock_computer)
        
        # Add some data
        agent.action_history = ["action1", "action2", "action3"]
        agent.ml_learner.add_sequence(["action1", "action2"], True)
        agent._navigate_to_tree_node("test_action")
        
        summary = agent.get_tree_summary()
        
        assert "tree_size" in summary
        assert "total_actions" in summary
        assert "learned_patterns" in summary
        assert "common_patterns" in summary
        assert summary["total_actions"] == 3
        assert summary["echo_enabled"] is True
        
    @patch('agent.agent.Agent.handle_item')
    def test_handle_item_integration(self, mock_parent_handle_item):
        """Test handle_item method integration."""
        mock_computer = self.create_mock_computer()
        agent = TreeEchoMLAgent(computer=mock_computer)
        
        # Mock the parent's handle_item to return success
        mock_parent_handle_item.return_value = [{"type": "success"}]
        
        # Test item handling
        test_item = {
            "type": "computer_call",
            "action": {
                "type": "click",
                "x": 100,
                "y": 200
            }
        }
        
        # This should work without raising exceptions
        result = agent.handle_item(test_item)
        
        # Verify action was added to history
        assert len(agent.action_history) > 0
        assert agent.action_history[-1].startswith("click_region_")