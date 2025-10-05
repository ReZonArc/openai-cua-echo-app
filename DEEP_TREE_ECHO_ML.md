# Deep Tree Echo ML Agent 🌳🤖

## Overview

The Deep Tree Echo ML Agent is an enhanced Computer Use Agent (CUA) that extends the base functionality with:

- **Deep Tree Organization**: Hierarchical organization of user interactions and computer actions
- **Machine Learning Pattern Recognition**: Learns from interaction patterns and predicts successful action sequences
- **Echo Functionality**: Provides intelligent suggestions based on learned patterns
- **Persistent Memory**: Saves and loads learning data across sessions

## Key Features

### 🌳 Deep Tree Structure

The agent organizes all interactions in a hierarchical tree structure where:
- Each node represents an action or interaction
- Nodes track frequency, success rate, and context
- The tree grows organically based on user behavior
- Navigation through the tree enables context-aware responses

### 🧠 ML Pattern Learning

The integrated ML system:
- Tracks action sequences and their outcomes
- Predicts success probability for new sequences
- Identifies common successful patterns
- Provides partial matching for similar scenarios

### 🔊 Echo Functionality

The echo system provides intelligent feedback by:
- Recognizing when current actions match successful patterns
- Suggesting actions based on learned behavior
- Providing confidence scores for predictions
- Adapting to user preferences over time

### 💾 Persistent Learning

All learning data is automatically:
- Saved to JSON files for persistence
- Loaded on agent initialization
- Updated in real-time during interactions
- Exportable for analysis or backup

## Quick Start

### Using the Enhanced CLI

```bash
# Basic usage with default settings
python deep_tree_echo_cli.py

# Customize computer environment and settings
python deep_tree_echo_cli.py --computer local-playwright --show --tree-file my_learning.json

# Disable echo and enable debug mode
python deep_tree_echo_cli.py --disable-echo --debug

# Set custom echo threshold (0.0-1.0)
python deep_tree_echo_cli.py --echo-threshold 0.8
```

### CLI Commands

During interaction, you can use these special commands:

- `tree_summary` - Show learning statistics and patterns
- `echo on/off` - Toggle echo functionality
- `exit` - Quit and save learning data

### Using in Code

```python
from agent import TreeEchoMLAgent
from computers import LocalPlaywrightBrowser

with LocalPlaywrightBrowser() as computer:
    agent = TreeEchoMLAgent(
        computer=computer,
        tree_file_path="my_learning_data.json"
    )
    
    # Configure echo settings
    agent.echo_threshold = 0.8
    agent.echo_enabled = True
    
    # Use like regular agent
    items = [{"role": "user", "content": "Navigate to example.com"}]
    result = agent.run_full_turn(items)
    
    # View learning progress
    agent.print_tree_summary()
```

## Architecture

### TreeNode Class

Represents nodes in the interaction tree:

```python
node = TreeNode("click", {"x": 100, "y": 200})
node.update_stats(success=True)  # Track outcomes
child = TreeNode("type", {"text": "hello"})
node.add_child("follow_up", child)
```

### MLPatternLearner Class

Handles pattern recognition and prediction:

```python
learner = MLPatternLearner()
learner.add_sequence(["click", "type", "enter"], success=True)
probability = learner.predict_success(["click", "type"])  # Returns confidence
```

### TreeEchoMLAgent Class

Main agent class extending the base Agent:

```python
agent = TreeEchoMLAgent(computer=computer)
agent.echo_enabled = True
agent.echo_threshold = 0.7  # 70% confidence threshold
```

## Learning Data Format

The agent saves learning data in JSON format:

```json
{
  "tree": {
    "action_type": "root",
    "context": {},
    "frequency": 0,
    "success_rate": 0.0,
    "children": {
      "click_region_1_2": {
        "action_type": "click_region_1_2",
        "frequency": 5,
        "success_rate": 0.8,
        "children": {...}
      }
    }
  },
  "ml_patterns": {
    "frequencies": {
      "click->type->enter": 10,
      "scroll->click": 5
    },
    "success_patterns": {
      "click->type->enter": [true, true, false, true, ...]
    }
  }
}
```

## Advanced Configuration

### Echo Thresholds

- `0.9-1.0`: Very conservative, only echoes highly confident patterns
- `0.7-0.8`: Balanced, good for most use cases  
- `0.5-0.6`: Liberal, provides more suggestions but may be less accurate
- `0.0-0.4`: Very liberal, experimental mode

### Tree Management

The tree automatically manages its size by:
- Limiting maximum depth to prevent overwhelming growth
- Tracking access patterns to prioritize important nodes
- Providing summary statistics for analysis

### Performance Considerations

- Tree data is saved every 10 actions to balance performance and data safety
- Action history is limited to recent interactions to prevent memory bloat
- Pattern matching uses efficient string-based keys for fast lookup

## Example Interactions

### Learning Web Navigation

```
> Navigate to github.com
🔮 Echo: Similar pattern 'click->type->enter' succeeded 85% of the time (n=12)
[Agent performs navigation successfully]

> Search for "openai"
✨ High success pattern detected: type->enter (confidence: 92%)
[Agent learns this search pattern]
```

### Building Complex Workflows

The agent learns multi-step workflows and can predict successful sequences:

```
📊 Top Patterns:
   click_region_2_1->type_short->keypress_Enter (n=15, success=93%)
   scroll_down->click_region_3_2 (n=8, success=75%)
   type_long->scroll_down->click_region_5_1 (n=3, success=100%)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run only Deep Tree Echo ML tests
python -m pytest tests/test_tree_echo_ml_agent.py -v

# Run tests with coverage
python -m pytest tests/test_tree_echo_ml_agent.py --cov=agent.tree_echo_ml_agent
```

## Contributing

When contributing to the Deep Tree Echo ML Agent:

1. Ensure all tests pass: `python -m pytest tests/`
2. Add tests for new functionality
3. Update documentation for new features
4. Consider backward compatibility with existing learning data
5. Test with different computer environments

## Troubleshooting

### Common Issues

**Learning data not persisting:**
- Check write permissions for the tree file path
- Ensure the directory exists for custom tree file paths
- Verify JSON format if loading existing data

**Echo not working:**
- Check if `echo_enabled` is True
- Verify `echo_threshold` is appropriate for your use case
- Ensure sufficient interaction history exists for pattern matching

**Performance issues:**
- Consider reducing tree depth limit
- Clear old learning data if files become too large
- Monitor memory usage with large interaction histories

### Debug Mode

Enable debug mode for detailed insights:

```bash
python deep_tree_echo_cli.py --debug
```

This provides:
- Detailed action key generation
- Tree navigation paths
- ML prediction confidence scores
- Error stack traces

## Future Enhancements

Potential areas for expansion:

- **Advanced ML Models**: Integration with neural networks for more sophisticated pattern recognition
- **Multi-Agent Learning**: Sharing learned patterns between agent instances
- **Visual Pattern Recognition**: Learning from screenshot similarities
- **Natural Language Processing**: Understanding user intent from text input
- **Automated Workflow Generation**: Creating reusable workflows from learned patterns

---

For more examples and detailed API documentation, see the `examples/` directory and inline code documentation.