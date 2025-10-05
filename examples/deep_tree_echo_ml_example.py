"""
Deep Tree Echo ML Agent Example

This example demonstrates the enhanced Computer Use Agent with:
- Deep tree organization of interactions
- ML-based pattern learning and prediction
- Echo functionality that learns from user patterns
- Intelligent action suggestion based on learned patterns
"""

from agent import TreeEchoMLAgent
from computers.default import LocalPlaywrightBrowser


def main():
    """Run the Deep Tree Echo ML Agent example."""
    print("🌳 Starting Deep Tree Echo ML Agent Example")
    print("This agent learns from your interactions and provides intelligent echoing.")
    print("Type 'tree_summary' to see learning statistics")
    print("Type 'exit' to quit\n")
    
    with LocalPlaywrightBrowser() as computer:
        # Initialize the enhanced agent
        agent = TreeEchoMLAgent(
            computer=computer,
            tree_file_path="/tmp/deep_tree_echo_example.json"
        )
        
        # Add some initial context
        items = [
            {
                "role": "developer",
                "content": (
                    "You are an enhanced CUA with deep tree memory and ML learning. "
                    "Learn patterns from user interactions and provide intelligent suggestions. "
                    "Navigate to websites and interact naturally while building your knowledge tree."
                ),
            }
        ]
        
        # Start with bing.com
        try:
            agent.computer.goto("https://bing.com")
        except Exception as e:
            print(f"Note: Could not navigate to bing.com: {e}")
        
        while True:
            try:
                user_input = input("> ")
                
                if user_input.lower() == "exit":
                    break
                elif user_input.lower() == "tree_summary":
                    agent.print_tree_summary()
                    continue
                elif user_input.lower().startswith("echo"):
                    # Toggle echo functionality
                    if "off" in user_input.lower():
                        agent.echo_enabled = False
                        print("🔕 Echo disabled")
                    elif "on" in user_input.lower():
                        agent.echo_enabled = True
                        print("🔔 Echo enabled")
                    else:
                        print(f"🔔 Echo is {'enabled' if agent.echo_enabled else 'disabled'}")
                    continue
                    
            except EOFError:
                break
                
            items.append({"role": "user", "content": user_input})
            
            try:
                output_items = agent.run_full_turn(
                    items,
                    print_steps=True,
                    show_images=False,
                    debug=False
                )
                items += output_items
                
                # Show learning progress every few interactions
                if len(agent.action_history) % 5 == 0 and len(agent.action_history) > 0:
                    summary = agent.get_tree_summary()
                    print(f"\n📈 Learning Progress: {summary['learned_patterns']} patterns, "
                          f"{summary['tree_size']} tree nodes")
                    
            except Exception as e:
                print(f"Error during interaction: {e}")
                
        # Final summary
        print("\n🏁 Session completed!")
        agent.print_tree_summary()
        agent.save_tree_data()


if __name__ == "__main__":
    main()