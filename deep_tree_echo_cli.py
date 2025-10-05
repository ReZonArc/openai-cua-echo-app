"""
Enhanced CLI for Deep Tree Echo ML Agent

This CLI provides access to the Deep Tree Echo ML Agent functionality,
allowing users to interact with an AI agent that learns patterns and
organizes interactions in a hierarchical tree structure.
"""

import argparse
from agent import TreeEchoMLAgent
from computers.config import *
from computers.default import *
from computers import computers_config


def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response.lower().strip() == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Deep Tree Echo ML Computer Use Agent - Enhanced AI agent with learning capabilities."
    )
    parser.add_argument(
        "--computer",
        choices=computers_config.keys(),
        help="Choose the computer environment to use.",
        default="local-playwright",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Initial input to use instead of asking the user.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images during the execution.",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        help="Start the browsing session with a specific URL (only for browser environments).",
        default="https://bing.com",
    )
    parser.add_argument(
        "--tree-file",
        type=str,
        help="Path to save/load the tree learning data.",
        default="/tmp/deep_tree_echo_data.json",
    )
    parser.add_argument(
        "--echo-threshold",
        type=float,
        help="Confidence threshold for echo predictions (0.0-1.0).",
        default=0.7,
    )
    parser.add_argument(
        "--disable-echo",
        action="store_true",
        help="Disable echo functionality.",
    )
    
    args = parser.parse_args()
    ComputerClass = computers_config[args.computer]

    print("🌳 Deep Tree Echo ML Agent Starting")
    print(f"   Computer: {args.computer}")
    print(f"   Echo: {'Disabled' if args.disable_echo else 'Enabled'}")
    print(f"   Tree File: {args.tree_file}")
    print(f"   Echo Threshold: {args.echo_threshold}")
    print()
    print("Special commands:")
    print("  - 'tree_summary' - Show learning statistics")
    print("  - 'echo on/off' - Toggle echo functionality")
    print("  - 'exit' - Quit the application")
    print()

    with ComputerClass() as computer:
        agent = TreeEchoMLAgent(
            computer=computer,
            acknowledge_safety_check_callback=acknowledge_safety_check_callback,
            tree_file_path=args.tree_file,
        )
        
        # Configure echo settings
        agent.echo_enabled = not args.disable_echo
        agent.echo_threshold = args.echo_threshold
        
        items = [
            {
                "role": "developer",
                "content": (
                    "You are an enhanced Computer Use Agent with deep tree memory and ML learning capabilities. "
                    "You learn patterns from user interactions and provide intelligent suggestions. "
                    "When you recognize successful patterns from your learning, mention them to the user. "
                    "Navigate websites and interact naturally while building your knowledge tree."
                ),
            }
        ]

        if args.computer in ["browserbase", "local-playwright"]:
            if not args.start_url.startswith("http"):
                args.start_url = "https://" + args.start_url
            try:
                agent.computer.goto(args.start_url)
                print(f"🌐 Navigated to {args.start_url}")
            except Exception as e:
                print(f"⚠️  Could not navigate to {args.start_url}: {e}")

        while True:
            try:
                user_input = args.input or input("> ")
                if user_input == "exit":
                    break
                elif user_input == "tree_summary":
                    agent.print_tree_summary()
                    continue
                elif user_input.startswith("echo"):
                    if "off" in user_input.lower():
                        agent.echo_enabled = False
                        print("🔕 Echo disabled")
                    elif "on" in user_input.lower():
                        agent.echo_enabled = True
                        print("🔔 Echo enabled")
                    else:
                        print(f"🔔 Echo is {'enabled' if agent.echo_enabled else 'disabled'}")
                    continue
                    
            except EOFError as e:
                print(f"An error occurred: {e}")
                break
                
            items.append({"role": "user", "content": user_input})
            
            try:
                output_items = agent.run_full_turn(
                    items,
                    print_steps=True,
                    show_images=args.show,
                    debug=args.debug,
                )
                items += output_items
                
                # Show learning progress periodically
                if len(agent.action_history) % 5 == 0 and len(agent.action_history) > 0:
                    summary = agent.get_tree_summary()
                    print(f"\n📈 Learning Progress: {summary['learned_patterns']} patterns, "
                          f"{summary['tree_size']} tree nodes")
                    
            except Exception as e:
                print(f"Error during interaction: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                    
            args.input = None

        # Final summary and save
        print("\n🏁 Session completed!")
        agent.print_tree_summary()
        agent.save_tree_data()
        print(f"💾 Learning data saved to {args.tree_file}")


if __name__ == "__main__":
    main()