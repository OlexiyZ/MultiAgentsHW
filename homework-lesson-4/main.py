from agent import agent


def main() -> None:
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = agent.invoke(user_input)
        except Exception as exc:
            print(f"\nAgent error: {exc}")
            continue

        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
