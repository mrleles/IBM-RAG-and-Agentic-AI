from shared_functions import *

food_items = []

def main():
    try:
        print("Interactive Food Recommendation system")
        print("=" * 50)
        print("Loading food database...")

        global food_items
        food_items = load_food_data('./FoodDataSet.json')
        print(f"Loaded {len(food_items)} food items successfully")

        collection = create_similarity_search_collection(
            "interactive_food_search",
            {'description': 'A collection for interactive food search'}
        )
        populate_similarity_collection(collection, food_items)

        interactive_food_chatbot(collection)

    except Exception as error:
        print(f"Error initializing system: {error}")

def interactive_food_chatbot(collection):
    """Interactive CLI chatbot for food recommendations"""
    print("\n" + "="*50)
    print("INTERACTIVE FOOD SEARCH CHATBOT")
    print("="*50)
    print("Commands:")
    print(" - Type any food name or description to search")
    print(" - 'help' - Show availale commands")
    print(" - 'quit' or 'exit' - Exit the system")
    print(" - Ctrl+C - Emergency exit")
    print("="*50)

    while True:
        try:
            user_input = input("\n Search for food: ").strip()

            if not user_input:
                print(" Please enter a search term or 'help' for commands")
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Food Recommendation system!")
                print(" Goodbye!")
                break

            elif user_input.lower() in ['help', 'h']:
                show_help_menu()

            else:
                handle_food_search(collection, user_input)

        except KeyboardInterrupt:
            print("\n\nSystem interrupted. Goodbye!")
            break

        except Exception as e:
            print(f"Error processing request: {e}")

def show_help_menu():
    """Display help information for user"""
    print("\nHELP MENU")
    print("="*30)
    print("Search Examples:")
    print(" - 'chocolate dessert' - Find chocolate desserts")
    print(" - 'Italian food' - Find Italian cuisine")
    print(" - 'sweet treats' - Find sweet desserts")
    print(" - 'baked goods' - Find baked items")
    print(" - 'low calorie' - Find lower-calorie options")
    print("\nCommands:")
    print(" - 'help' - Show this help menu")
    print(" - 'quit' - Exit the system")

def handle_food_search(collection, query):
    """Handle food similarity search with enhanced display"""
    print("\n Searching for '{query}'...")
    print(" Please wait...")

    results = perform_similarity_search(collection, query, 5)

    if not results:
        print("No matching foods found.")
        print("Try different keywords like:")
        print(" - Cuisine types: 'Italian', 'American'")
        print(" - Ingredients: 'chocolate', 'flour', 'cheese'")
        print(" - Descriptors: 'sweet', 'baked', 'dessert'")
        return

    print(f"\n Found {len(results)} recommendations:")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        percentage_score = result['similarity_score'] * 100

        print(f"\n{i}. {result['food_name']}")
        print(f" Match Score: {percentage_score:1f}%")
        print(f" Cuisine: {result['cuisine_type']}")
        print(f" Calories: {result['food_calories_per_serving']} per serving")
        print(f" Description: {result['food_description']}")

        if i < len(results):
            print("  " + "-" * 50)

    print("=" * 60)

    suggest_related_searches(results)

def suggest_related_searches(results):
    if not results:
        return

    cuisines = list(set([r['cuisine_type'] for r in results]))

    print("\n Related searches you might like:")
    for cuisine in cuisines[:3]:
        print(f" - Try '{cuisine} dishes' for more {cuisine} options")

    avg_calories = sum([r['food_calories_per_serving'] for r in results]) / len(results)
    if avg_calories > 350:
        print(" - Try 'low calorie' for lighter options")
    else:
        print(" - Try 'hearty meal' for more substantial dishes")

if __name__ == "__main__":
    main()