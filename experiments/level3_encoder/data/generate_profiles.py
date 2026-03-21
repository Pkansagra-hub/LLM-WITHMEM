"""
Synthetic profile generator — template-based, zero API cost.

Generates diverse user profiles with testable attributes:
  location, profession, food preferences, hobbies, pets, family, quirks.
"""

import json
import random
from pathlib import Path

# ── Building blocks ──────────────────────────────────────────────────────

FIRST_NAMES = [
    "Alex",
    "Priya",
    "Marcus",
    "Yuki",
    "Fatima",
    "Liam",
    "Sofia",
    "Chen",
    "Amara",
    "Diego",
    "Emma",
    "Raj",
    "Hana",
    "Omar",
    "Zoe",
    "Kenji",
    "Isabella",
    "Kwame",
    "Mia",
    "Arjun",
    "Elena",
    "Tobias",
    "Nina",
    "Samuel",
    "Leila",
    "Viktor",
    "Aisha",
    "Lucas",
    "Sakura",
    "Mateo",
    "Chloe",
    "Hassan",
    "Maya",
    "Felix",
    "Anya",
    "Rafael",
    "Suki",
    "Gabriel",
    "Ingrid",
    "Dante",
    "Nadia",
    "Oscar",
    "Thalia",
    "Bjorn",
    "Carmen",
    "Idris",
    "Freya",
    "Ravi",
    "Petra",
    "Jamal",
]

CITIES = [
    ("Seattle", "Washington"),
    ("Bangalore", "India"),
    ("Austin", "Texas"),
    ("Tokyo", "Japan"),
    ("London", "UK"),
    ("Paris", "France"),
    ("Berlin", "Germany"),
    ("Sydney", "Australia"),
    ("Toronto", "Canada"),
    ("Mumbai", "India"),
    ("São Paulo", "Brazil"),
    ("Cape Town", "South Africa"),
    ("Stockholm", "Sweden"),
    ("Seoul", "South Korea"),
    ("Dubai", "UAE"),
    ("Mexico City", "Mexico"),
    ("Amsterdam", "Netherlands"),
    ("Singapore", "Singapore"),
    ("Bangkok", "Thailand"),
    ("Istanbul", "Turkey"),
    ("Nairobi", "Kenya"),
    ("Dublin", "Ireland"),
    ("Lisbon", "Portugal"),
    ("Oslo", "Norway"),
    ("Denver", "Colorado"),
    ("Portland", "Oregon"),
    ("Chicago", "Illinois"),
    ("Boston", "Massachusetts"),
    ("New York", "New York"),
    ("San Francisco", "California"),
    ("Vancouver", "Canada"),
    ("Melbourne", "Australia"),
    ("Edinburgh", "Scotland"),
    ("Barcelona", "Spain"),
    ("Vienna", "Austria"),
    ("Zurich", "Switzerland"),
    ("Helsinki", "Finland"),
    ("Warsaw", "Poland"),
    ("Prague", "Czech Republic"),
    ("Budapest", "Hungary"),
]

PROFESSIONS = [
    "software engineer",
    "marine biologist",
    "retired firefighter",
    "concert pianist",
    "pediatric surgeon",
    "high school teacher",
    "graphic designer",
    "data scientist",
    "chef",
    "architect",
    "journalist",
    "veterinarian",
    "civil engineer",
    "pharmacist",
    "photographer",
    "social worker",
    "mechanical engineer",
    "librarian",
    "dentist",
    "pilot",
    "lawyer",
    "nurse",
    "archaeologist",
    "electrician",
    "financial analyst",
    "physical therapist",
    "urban planner",
    "marine engineer",
    "botanist",
    "forensic scientist",
    "translator",
    "baker",
    "museum curator",
    "wildlife biologist",
    "sports coach",
    "carpenter",
    "psychologist",
    "astronomer",
    "winemaker",
    "blockchain developer",
]

FOODS_LOVE = [
    "Thai food",
    "sushi",
    "Italian pasta",
    "Mexican tacos",
    "Korean BBQ",
    "Ethiopian injera",
    "French pastries",
    "Indian curry",
    "Vietnamese pho",
    "Peruvian ceviche",
    "Greek souvlaki",
    "Japanese ramen",
    "Moroccan tagine",
    "Cajun gumbo",
    "Turkish kebab",
    "dim sum",
    "Neapolitan pizza",
    "Argentine empanadas",
    "Lebanese falafel",
    "Jamaican jerk chicken",
]

FOODS_HATE = [
    "cilantro",
    "olives",
    "blue cheese",
    "anchovies",
    "liver",
    "raw onions",
    "pickles",
    "mushrooms",
    "beets",
    "truffle oil",
    "licorice",
    "eggplant",
    "okra",
    "durian",
    "cottage cheese",
]

HOBBIES = [
    "kayaking",
    "hiking",
    "woodworking",
    "playing piano",
    "painting watercolors",
    "rock climbing",
    "birdwatching",
    "pottery",
    "running marathons",
    "gardening",
    "playing chess",
    "photography",
    "surfing",
    "knitting",
    "mountain biking",
    "stargazing",
    "writing poetry",
    "playing guitar",
    "cooking",
    "scuba diving",
    "fencing",
    "origami",
    "beekeeping",
    "sailing",
    "sketching",
    "yoga",
    "dancing salsa",
    "restoring vintage cars",
    "collecting vinyl records",
    "brewing craft beer",
    "playing tennis",
    "bouldering",
    "calligraphy",
    "building model trains",
    "foraging wild mushrooms",
]

PETS = [
    ("dog", "Biscuit"),
    ("dog", "Luna"),
    ("dog", "Max"),
    ("cat", "Mochi"),
    ("cat", "Shadow"),
    ("cat", "Cleo"),
    ("rabbit", "Pepper"),
    ("parrot", "Rio"),
    ("turtle", "Sheldon"),
    ("goldfish", "Bubbles"),
    ("hamster", "Nugget"),
    (None, None),  # no pet
    (None, None),
    (None, None),
]

WEEKEND_ACTIVITIES = [
    "volunteers at a local animal shelter",
    "coaches little league baseball",
    "visits farmers markets",
    "explores bookstores",
    "goes to flea markets",
    "practices martial arts",
    "mentors young students",
    "leads community clean-up events",
    "attends live music shows",
    "visits art galleries",
    "does trail running",
    "teaches coding workshops",
    "goes fishing",
    "reads at coffee shops",
    "does crossword puzzles in the park",
    "volunteers at the local food bank",
    "goes to open mic nights",
    "attends pottery classes",
]

QUIRKS = [
    "speaks four languages",
    "is training for a triathlon",
    "collects vintage maps",
    "has a blog about urban farming",
    "makes homemade pasta every Sunday",
    "is learning to play the accordion",
    "has visited 40 countries",
    "keeps a daily journal",
    "practices meditation every morning",
    "is writing a novel",
    "makes their own hot sauce",
    "is obsessed with true crime podcasts",
    "reads a book a week",
    "builds mechanical keyboards",
    "collects rare succulents",
    "does stand-up comedy on weekends",
    "is a certified sommelier",
    "hand-roasts their own coffee beans",
    "restores antique furniture",
    "studies ancient history",
]


def generate_one_profile(rng: random.Random) -> dict:
    """Generate a single synthetic profile with testable attributes."""
    name = rng.choice(FIRST_NAMES)
    city, region = rng.choice(CITIES)
    profession = rng.choice(PROFESSIONS)
    food_love = rng.choice(FOODS_LOVE)
    food_hate = rng.choice(FOODS_HATE)
    hobby1, hobby2 = rng.sample(HOBBIES, 2)
    pet_type, pet_name = rng.choice(PETS)
    weekend = rng.choice(WEEKEND_ACTIVITIES)
    quirk = rng.choice(QUIRKS)

    # Build profile text (2-4 sentences)
    sentences = [
        f"{name} is a {profession} living in {city}, {region} who loves {food_love} and hates {food_hate}.",
    ]
    sentences.append(f"{name} enjoys {hobby1} and {hobby2} in their free time.")
    if pet_type:
        sentences.append(f"{name} has a {pet_type} named {pet_name}.")
    sentences.append(f"{name} {weekend} and {quirk}.")

    text = " ".join(sentences)

    # Build keyword list for evaluation
    keywords = [
        city.lower(),
        region.lower() if region != city else "",
        profession.split()[-1].lower(),  # e.g. "engineer", "biologist"
        food_love.split()[-1].lower(),  # e.g. "food", "sushi", "tacos"
        hobby1.split()[-1].lower(),  # e.g. "kayaking", "piano"
        hobby2.split()[-1].lower(),
    ]
    if pet_type:
        keywords.append(pet_name.lower())
        keywords.append(pet_type.lower())

    # Remove empty strings
    keywords = [k for k in keywords if k]

    return {
        "name": name,
        "city": city,
        "region": region,
        "profession": profession,
        "food_love": food_love,
        "food_hate": food_hate,
        "hobbies": [hobby1, hobby2],
        "pet": {"type": pet_type, "name": pet_name} if pet_type else None,
        "weekend_activity": weekend,
        "quirk": quirk,
        "text": text,
        "keywords": keywords,
    }


def generate_profiles(n: int, seed: int = 42) -> list[dict]:
    """Generate n unique synthetic profiles."""
    rng = random.Random(seed)
    profiles = []
    seen_names_cities = set()

    attempts = 0
    while len(profiles) < n and attempts < n * 10:
        p = generate_one_profile(rng)
        key = (p["name"], p["city"])
        if key not in seen_names_cities:
            seen_names_cities.add(key)
            profiles.append(p)
        attempts += 1

    return profiles


QUERY_TEMPLATES = [
    "What restaurant should I go to tonight?",
    "What's a good hobby to pick up?",
    "What should I do this weekend?",
    "What gift should I get for a friend?",
    "Tell me something interesting about where I live.",
    "What book should I read next?",
    "What should I cook for dinner?",
    "Suggest a vacation destination for me.",
    "What's a good workout routine for me?",
    "What podcast should I listen to?",
    "What's a fun date night idea?",
    "How should I decorate my living room?",
    "What skill should I learn next?",
    "What movie should I watch tonight?",
    "What's a good way to relax after work?",
    "What should I do for my birthday?",
    "Suggest a weekend project for me.",
    "What's a good gift for my partner?",
    "What local event should I check out?",
    "How can I make new friends in my area?",
    "What should I bring to a potluck?",
    "What's a good morning routine for me?",
    "Suggest a creative outlet for me.",
    "What kind of pet should I get?",
    "What's a good side hustle for someone like me?",
    "What should I plant in my garden?",
    "What music should I listen to?",
    "Suggest a day trip from where I live.",
    "What's a good way to give back to my community?",
    "What should I do on a rainy day?",
    "What's a fun activity I can do alone?",
    "Suggest a recipe I might enjoy.",
    "What's a good conversation starter at a party?",
    "How should I spend my lunch break?",
    "What's a good New Year's resolution for me?",
    "What should I add to my bucket list?",
    "Suggest a board game for me and friends.",
    "What's a good way to start my morning?",
    "What should I wear to a casual dinner?",
    "Suggest a documentary I'd find interesting.",
    "What's a good way to celebrate a small win?",
    "How can I make my commute more enjoyable?",
    "What should I do on my next day off?",
    "Suggest a café or hangout spot for me.",
    "What's a good icebreaker for a new group?",
    "How can I improve my work-life balance?",
    "What's a fun challenge I should try?",
    "Suggest a DIY project for my home.",
    "What should I do to de-stress this evening?",
    "What's something new I should try this month?",
]


def main():
    """Generate and save training/validation profile sets."""
    out_dir = Path(__file__).parent

    train_profiles = generate_profiles(1000, seed=42)
    val_profiles = generate_profiles(200, seed=9999)

    train_path = out_dir / "profiles_train.json"
    val_path = out_dir / "profiles_val.json"
    queries_path = out_dir / "queries.json"

    train_path.write_text(json.dumps(train_profiles, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_profiles, indent=2), encoding="utf-8")
    queries_path.write_text(json.dumps(QUERY_TEMPLATES, indent=2), encoding="utf-8")

    print(f"Generated {len(train_profiles)} training profiles -> {train_path}")
    print(f"Generated {len(val_profiles)} validation profiles -> {val_path}")
    print(f"Saved {len(QUERY_TEMPLATES)} query templates -> {queries_path}")

    # Quick stats
    cities = {p["city"] for p in train_profiles}
    profs = {p["profession"] for p in train_profiles}
    print(f"Unique cities: {len(cities)}, unique professions: {len(profs)}")


if __name__ == "__main__":
    main()
