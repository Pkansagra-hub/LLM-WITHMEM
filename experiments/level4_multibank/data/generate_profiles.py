"""
Structured profile generator for Level 4 multi-bank training.

Each profile has facts tagged by memory type:
  episodic   — time-stamped events, experiences
  semantic   — stable facts, knowledge
  procedural — behavioral patterns, style
  emotional  — valence associations
  prospective — future intents, pending tasks

Output: JSON files with structured profiles + typed queries.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Vocabulary pools ──────────────────────────────────────────────────

NAMES = [
    "Alice Chen",
    "Bob Martinez",
    "Priya Sharma",
    "Marcus Johnson",
    "Yuki Tanaka",
    "Fatima Al-Hassan",
    "Liam O'Brien",
    "Sofia Rossi",
    "David Kim",
    "Elena Volkov",
    "James Wright",
    "Aisha Patel",
    "Carlos Rivera",
    "Hannah Mueller",
    "Raj Kapoor",
    "Emily Watson",
    "Omar Farouk",
    "Mei Lin",
    "Patrick Okafor",
    "Sarah Cohen",
    "Thomas Berg",
    "Zara Ahmed",
    "Lucas Ferreira",
    "Nina Johansson",
    "Wei Zhang",
    "Olivia Brown",
    "Hassan Demir",
    "Chloe Dubois",
    "Kevin Nakamura",
    "Isla MacLeod",
]

CITIES = [
    "Berlin",
    "Tokyo",
    "San Francisco",
    "London",
    "Mumbai",
    "Paris",
    "Seoul",
    "Toronto",
    "Sydney",
    "Amsterdam",
    "Singapore",
    "New York",
    "Lisbon",
    "Dubai",
    "Stockholm",
    "Buenos Aires",
    "Bangkok",
    "Cape Town",
    "Barcelona",
    "Chicago",
]

PROFESSIONS = [
    "software engineer",
    "data scientist",
    "product manager",
    "UX designer",
    "machine learning researcher",
    "DevOps engineer",
    "technical writer",
    "backend developer",
    "frontend developer",
    "systems architect",
    "security analyst",
    "mobile developer",
    "QA engineer",
    "team lead",
]

COMPANIES = [
    "a startup",
    "Google",
    "a fintech company",
    "a healthcare platform",
    "a remote-first agency",
    "an AI lab",
    "a consulting firm",
    "a gaming studio",
    "an e-commerce company",
    "a biotech firm",
]

LANGUAGES = ["Python", "Rust", "TypeScript", "Go", "Java", "C++", "Kotlin", "Swift"]
FOODS = ["peanuts", "shellfish", "dairy", "gluten", "soy", "eggs", "tree nuts"]
HOBBIES = [
    "hiking",
    "photography",
    "cooking",
    "reading sci-fi",
    "playing guitar",
    "swimming",
    "painting",
    "running",
    "chess",
    "gardening",
    "cycling",
    "board games",
    "writing poetry",
    "yoga",
    "woodworking",
]
TOPICS = [
    "machine learning",
    "distributed systems",
    "game development",
    "climate science",
    "space exploration",
    "philosophy",
    "music theory",
    "quantum computing",
    "history",
    "economics",
    "neuroscience",
    "robotics",
    "cryptography",
]
NEGATIVE_TOPICS = [
    "bureaucracy",
    "slow internet",
    "meetings that could have been emails",
    "unclear requirements",
    "micromanagement",
    "commuting",
    "spam emails",
]
SKILLS = [
    "Rust",
    "Kubernetes",
    "French",
    "piano",
    "public speaking",
    "data visualization",
    "3D printing",
    "sign language",
    "sailing",
]
TIME_AGO = ["last week", "two weeks ago", "last month", "three months ago", "yesterday"]
FUTURE_TIME = [
    "next month",
    "next quarter",
    "by end of year",
    "this summer",
    "by March",
]

# ── Fact Templates ────────────────────────────────────────────────────

EPISODIC_TEMPLATES = [
    "Visited {city} {time_ago}",
    "Had a job interview at {company} {time_ago}",
    "Started learning {skill} {time_ago}",
    "Went {hobby} in {city} {time_ago}",
    "Attended a conference about {topic} {time_ago}",
    "Moved to {city} {time_ago}",
    "Lost their phone {time_ago}",
    "Had a great dinner with friends {time_ago}",
    "Got promoted at work {time_ago}",
    "Adopted a pet {time_ago}",
    "Finished reading a book about {topic} {time_ago}",
    "Traveled to {city} for vacation {time_ago}",
]

SEMANTIC_TEMPLATES = [
    "Works as a {profession} at {company}",
    "Lives in {city}",
    "Allergic to {food}",
    "Prefers {language} for programming",
    "Enjoys {hobby} in their free time",
    "Has a degree in {topic}",
    "Speaks English and {language_natural}",
    "Is {age} years old",
    "Has {family_status}",
    "Favorite cuisine is {cuisine}",
]

PROCEDURAL_TEMPLATES = [
    "Prefers concise bullet-point answers",
    "Usually works late at night",
    "Likes code examples with comments",
    "Prefers formal tone in communication",
    "Asks follow-up questions after explanations",
    "Prefers step-by-step instructions",
    "Likes summaries before detailed explanations",
    "Uses dark mode everywhere",
    "Prefers terminal over GUI tools",
    "Likes to start meetings with small talk",
]

EMOTIONAL_TEMPLATES = [
    "Gets excited about {topic}",
    "Frustrated with {negative}",
    "Nostalgic about their college days",
    "Anxious about upcoming deadlines",
    "Happy when working on {hobby}",
    "Stressed about work-life balance",
    "Enthusiastic about {topic} discussions",
    "Calm and collected during code reviews",
    "Passionate about {hobby}",
    "Uncomfortable with public presentations",
]

PROSPECTIVE_TEMPLATES = [
    "Planning to learn {skill} {future_time}",
    "Wants to run a marathon {future_time}",
    "Needs to renew passport before summer",
    "Planning a trip to {city} {future_time}",
    "Wants to switch to {profession} {future_time}",
    "Has a dentist appointment {future_time}",
    "Planning to start a blog about {topic}",
    "Needs to finish a project at work {future_time}",
    "Wants to read more about {topic}",
    "Planning to move to {city} {future_time}",
]

NATURAL_LANGUAGES = [
    "Spanish",
    "French",
    "Mandarin",
    "Japanese",
    "German",
    "Korean",
    "Hindi",
    "Arabic",
]
AGES = list(range(22, 55))
FAMILY_STATUSES = [
    "a partner",
    "two kids",
    "a cat and a dog",
    "three siblings",
    "no dependents",
]
CUISINES = [
    "Italian",
    "Japanese",
    "Mexican",
    "Indian",
    "Thai",
    "Korean",
    "Mediterranean",
    "Ethiopian",
]


def _fill_template(template: str) -> str:
    """Fill a template with random vocabulary."""
    result = template
    # Single-pass replacement to avoid double-filling
    replacements = {
        "{city}": random.choice(CITIES),
        "{profession}": random.choice(PROFESSIONS),
        "{company}": random.choice(COMPANIES),
        "{language}": random.choice(LANGUAGES),
        "{food}": random.choice(FOODS),
        "{hobby}": random.choice(HOBBIES),
        "{topic}": random.choice(TOPICS),
        "{negative}": random.choice(NEGATIVE_TOPICS),
        "{skill}": random.choice(SKILLS),
        "{time_ago}": random.choice(TIME_AGO),
        "{future_time}": random.choice(FUTURE_TIME),
        "{language_natural}": random.choice(NATURAL_LANGUAGES),
        "{age}": str(random.choice(AGES)),
        "{family_status}": random.choice(FAMILY_STATUSES),
        "{cuisine}": random.choice(CUISINES),
    }
    for key, val in replacements.items():
        result = result.replace(key, val, 1)
    return result


# ── Profile generation ────────────────────────────────────────────────


@dataclass
class Profile:
    name: str
    episodic: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)
    procedural: list[str] = field(default_factory=list)
    emotional: list[str] = field(default_factory=list)
    prospective: list[str] = field(default_factory=list)

    @property
    def profile_text(self) -> str:
        """Full profile with section markers for encoder input."""
        sections = []
        sections.append(f"User: {self.name}.")
        if self.episodic:
            sections.append("[EPISODIC] " + ". ".join(self.episodic) + ".")
        if self.semantic:
            sections.append("[SEMANTIC] " + ". ".join(self.semantic) + ".")
        if self.procedural:
            sections.append("[PROCEDURAL] " + ". ".join(self.procedural) + ".")
        if self.emotional:
            sections.append("[EMOTIONAL] " + ". ".join(self.emotional) + ".")
        if self.prospective:
            sections.append("[PROSPECTIVE] " + ". ".join(self.prospective) + ".")
        return " ".join(sections)

    def get_facts(self, types: list[str]) -> list[str]:
        """Get facts from specified memory types."""
        facts = []
        for t in types:
            facts.extend(getattr(self, t, []))
        return facts


def generate_profile() -> Profile:
    """Generate a single structured profile with facts for all 5 memory types."""
    name = random.choice(NAMES)

    episodic = [
        _fill_template(t)
        for t in random.sample(EPISODIC_TEMPLATES, k=random.randint(3, 5))
    ]
    semantic = [
        _fill_template(t)
        for t in random.sample(SEMANTIC_TEMPLATES, k=random.randint(3, 5))
    ]
    procedural = [
        _fill_template(t)
        for t in random.sample(PROCEDURAL_TEMPLATES, k=random.randint(2, 3))
    ]
    emotional = [
        _fill_template(t)
        for t in random.sample(EMOTIONAL_TEMPLATES, k=random.randint(2, 4))
    ]
    prospective = [
        _fill_template(t)
        for t in random.sample(PROSPECTIVE_TEMPLATES, k=random.randint(2, 3))
    ]

    return Profile(
        name=name,
        episodic=episodic,
        semantic=semantic,
        procedural=procedural,
        emotional=emotional,
        prospective=prospective,
    )


# ── Query templates (tagged with relevant memory types) ───────────────

QUERY_TEMPLATES = [
    # Semantic-focused
    {"text": "Where do I live?", "relevant_types": ["semantic"]},
    {"text": "What do I do for work?", "relevant_types": ["semantic"]},
    {"text": "What am I allergic to?", "relevant_types": ["semantic"]},
    {"text": "What programming language do I prefer?", "relevant_types": ["semantic"]},
    {"text": "Tell me about myself.", "relevant_types": ["semantic"]},
    {"text": "What are my hobbies?", "relevant_types": ["semantic"]},
    # Episodic-focused
    {"text": "What did I do recently?", "relevant_types": ["episodic"]},
    {"text": "Tell me about my recent activities.", "relevant_types": ["episodic"]},
    {"text": "Have I traveled anywhere lately?", "relevant_types": ["episodic"]},
    {
        "text": "What events happened in my life recently?",
        "relevant_types": ["episodic"],
    },
    # Emotional-focused
    {"text": "What topics make me excited?", "relevant_types": ["emotional"]},
    {"text": "What frustrates me?", "relevant_types": ["emotional"]},
    {"text": "How am I feeling about things?", "relevant_types": ["emotional"]},
    # Procedural-focused
    {"text": "How should you talk to me?", "relevant_types": ["procedural"]},
    {"text": "What communication style do I prefer?", "relevant_types": ["procedural"]},
    {"text": "How do I like to work?", "relevant_types": ["procedural"]},
    # Prospective-focused
    {"text": "What are my upcoming plans?", "relevant_types": ["prospective"]},
    {"text": "Am I forgetting to do anything?", "relevant_types": ["prospective"]},
    {"text": "What goals do I have?", "relevant_types": ["prospective"]},
    # Multi-type queries
    {
        "text": "What should I eat for dinner?",
        "relevant_types": ["semantic", "emotional"],
    },
    {
        "text": "Suggest a weekend activity for me.",
        "relevant_types": ["semantic", "emotional", "episodic"],
    },
    {"text": "Help me plan my week.", "relevant_types": ["prospective", "procedural"]},
    {
        "text": "What should I learn next?",
        "relevant_types": ["semantic", "prospective"],
    },
    {"text": "Recommend a book for me.", "relevant_types": ["semantic", "emotional"]},
    {
        "text": "How can I be more productive?",
        "relevant_types": ["procedural", "emotional"],
    },
    {
        "text": "What have I been up to and what's next?",
        "relevant_types": ["episodic", "prospective"],
    },
    {
        "text": "Help me prepare for my upcoming trip.",
        "relevant_types": ["prospective", "semantic", "episodic"],
    },
    {
        "text": "What would make me happy right now?",
        "relevant_types": ["emotional", "semantic"],
    },
    {
        "text": "Summarize what you know about me.",
        "relevant_types": ["semantic", "emotional", "procedural"],
    },
]


# ── Dataset entry generation ──────────────────────────────────────────


@dataclass
class TrainingExample:
    """One (profile, query) pair for training."""

    profile_text: str  # full profile with all memory types (encoder input)
    query_text: str  # raw query (query encoder input)
    relevant_types: list[str]  # which memory types are relevant
    relevant_facts: list[str]  # facts from relevant types (for gold prompt)
    all_facts: list[str]  # all facts across all types
    name: str


def make_example(profile: Profile, query_template: dict) -> TrainingExample:
    """Combine a profile and query template into a training example."""
    relevant_facts = profile.get_facts(query_template["relevant_types"])
    all_facts = profile.get_facts(
        ["episodic", "semantic", "procedural", "emotional", "prospective"]
    )

    return TrainingExample(
        profile_text=profile.profile_text,
        query_text=query_template["text"],
        relevant_types=query_template["relevant_types"],
        relevant_facts=relevant_facts,
        all_facts=all_facts,
        name=profile.name,
    )


def generate_dataset(num_profiles: int, seed: int = 42) -> list[dict]:
    """Generate a full dataset: each profile paired with all query templates."""
    random.seed(seed)
    profiles = [generate_profile() for _ in range(num_profiles)]
    examples = []

    for profile in profiles:
        for qt in QUERY_TEMPLATES:
            ex = make_example(profile, qt)
            examples.append(asdict(ex))

    random.shuffle(examples)
    return examples


# ── CLI entry point ───────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate structured training data")
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.num_train} train profiles × {len(QUERY_TEMPLATES)} queries..."
    )
    train_data = generate_dataset(args.num_train, seed=42)
    with open(out / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"  → {len(train_data)} train examples saved")

    print(f"Generating {args.num_val} val profiles × {len(QUERY_TEMPLATES)} queries...")
    val_data = generate_dataset(args.num_val, seed=123)
    with open(out / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    print(f"  → {len(val_data)} val examples saved")


if __name__ == "__main__":
    main()
