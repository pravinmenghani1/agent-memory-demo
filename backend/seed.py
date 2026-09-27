"""Seed the demo with a believable history so retrieval has something to find
on the very first run: prior preferences for the demo user, and a small
episodic case log from *other* customers."""
from .memory import LongTermMemory, EpisodicMemory

DEMO_USER = "alex_rivera"


def seed():
    lt = LongTermMemory()
    ep = EpisodicMemory()

    if not lt.all_for_user(DEMO_USER):
        for fact, category in [
            ("Prefers email updates over phone calls", "preference"),
            ("Has a NimbusCart Plus membership (free expedited shipping)", "account_info"),
            ("Owns a NimbusCart SoundBar Pro (purchased 2025-11-02)", "product_info"),
        ]:
            lt.add_fact(DEMO_USER, fact, category)

    if not ep.all_episodes():
        cases = [
            ("Customer's wireless earbuds wouldn't pair with a new phone after a firmware update.",
             "Had the customer hold the case button for 10s to force a factory reset, then re-pair; resolved.",
             "technical", "resolved"),
            ("Customer was charged twice for the same order due to a checkout retry glitch.",
             "Verified the duplicate charge in billing system, issued an immediate refund for the second charge, and applied a 10% goodwill credit.",
             "billing", "resolved"),
            ("Customer's SoundBar Pro arrived with a cracked casing.",
             "Approved a free replacement unit with expedited shipping at no cost, waived the return label.",
             "shipping", "resolved"),
            ("Customer wanted to know why their Plus membership renewal was priced differently than last year.",
             "Explained the annual price adjustment and applied a loyalty discount to match last year's rate.",
             "billing", "resolved"),
        ]
        for issue, resolution, category, outcome in cases:
            ep.log_episode(user_id="other_customer", ticket_id=None, issue=issue,
                            resolution=resolution, category=category, outcome=outcome)


if __name__ == "__main__":
    from .db import init_db
    init_db()
    seed()
    print("Seeded.")
