"""Tests for the deterministic refinement guards (no LLM involved)."""

import pytest

from app.pipeline.guards import check_refinement, normalize_tokens, numbers_in, proper_nouns


# ---------------------------------------------------------------------------
# Normalisation: formatting-only differences must compare equal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("The budget is forty-two thousand dollars.", "The budget is $42,000."),
        ("version one point two nine", "version 1.29"),
        ("Friday, October tenth", "Friday, Oct 10th"),
        ("two thousand and twenty six", "2026"),
        ("a hundred and five people", "105 people"),
        ("We won't change it", "we will not change it"),
        ("we can't", "we cannot"),
        ("twenty five percent", "25%"),
        ("one point five million dollars", "$1.5 million"),
        ("forty k users", "40k users"),
        ("two lakh rupees", "₹2,00,000"),
        ("we're gonna ship", "we are going to ship"),
        ("follow-up", "follow up"),
        ("GPT four", "GPT-4"),
    ],
)
def test_equivalent_forms_normalise_equal(a, b):
    assert normalize_tokens(a) == normalize_tokens(b)


def test_numbers_counted_including_inside_identifiers():
    assert numbers_in(normalize_tokens("k8s v1.29 and 3 nodes")) == {"8": 1, "1.29": 1, "3": 1}


def test_proper_nouns_skip_sentence_starts():
    assert proper_nouns("Priya said ask Rahul. Then we ship on Friday.") == {"rahul"}


# ---------------------------------------------------------------------------
# Acceptable cleanups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("we use cubernetties for deployment", "We use Kubernetes for deployment."),
        ("deploy it on cube nettees", "Deploy it on Kubernetes."),
        ("The budget is forty-two thousand dollars.", "The budget is $42,000."),
        ("migrate to version one point two nine", "Migrate to version 1.29."),
        ("so um we we need the a p i docs", "So we need the API docs."),
        ("deploy it to the s three bucket", "Deploy it to the S3 bucket."),
        ("Ask Pria about it.", "Ask Priya about it."),
        ("the post gres database", "the Postgres database"),
        ("we're gonna use react", "We're going to use React."),
        ("Can we ship on Friday", "Can we ship on Friday?"),
        ("growth was twenty five percent", "Growth was 25%."),
        ("Unchanged sentence.", "Unchanged sentence."),
    ],
)
def test_valid_cleanups_pass(raw, refined):
    assert check_refinement(raw, refined) == []


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def issues_for(raw, refined, changes=None):
    issues = check_refinement(raw, refined, changes)
    assert issues, f"expected rejection: {raw!r} -> {refined!r}"
    return " | ".join(issues)


def test_empty_refined_text():
    assert issues_for("Okay, let's start.", "  ") == "refined text is empty"


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("The budget is forty-two thousand dollars.", "The budget is $24,000."),
        ("We need three servers.", "We need four servers."),
        ("Version 1.29 is out.", "Version 1.30 is out."),
        ("We have ten users.", "We have users."),
    ],
)
def test_number_alteration(raw, refined):
    assert "numbers changed" in issues_for(raw, refined)


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("We will not change the database schema.", "We will change the database schema."),
        ("We can't ship this week.", "We can ship this week."),
        ("We never use that.", "We use that."),
        ("We should change it.", "We should not change it."),
        ("No, we keep it.", "We keep it."),
    ],
)
def test_negation_alteration(raw, refined):
    assert "negation changed" in issues_for(raw, refined)


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("We ship next week.", "We ship next week, on October 10."),
        ("The meeting is on Tuesday.", "The meeting is on Thursday."),
        ("Let's do it soon.", "Let's do it tomorrow."),
    ],
)
def test_new_or_changed_date(raw, refined):
    assert "dates changed" in issues_for(raw, refined)


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("It costs a lot.", "It costs $5,000."),
        ("It costs 50 rupees.", "It costs 50 dollars."),
        ("The budget is fine.", "The budget of 40k dollars is fine."),
    ],
)
def test_new_or_changed_money(raw, refined):
    text = issues_for(raw, refined)
    assert "monetary units changed" in text or "numbers changed" in text


def test_new_percentage():
    assert "percentages changed" in issues_for("Growth was strong.", "Growth was 20% strong.")


def test_new_technical_identifier():
    assert "new technical identifier" in issues_for("We store logs in the bucket.", "We store logs in S3.")


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("We might migrate to Kubernetes next month.", "We agreed to migrate to Kubernetes next month."),
        ("Priya could update the API documentation.", "Priya will update the API documentation."),
        ("Maybe we use Redis.", "We use Redis."),
        ("I suggest we delay the launch.", "We delay the launch."),
    ],
)
def test_proposal_turned_into_decision(raw, refined):
    assert "certainty/commitment words changed" in issues_for(raw, refined)


def test_name_replacement():
    assert "name changed" in issues_for("Ask Priya about it.", "Ask Maria about it.")


def test_invented_owner():
    assert issues_for("Someone should update the docs.", "Rahul should update the docs.")


def test_question_turned_into_statement():
    assert "question" in issues_for("Can we ship on Friday?", "We can ship on Friday.")


@pytest.mark.parametrize(
    "raw,refined",
    [
        ("Let's discuss the roadmap.", "The team reviewed and finalized the roadmap for next quarter."),
        ("ok so the thing with the servers is slow", "Server latency has been identified as a key concern."),
    ],
)
def test_suspiciously_large_rewrite(raw, refined):
    assert "rewrite too large" in issues_for(raw, refined)


def test_added_sentence():
    assert "added words" in issues_for("We ship Friday.", "We ship Friday. Everyone agreed.")


def test_removed_meaningful_words():
    assert "removed words" in issues_for("Send the final report to the client.", "Send the report to the client.")


def test_reported_change_must_exist_in_raw():
    text = issues_for("we use cubernetties", "We use Kubernetes.",
                      [("kubernetties!!", "Kubernetes")])
    assert "not found in the raw text" in text
