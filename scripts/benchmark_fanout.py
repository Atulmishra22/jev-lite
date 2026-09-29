import time
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"

# 1. Realistic, rich multi-part enterprise State
ENTERPRISE_STATE = {
    "customer_profile": {
        "id": "CUST-99214",
        "name": "David Miller",
        "account_type": "Business Premium",
        "lifetime_spend_usd": 14250.00,
        "tenure_months": 28,
        "past_chargebacks": 0,
        "risk_tier": "Low"
    },
    "support_ticket": {
        "ticket_id": "TICK-4401",
        "opened_at": "2026-09-30T03:15:00Z",
        "subject": "Unauthorized recurring fee on my corporate card",
        "messages": [
            {
                "sender": "customer",
                "text": "Hello, I noticed an unexpected charge of $499.00 on my corporate card this morning for an 'Enterprise Add-on' that we never authorized or requested. Our contract specifically states that all add-ons require prior written sign-off from our CFO. Please immediately reverse this charge and confirm that our auto-renew settings have not been altered. We have been loyal customers for over 2 years, but unauthorized billing is unacceptable for our compliance team."
            }
        ]
    },
    "transaction_record": {
        "transaction_id": "TXN-88319",
        "amount_usd": 499.00,
        "currency": "USD",
        "merchant": "SaaS Platform Pro",
        "timestamp": "2026-09-30T01:00:22Z",
        "payment_method": "Corporate Visa ending in 4102",
        "status": "Settled",
        "is_recurring": True
    },
    "company_policy": {
        "section_4_billing": "Disputed add-on fees reported within 60 days are eligible for immediate temporary credit pending audit.",
        "section_7_sla": "Enterprise tier billing disputes have an SLA of 4 business hours.",
        "section_12_refund_approval": "Refunds under $1,000 for accounts with > $10,000 lifetime spend can be auto-approved by L1 support."
    }
}

def generate_60_questions() -> dict:
    """Generates 60 atomic questions across Noul, Choice, and Score primitives."""
    questions = {}

    # --- 20 NOUL (True/False) Questions ---
    noul_statements = [
        "is_refund_demanded: The customer is explicitly asking for a monetary refund.",
        "is_unauthorized_claimed: The customer claims the transaction was unauthorized.",
        "is_enterprise_tier: The account tier is Enterprise or Business Premium.",
        "is_high_spend_customer: The customer lifetime spend exceeds $10,000.",
        "is_charge_settled: The disputed transaction status is settled.",
        "has_prior_chargebacks: The customer has a history of prior chargebacks.",
        "requires_cfo_signoff: Contractual policy requires CFO sign-off for add-ons.",
        "is_auto_renew_questioned: The customer asks about auto-renew settings.",
        "is_within_60_days: The dispute was raised within the 60-day policy window.",
        "eligible_for_auto_approval: The refund amount is eligible for auto-approval under policy 12.",
        "is_card_corporate: The payment method is a corporate card.",
        "customer_threatens_churn: The customer mentions compliance issues or churn risk.",
        "is_sla_under_4_hours: Enterprise billing disputes require resolution within 4 hours.",
        "is_first_dispute: This is the customer's first dispute with the company.",
        "is_fraud_suspected: There is evidence of external fraudulent account takeover.",
        "is_hardware_issue: The issue is related to physical hardware failure.",
        "is_contract_dispute: The dispute involves contractual add-on terms.",
        "is_loyalty_mentioned: The customer explicitly mentions their multi-year tenure.",
        "is_disputed_amount_under_1000: The disputed transaction amount is less than $1,000.",
        "requires_manager_escalation: The dispute requires executive manager override."
    ]
    for item in noul_statements:
        qid, stmt = item.split(": ")
        questions[qid] = {"type": "noul", "statement": stmt}

    # --- 20 CHOICE Questions ---
    choice_configs = [
        ("primary_department", "Which department should handle this dispute?", {"billing": "Invoices and card charges", "legal": "Contract disputes", "technical": "Software bugs"}),
        ("escalation_level", "What escalation tier is appropriate?", {"tier_1": "Standard support", "tier_2": "Senior specialist", "tier_3": "Executive account manager"}),
        ("ticket_sentiment", "What is the customer's emotional tone?", {"neutral": "Calm query", "frustrated": "Dissatisfied", "critical": "Severely upset"}),
        ("suggested_action", "What is the recommended next action?", {"credit": "Issue immediate refund credit", "investigate": "Hold for billing review", "reject": "Decline dispute"}),
        ("churn_risk_level", "What is the risk of losing this account?", {"low": "Stable account", "medium": "At risk", "high": "Immediate flight risk"}),
        ("charge_type", "What kind of charge is being disputed?", {"subscription": "Recurring plan", "addon": "Optional add-on", "usage": "Overage charge"}),
        ("contact_channel", "What communication channel should be used for reply?", {"email": "Standard email reply", "phone": "Immediate phone call", "portal": "In-app ticket"}),
        ("contract_status", "What is the contract standing?", {"standard": "Standard terms", "custom": "Custom enterprise SLA", "expired": "Lapsed agreement"}),
        ("policy_clause", "Which policy section governs this dispute?", {"section_4": "Billing dispute clause", "section_7": "SLA clause", "section_12": "Auto-approval clause"}),
        ("payment_rail", "What payment mechanism was utilized?", {"card": "Corporate credit card", "ach": "Direct bank wire", "invoice": "Net-30 invoice"}),
        ("resolution_type", "What type of resolution is required?", {"reversal": "Charge reversal", "credit_note": "Future billing credit", "explanation": "Invoice clarification"}),
        ("account_stability", "How healthy is this customer relationship?", {"strong": "High loyalty", "shaky": "Deteriorating", "critical": "Imminent loss"}),
        ("sla_window", "What SLA window applies to this request?", {"urgent_4h": "4 hour SLA", "standard_24h": "24 hour SLA", "extended_72h": "72 hour SLA"}),
        ("internal_tag", "What system tag should be attached?", {"billing_error": "Billing malfunction", "sales_miscommunication": "Sales miscommunication", "fraud": "Fraud alert"}),
        ("customer_authority", "Who opened this dispute?", {"authorized": "Authorized account manager", "billing_admin": "Billing contact", "unknown": "Unknown user"}),
        ("impact_severity", "What is the business impact severity?", {"minor": "Individual annoyance", "major": "Corporate compliance issue", "catastrophic": "Legal threat"}),
        ("audit_requirement", "Is an internal audit required?", {"mandatory": "Requires audit review", "optional": "Discretionary audit", "none": "No audit needed"}),
        ("communication_style", "What tone should the response adopt?", {"apologetic": "Formal apology and credit", "firm": "Policy enforcement", "inquisitive": "Fact-finding"}),
        ("payment_status", "What is the status of the funds?", {"settled": "Settled funds", "pending": "Pending authorization", "voided": "Voided"}),
        ("retention_priority", "What is the customer retention priority?", {"tier_gold": "Top 5% VIP client", "tier_silver": "Valued customer", "tier_bronze": "Standard user"})
    ]
    for qid, inst, crit in choice_configs:
        questions[qid] = {"type": "choice", "instructions": inst, "criteria": crit}

    # --- 20 SCORE Questions ---
    score_configs = [
        ("customer_frustration", "Rate the customer's frustration level", {"1": "Calm", "2": "Mildly annoyed", "3": "Very frustrated"}),
        ("urgency_rating", "Rate the operational urgency", {"1": "Low", "2": "Medium", "3": "High"}),
        ("financial_risk", "Rate the financial risk to the company", {"1": "Negligible", "2": "Moderate", "3": "Substantial"}),
        ("brand_reputation_risk", "Rate potential brand reputation damage", {"1": "None", "2": "Minor", "3": "Significant"}),
        ("contract_complexity", "Rate the complexity of the agreement", {"1": "Simple", "2": "Moderate", "3": "Complex"}),
        ("churn_probability", "Rate the likelihood of customer cancellation", {"1": "Unlikely", "2": "Possible", "3": "Imminent"}),
        ("actionability_score", "Rate how actionable this ticket is for support", {"1": "Vague", "2": "Partially clear", "3": "Immediately actionable"}),
        ("evidence_strength", "Rate the strength of evidence provided", {"1": "Weak", "2": "Moderate", "3": "Strong"}),
        ("policy_alignment", "Rate how well this claim aligns with policy", {"1": "Divergent", "2": "Partial match", "3": "Full policy match"}),
        ("sla_criticality", "Rate the severity of breaching SLA", {"1": "Low penalty", "2": "Moderate penalty", "3": "High penalty"}),
        ("account_value_score", "Rate the strategic value of this account", {"1": "Standard", "2": "Valuable", "3": "Key enterprise"}),
        ("dispute_validity", "Rate the apparent validity of the dispute", {"1": "Doubtful", "2": "Plausible", "3": "Highly valid"}),
        ("compliance_exposure", "Rate the regulatory or compliance exposure", {"1": "Low", "2": "Medium", "3": "High"}),
        ("customer_clout", "Rate the customer's market influence", {"1": "Low", "2": "Moderate", "3": "Influential"}),
        ("support_effort_score", "Rate the effort required to resolve", {"1": "Quick 1-click credit", "2": "Multi-team review", "3": "Deep forensic audit"}),
        ("refund_confidence", "Rate confidence in granting a full refund", {"1": "Low confidence", "2": "Medium confidence", "3": "High confidence"}),
        ("repeat_issue_risk", "Rate risk of this recurring next month", {"1": "One-off", "2": "Possible repeat", "3": "Systemic error"}),
        ("cfo_involvement_need", "Rate need for executive sign-off", {"1": "Unnecessary", "2": "Advisory", "3": "Mandatory"}),
        ("customer_satisfaction_potential", "Rate potential to turn this into a positive experience", {"1": "Low", "2": "Fair", "3": "High"}),
        ("overall_priority", "Rate the overall priority score", {"1": "P3 Normal", "2": "P2 Urgent", "3": "P1 Critical"})
    ]
    for qid, inst, levels in score_configs:
        questions[qid] = {"type": "score", "instructions": inst, "levels": levels}

    return questions

def run_benchmark():
    all_questions = generate_60_questions()
    q_keys = list(all_questions.keys())

    test_batches = [
        ("Batch 1 (1 Question)", 1),
        ("Batch 2 (10 Questions)", 10),
        ("Batch 3 (30 Questions)", 30),
        ("Batch 4 (60 Questions)", 60),
    ]

    print("=" * 70)
    print("🚀 SPECULATIVE FAN-OUT BENCHMARK (1 vs 10 vs 30 vs 60 Questions)")
    print("=" * 70)

    results = []

    # Warmup
    print("Warming up engine...")
    warmup_payload = {"state": ENTERPRISE_STATE, "questions": {q_keys[0]: all_questions[q_keys[0]]}}
    httpx.post(f"{BASE_URL}/v1/systemone", json=warmup_payload, timeout=60.0)
    print("Warmup complete!\n")

    for batch_name, count in test_batches:
        batch_qs = {k: all_questions[k] for k in q_keys[:count]}
        payload = {"state": ENTERPRISE_STATE, "questions": batch_qs}

        start_t = time.perf_counter()
        resp = httpx.post(f"{BASE_URL}/v1/systemone", json=payload, timeout=60.0)
        total_time_ms = (time.perf_counter() - start_t) * 1000

        if resp.status_code == 200:
            time_per_q = total_time_ms / count
            results.append((batch_name, count, total_time_ms, time_per_q))
            print(f" {batch_name:<25} | Total: {total_time_ms:>7.2f} ms | Per Question: {time_per_q:>6.2f} ms")
        else:
            print(f" {batch_name} failed with status {resp.status_code}: {resp.text}")

    print("\n" + "=" * 70)
    print(" SPECULATIVE FAN-OUT PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Test Batch':<25} | {'Questions':<10} | {'Total Latency':<15} | {'Per-Question Latency'}")
    print("-" * 70)
    for name, count, total, per_q in results:
        print(f"{name:<25} | {count:<10} | {total:>8.2f} ms     | {per_q:>8.2f} ms/q")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()