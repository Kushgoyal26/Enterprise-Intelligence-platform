"""
Generates fake contract and policy documents for the RAG engine to ingest.

Contracts are tied to real customer names from data/raw/customers.csv, so
questions like "What is Acme Corp's termination notice period?" have a real,
checkable answer. Terms are randomized per customer (payment terms, notice
period, SLA %, etc.) so different documents actually say different things —
this matters for testing that RAG retrieves the RIGHT document, not just any
document with the word "contract" in it.

Policies are company-wide (not per-customer) and cover common business
questions: refunds, data retention, support SLAs, security, onboarding.

Usage:
    python data/docs/generate_docs.py
"""

import csv
import os
import random
from datetime import date, timedelta

random.seed(7)

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw")
CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "contracts")
POLICIES_DIR = os.path.join(os.path.dirname(__file__), "policies")

N_CONTRACTS = 15

PAYMENT_TERMS = ["Net 15", "Net 30", "Net 45", "Net 60"]
NOTICE_PERIODS = [30, 45, 60, 90]
SLA_UPTIMES = ["99.5%", "99.9%", "99.95%"]
RENEWAL_TYPES = ["automatic annual renewal unless either party provides written notice",
                 "manual renewal requiring mutual written agreement"]
LIABILITY_CAPS = ["the total fees paid in the preceding 12 months",
                   "two times the total fees paid in the preceding 12 months",
                   "the total fees paid in the preceding 6 months"]


def load_customers():
    path = f"{RAW_DIR}/customers.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def generate_contract(customer: dict) -> str:
    company = customer["company_name"]
    plan = customer["plan_tier"]
    signup = customer["signup_date"]

    payment_terms = random.choice(PAYMENT_TERMS)
    notice_days = random.choice(NOTICE_PERIODS)
    sla = random.choice(SLA_UPTIMES)
    renewal = random.choice(RENEWAL_TYPES)
    liability_cap = random.choice(LIABILITY_CAPS)
    governing_law = random.choice(["the laws of India", "the laws of Delaware, USA", "the laws of Singapore"])

    contract_id = f"CTR-{customer['customer_id'].zfill(4)}"

    return f"""MASTER SUBSCRIPTION AGREEMENT

Contract ID: {contract_id}
Customer: {company}
Plan Tier: {plan}
Effective Date: {signup}

1. SERVICES
The Provider agrees to provide {company} with access to the platform under the
{plan} subscription tier, as further described in the applicable Order Form.

2. PAYMENT TERMS
Payment is due {payment_terms} from the date of invoice. Late payments beyond
15 days past the due date may result in suspension of service access.

3. TERM AND TERMINATION
This Agreement shall continue in effect via {renewal}. Either party may
terminate this Agreement for convenience by providing {notice_days} days'
written notice to the other party. Either party may terminate immediately
upon material breach that remains uncured for 30 days after written notice.

4. SERVICE LEVEL AGREEMENT
Provider commits to a monthly uptime of {sla}. In the event Provider fails to
meet this commitment, Customer shall be entitled to service credits as
described in Exhibit B.

5. LIMITATION OF LIABILITY
Provider's total liability under this Agreement shall not exceed {liability_cap}.
Neither party shall be liable for indirect, incidental, or consequential damages.

6. CONFIDENTIALITY
Each party agrees to maintain the confidentiality of the other party's
proprietary information disclosed under this Agreement for a period of 3 years
following termination.

7. GOVERNING LAW
This Agreement shall be governed by {governing_law}.

8. DATA PROTECTION
Provider shall implement reasonable technical and organizational measures to
protect Customer data and shall not use Customer data for any purpose other
than providing the Services.

Signed on behalf of {company} and the Provider on {signup}.
"""


def generate_policies() -> dict:
    """Return {filename: content} for company-wide policy documents."""
    policies = {}

    policies["refund_policy.txt"] = """REFUND POLICY

Effective for all subscription plans (Starter, Growth, Enterprise).

1. Customers may request a full refund within 14 days of initial signup if
   they have not exceeded 100 API calls or 5 active user seats.
2. After the 14-day window, subscriptions are non-refundable for the current
   billing period, but customers may cancel to prevent future charges.
3. Annual plan customers who cancel after 30 days receive a prorated refund
   for unused months, minus a 10% early termination processing fee.
4. Refunds are processed within 7-10 business days to the original payment
   method.
5. Refunds are not issued for add-on services, custom integrations, or
   professional services engagements once work has commenced.
"""

    policies["data_retention_policy.txt"] = """DATA RETENTION POLICY

1. Active customer data is retained for the duration of the subscription
   plus 90 days after cancellation, to allow for reactivation.
2. After the 90-day grace period, customer data is permanently deleted from
   production systems within 30 additional days.
3. Backup copies are retained for a maximum of 180 days for disaster
   recovery purposes, after which they are purged.
4. Support ticket records and billing history are retained for 7 years to
   comply with financial and tax regulations.
5. Customers may request early data deletion by submitting a formal request
   to privacy@company.com, which will be honored within 30 days, subject to
   legal retention requirements.
"""

    policies["support_sla_policy.txt"] = """SUPPORT SLA POLICY

Response time commitments by ticket priority and plan tier:

Starter Plan:
  - Critical: 8 business hours
  - High: 1 business day
  - Medium: 2 business days
  - Low: 3 business days

Growth Plan:
  - Critical: 4 business hours
  - High: 8 business hours
  - Medium: 1 business day
  - Low: 2 business days

Enterprise Plan:
  - Critical: 1 hour, 24/7
  - High: 4 hours, 24/7
  - Medium: 1 business day
  - Low: 2 business days

Enterprise customers are assigned a dedicated Customer Success Manager and
receive quarterly business reviews. Critical priority is defined as a
complete service outage or data loss event affecting production usage.
"""

    policies["security_compliance_policy.txt"] = """SECURITY AND COMPLIANCE POLICY

1. All customer data is encrypted at rest using AES-256 and in transit using
   TLS 1.2 or higher.
2. The platform undergoes an annual SOC 2 Type II audit, with reports
   available to Enterprise customers under NDA.
3. Access to production systems requires multi-factor authentication and is
   logged and reviewed quarterly.
4. Security incidents affecting customer data will be disclosed to affected
   customers within 72 hours of confirmed detection, in line with applicable
   data protection regulations.
5. Penetration testing is conducted by a third-party firm twice per year.
6. Employees undergo mandatory security awareness training upon hire and
   annually thereafter.
"""

    policies["onboarding_policy.txt"] = """CUSTOMER ONBOARDING POLICY

1. Starter plan customers receive self-service onboarding with access to
   documentation, video tutorials, and community support forums.
2. Growth plan customers are assigned an onboarding specialist for the first
   30 days, including a kickoff call and two check-in sessions.
3. Enterprise plan customers receive a dedicated implementation team, a
   custom onboarding plan, and a target time-to-first-value of 21 days.
4. All new customers receive access to a sandbox environment for testing
   before going live in production.
5. Onboarding is considered complete when the customer has successfully
   integrated at least one core workflow and completed platform training.
"""

    policies["billing_and_invoicing_policy.txt"] = """BILLING AND INVOICING POLICY

1. Subscriptions are billed monthly or annually in advance, based on the
   plan selected at signup.
2. Invoices are generated on the first business day of each billing cycle
   and sent to the billing contact on file.
3. Failed payments trigger up to 3 automatic retry attempts over 7 days.
   If payment continues to fail, the account is downgraded to read-only
   access until payment is received.
4. Mid-cycle plan upgrades are billed on a prorated basis immediately.
   Downgrades take effect at the start of the next billing cycle.
5. All prices are in INR unless otherwise specified in a custom Enterprise
   agreement. Applicable taxes are added at checkout based on the billing
   address on file.
"""

    return policies


if __name__ == "__main__":
    os.makedirs(CONTRACTS_DIR, exist_ok=True)
    os.makedirs(POLICIES_DIR, exist_ok=True)

    customers = load_customers()
    sample = random.sample(customers, min(N_CONTRACTS, len(customers)))

    for c in sample:
        filename = c["company_name"].replace(" ", "_").replace(",", "").replace(".", "") + "_Contract.txt"
        content = generate_contract(c)
        with open(os.path.join(CONTRACTS_DIR, filename), "w", encoding="utf-8") as f:
            f.write(content)

    policies = generate_policies()
    for filename, content in policies.items():
        with open(os.path.join(POLICIES_DIR, filename), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Generated {len(sample)} contracts in {CONTRACTS_DIR}")
    print(f"Generated {len(policies)} policy documents in {POLICIES_DIR}")
    print("\nSample contract customers:")
    for c in sample[:5]:
        print(f"  - {c['company_name']}")
