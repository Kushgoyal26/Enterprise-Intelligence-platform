-- Enterprise Intelligence Platform — SaaS Domain Schema
-- Star-schema style: fact tables (orders, tickets, payments) + dimension (customers)

DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Dimension: Customers
CREATE TABLE customers (
    customer_id     SERIAL PRIMARY KEY,
    company_name    VARCHAR(200) NOT NULL,
    industry        VARCHAR(100),
    region          VARCHAR(50),        -- North, South, East, West, Central
    plan_tier       VARCHAR(50),        -- Starter, Growth, Enterprise
    signup_date     DATE NOT NULL,
    employee_count  INT,
    account_owner   VARCHAR(100),       -- sales rep name
    is_active       BOOLEAN DEFAULT TRUE,
    churned_date    DATE                -- NULL if still active
);

-- Fact: Subscriptions (monthly recurring revenue per customer per month)
CREATE TABLE subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    customer_id     INT REFERENCES customers(customer_id),
    billing_month   DATE NOT NULL,      -- first day of month
    mrr_amount      NUMERIC(12,2) NOT NULL,   -- monthly recurring revenue (INR)
    seats           INT,
    login_count     INT,                -- product usage signal for churn
    feature_usage_score NUMERIC(5,2)    -- 0-100, engagement proxy
);

-- Fact: Support Tickets
CREATE TABLE support_tickets (
    ticket_id       SERIAL PRIMARY KEY,
    customer_id     INT REFERENCES customers(customer_id),
    created_date    DATE NOT NULL,
    priority        VARCHAR(20),        -- Low, Medium, High, Critical
    category        VARCHAR(50),        -- Billing, Bug, Feature Request, Onboarding
    resolution_hours NUMERIC(6,2),
    satisfaction_score INT              -- 1-5, NULL if unrated
);

-- Fact: Payments (transactions, includes some anomalies for fraud/anomaly detection)
CREATE TABLE payments (
    payment_id      SERIAL PRIMARY KEY,
    customer_id     INT REFERENCES customers(customer_id),
    payment_date    DATE NOT NULL,
    amount          NUMERIC(12,2) NOT NULL,
    payment_method  VARCHAR(50),        -- Card, Bank Transfer, UPI
    status          VARCHAR(20),        -- Success, Failed, Refunded
    is_anomaly      BOOLEAN DEFAULT FALSE  -- ground-truth flag for testing anomaly detection
);

-- Indexes for query performance
CREATE INDEX idx_subs_customer ON subscriptions(customer_id);
CREATE INDEX idx_subs_month ON subscriptions(billing_month);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX idx_payments_customer ON payments(customer_id);
CREATE INDEX idx_payments_date ON payments(payment_date);
