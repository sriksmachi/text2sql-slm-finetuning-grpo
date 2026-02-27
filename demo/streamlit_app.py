"""
streamlit_app.py – Demo front-end that calls an Azure ML managed online endpoint.

Run locally
-----------
    streamlit run demo/streamlit_app.py

Environment variables
---------------------
AZURE_ML_ENDPOINT_URL   : Scoring URI of the managed online endpoint.
AZURE_ML_ENDPOINT_KEY   : Primary or secondary authentication key.
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENDPOINT_URL = os.getenv("AZURE_ML_ENDPOINT_URL", "")
ENDPOINT_KEY = os.getenv("AZURE_ML_ENDPOINT_KEY", "")

SAMPLE_SCHEMAS = {
    "TPC-H": (
        "lineitem(l_orderkey, l_partkey, l_suppkey, l_quantity, l_extendedprice, l_discount, l_shipdate) | "
        "orders(o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate) | "
        "customer(c_custkey, c_name, c_address, c_nationkey, c_acctbal) | "
        "part(p_partkey, p_name, p_mfgr, p_brand, p_type, p_size) | "
        "supplier(s_suppkey, s_name, s_address, s_nationkey, s_acctbal)"
    ),
    "Enterprise HR": (
        "employee(employee_id, first_name, last_name, email, salary, department_id, manager_id) | "
        "department(department_id, department_name, manager_id, location_id) | "
        "job(job_id, job_title, min_salary, max_salary) | "
        "location(location_id, city, state_province, country_id)"
    ),
    "Enterprise Sales": (
        "product(product_id, product_name, category_id, unit_price, units_in_stock) | "
        "order(order_id, customer_id, employee_id, order_date, freight) | "
        "order_detail(order_id, product_id, unit_price, quantity, discount) | "
        "customer(customer_id, company_name, contact_name, country)"
    ),
    "Custom": "",
}


# ---------------------------------------------------------------------------
# Endpoint call
# ---------------------------------------------------------------------------


def call_endpoint(question: str, schema: str, endpoint_url: str, api_key: str) -> dict:
    """POST to Azure ML managed endpoint and return the parsed JSON response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"question": question, "schema": schema}
    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Text-to-SQL · GRPO Demo",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Text-to-SQL – GRPO Azure ML Demo")
    st.caption(
        "Powered by GRPO fine-tuned LLM on Azure ML | "
        "[GitHub](https://github.com/sriksmachi/text2sql-grpo-azure-ml)"
    )

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Endpoint Configuration")
        endpoint_url = st.text_input(
            "Endpoint URL",
            value=ENDPOINT_URL,
            placeholder="https://<endpoint>.inference.ml.azure.com/score",
        )
        api_key = st.text_input("API Key", value=ENDPOINT_KEY, type="password")

        st.divider()
        st.header("📋 Schema Preset")
        schema_preset = st.selectbox("Choose a preset schema", list(SAMPLE_SCHEMAS.keys()))

    # ── Main area ──────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Schema")
        schema_default = SAMPLE_SCHEMAS[schema_preset]
        schema = st.text_area(
            "Database schema (table(col1, col2, ...) | ...)",
            value=schema_default,
            height=200,
            help="Describe your schema in compact form. Tables separated by ' | '.",
        )

        st.subheader("Question")
        question = st.text_input(
            "Natural language question",
            placeholder="What is the total revenue for orders shipped in January 1995?",
        )

        generate = st.button("⚡ Generate SQL", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Generated SQL")
        if generate:
            if not endpoint_url or not api_key:
                st.warning("Please provide the endpoint URL and API key in the sidebar.")
            elif not question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Calling endpoint…"):
                    try:
                        result = call_endpoint(question, schema, endpoint_url, api_key)
                        sql = result.get("sql") or result.get("output") or str(result)
                        st.code(sql, language="sql")
                        with st.expander("Raw response"):
                            st.json(result)
                    except requests.HTTPError as exc:
                        st.error(f"HTTP error: {exc.response.status_code} – {exc.response.text}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Error: {exc}")
        else:
            st.info("Fill in the schema and question, then click **Generate SQL**.")

    # ── Footer ─────────────────────────────────────────────
    st.divider()
    st.caption(
        "Model: Qwen2.5-Coder-7B-Instruct (4-bit, GRPO fine-tuned) · "
        "Rewards: format + execution + schema fidelity"
    )


if __name__ == "__main__":
    main()
