"""
AeroMaintain AI - Synthetic Maintenance Knowledge Base Generator

Creates an original synthetic maintenance knowledge base for the
AeroMaintain AI RAG system.

The documents are fictional and designed only for software,
machine-learning, retrieval, and agent evaluation.

They are NOT OEM maintenance instructions and must not be used
for real aircraft maintenance.
"""

from pathlib import Path

from aeromaintain.config import DOCUMENTS_DIR


DISCLAIMER = """
IMPORTANT NOTICE

This document is synthetic and was created for the AeroMaintain AI
portfolio project.

It is not an aircraft manufacturer's maintenance manual, approved
maintenance data, regulatory guidance, or operational instruction.

Do not use this document for real aircraft maintenance.
""".strip()


DOCUMENTS = [
    {
        "document_id": "AM-KB-HYD-001",
        "title": "Hydraulic Pump Degradation Investigation",
        "component_type": "HYDRAULIC_PUMP",
        "symptoms": [
            "Increasing vibration trend",
            "Elevated operating temperature",
            "Repeated maintenance history",
            "Reduced operating stability",
        ],
        "inspection_steps": [
            (
                "Review recent vibration measurements and compare "
                "them with the component's historical trend."
            ),
            (
                "Review operating temperature for sustained or "
                "progressive increases."
            ),
            (
                "Inspect the synthetic maintenance history for "
                "recent repairs, inspections, or recurring findings."
            ),
            (
                "Check whether flight-cycle accumulation is "
                "approaching the synthetic expected-life range."
            ),
            (
                "Escalate recurring abnormal trends for additional "
                "engineering review."
            ),
        ],
        "possible_causes": [
            "Progressive simulated bearing degradation",
            "Synthetic internal wear",
            "Abnormal simulated loading",
            "Recurring degradation after previous maintenance",
        ],
    },
    {
        "document_id": "AM-KB-GEN-001",
        "title": "Generator Condition Investigation",
        "component_type": "GENERATOR",
        "symptoms": [
            "Elevated temperature trend",
            "Increasing vibration",
            "Repeated component faults",
            "Abnormal condition trend after maintenance",
        ],
        "inspection_steps": [
            (
                "Compare current temperature with recent 7-day and "
                "30-day historical averages."
            ),
            (
                "Review vibration trend for persistent increases "
                "rather than relying on a single measurement."
            ),
            (
                "Review previous simulated failures and maintenance "
                "actions associated with the component."
            ),
            (
                "Assess component usage relative to its synthetic "
                "expected-life cycles."
            ),
            (
                "Escalate persistent abnormal condition patterns "
                "for additional engineering investigation."
            ),
        ],
        "possible_causes": [
            "Synthetic bearing wear",
            "Progressive simulated thermal degradation",
            "Simulated mechanical imbalance",
            "Recurring condition deterioration",
        ],
    },
    {
        "document_id": "AM-KB-ACM-001",
        "title": "Air Cycle Machine Condition Investigation",
        "component_type": "AIR_CYCLE_MACHINE",
        "symptoms": [
            "Increasing vibration",
            "Abnormal temperature behavior",
            "Recurring maintenance findings",
            "Progressive degradation trend",
        ],
        "inspection_steps": [
            (
                "Review recent vibration values and rolling "
                "vibration averages."
            ),
            (
                "Review temperature measurements for persistent "
                "deviation from the component's own history."
            ),
            (
                "Check recent maintenance events and determine "
                "whether abnormal trends returned after intervention."
            ),
            (
                "Review accumulated flight cycles and synthetic "
                "component-life utilization."
            ),
            (
                "Request additional engineering assessment when "
                "multiple degradation indicators persist."
            ),
        ],
        "possible_causes": [
            "Synthetic rotating-component wear",
            "Simulated imbalance",
            "Progressive thermal degradation",
            "Recurring simulated mechanical deterioration",
        ],
    },
    {
        "document_id": "AM-KB-FUEL-001",
        "title": "Fuel Pump Degradation Investigation",
        "component_type": "FUEL_PUMP",
        "symptoms": [
            "Increasing vibration trend",
            "Elevated component temperature",
            "Repeated simulated failures",
            "Short interval between maintenance findings",
        ],
        "inspection_steps": [
            (
                "Compare current vibration with recent rolling "
                "historical measurements."
            ),
            (
                "Review temperature behavior for sustained increases."
            ),
            (
                "Inspect maintenance history for repeated corrective "
                "actions involving the same component."
            ),
            (
                "Review flight-cycle accumulation relative to the "
                "synthetic expected component life."
            ),
            (
                "Escalate recurring multi-signal degradation for "
                "additional engineering review."
            ),
        ],
        "possible_causes": [
            "Synthetic pump wear",
            "Simulated bearing deterioration",
            "Progressive mechanical degradation",
            "Recurring degradation after maintenance",
        ],
    },
    {
        "document_id": "AM-KB-ACT-001",
        "title": "Actuator Degradation Investigation",
        "component_type": "ACTUATOR",
        "symptoms": [
            "Abnormal vibration behavior",
            "Increasing temperature",
            "Recurring maintenance history",
            "Progressive simulated wear",
        ],
        "inspection_steps": [
            (
                "Review current and historical vibration behavior."
            ),
            (
                "Compare temperature with recent component-specific "
                "rolling averages."
            ),
            (
                "Review previous maintenance actions and simulated "
                "failure history."
            ),
            (
                "Assess accumulated flight cycles relative to "
                "synthetic expected life."
            ),
            (
                "Escalate persistent or recurring abnormal trends "
                "for engineering investigation."
            ),
        ],
        "possible_causes": [
            "Synthetic mechanical wear",
            "Simulated internal friction",
            "Progressive component degradation",
            "Recurring simulated deterioration",
        ],
    },
]


def render_document(document: dict) -> str:
    """Render one knowledge-base document as Markdown."""

    symptoms = "\n".join(
        f"- {item}"
        for item in document["symptoms"]
    )

    inspection_steps = "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(
            document["inspection_steps"],
            start=1,
        )
    )

    possible_causes = "\n".join(
        f"- {item}"
        for item in document["possible_causes"]
    )

    return f"""# {document["title"]}

Document ID: {document["document_id"]}

Component Type: {document["component_type"]}

## Purpose

This synthetic reference describes a structured investigation
approach for condition indicators associated with the
{document["component_type"]} component type.

## Typical Condition Indicators

{symptoms}

## Investigation Workflow

{inspection_steps}

## Possible Synthetic Causes

{possible_causes}

## Decision-Support Guidance

A single abnormal sensor measurement should not automatically be
interpreted as a component failure. Review multiple indicators,
historical trends, component usage, and maintenance history together.

AeroMaintain AI risk scores are decision-support signals rather than
approved maintenance determinations. Final maintenance decisions
require appropriately authorized personnel and approved technical data.

## Document Notice

{DISCLAIMER}
"""


def generate_documents(
    output_directory: Path = DOCUMENTS_DIR,
) -> None:
    """Generate all synthetic maintenance documents."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for document in DOCUMENTS:
        file_name = (
            f'{document["document_id"].lower()}.md'
        )

        output_path = (
            output_directory / file_name
        )

        output_path.write_text(
            render_document(document),
            encoding="utf-8",
        )

        print(
            f"Created: {output_path}"
        )

    print(
        f"\nGenerated {len(DOCUMENTS)} "
        "synthetic maintenance documents."
    )


if __name__ == "__main__":
    generate_documents()