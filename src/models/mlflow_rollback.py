"""
Stretch goal #5 — promote Staging → Production, or roll back Production.

Usage (from project root):
    python -m src.models.mlflow_rollback status
    python -m src.models.mlflow_rollback promote
    python -m src.models.mlflow_rollback rollback
"""

from __future__ import annotations

import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "purchase-intent-model"
TRACKING_URI = "sqlite:///mlflow.db"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def _versions_by_stage(client: MlflowClient) -> dict[str, list]:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    by_stage: dict[str, list] = {}
    for mv in versions:
        by_stage.setdefault(mv.current_stage, []).append(mv)
    for stage in by_stage:
        by_stage[stage].sort(key=lambda v: int(v.version), reverse=True)
    return by_stage


def status() -> None:
    client = _client()
    by_stage = _versions_by_stage(client)
    if not by_stage:
        print(f"No versions registered for '{MODEL_NAME}'.")
        print("Run: python -m src.models.mlflow_registry")
        return
    print(f"Model: {MODEL_NAME}")
    for stage, versions in sorted(by_stage.items()):
        for mv in versions:
            print(f"  [{stage}] v{mv.version}  run_id={mv.run_id}")


def promote() -> None:
    """Move current Staging version to Production; archive old Production."""
    client = _client()
    by_stage = _versions_by_stage(client)
    staging = by_stage.get("Staging") or []
    if not staging:
        raise SystemExit("No Staging version to promote.")

    staging_v = staging[0]
    # Archive current Production first (if any)
    for prod in by_stage.get("Production") or []:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=prod.version,
            stage="Archived",
        )
        logger.info("Archived previous Production v%s", prod.version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_v.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted v{staging_v.version} -> Production")
    status()


def rollback() -> None:
    """
    Roll back Production to the latest Archived version
    (typically the previous Production after a promote).
    """
    client = _client()
    by_stage = _versions_by_stage(client)
    archived = by_stage.get("Archived") or []
    production = by_stage.get("Production") or []

    if not archived:
        raise SystemExit(
            "No Archived version to roll back to. "
            "Promote a Staging model first so the old Production is Archived."
        )
    if not production:
        raise SystemExit("No current Production version to replace.")

    previous = archived[0]  # highest archived version
    current = production[0]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=current.version,
        stage="Archived",
    )
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=previous.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"Rolled back: Production v{current.version} -> Archived; "
        f"restored v{previous.version} -> Production"
    )
    status()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Promote or roll back purchase-intent-model in MLflow Registry"
    )
    parser.add_argument(
        "action",
        choices=["status", "promote", "rollback"],
        help="status | promote Staging→Production | rollback to previous Production",
    )
    args = parser.parse_args(argv)

    if args.action == "status":
        status()
    elif args.action == "promote":
        promote()
    elif args.action == "rollback":
        rollback()


if __name__ == "__main__":
    main(sys.argv[1:])
