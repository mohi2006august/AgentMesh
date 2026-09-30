"""`mesh` command line interface.

mesh validate examples/support-agent.yaml
mesh schema > agent.schema.json
"""

from __future__ import annotations

import json
import sys

import click

from mesh_core.errors import MeshError
from mesh_sdk.spec import agent_json_schema, load_agent_file


def _fail(error: MeshError) -> None:
    click.secho(f"error: {error.message}", fg="red", err=True)
    for key, value in error.detail.items():
        if isinstance(value, list):
            for item in value:
                click.echo(f"  - {item}", err=True)
        else:
            click.echo(f"  {key}: {value}", err=True)
    sys.exit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="agentmesh", prog_name="mesh")
def main() -> None:
    """Deploy and operate agents on AgentMesh."""


@main.command()
@click.argument("file", type=click.Path(dir_okay=False))
def validate(file: str) -> None:
    """Check an agent.yaml without deploying it."""
    try:
        spec = load_agent_file(file)
    except MeshError as error:
        _fail(error)
        return
    tools = ", ".join(tool.mcp for tool in spec.tools) or "none"
    knowledge = ", ".join(source.source for source in spec.knowledge) or "none"
    click.secho(f"ok: {spec.name} ({spec.spec_hash})", fg="green")
    click.echo(
        f"  model      {spec.model.default}"
        + (f" -> {spec.model.fallback}" if spec.model.fallback else "")
    )
    click.echo(f"  tools      {tools}")
    click.echo(f"  knowledge  {knowledge}")
    click.echo(
        f"  limits     {spec.limits.max_steps} steps, ${spec.limits.max_usd_per_run:.2f} per run"
    )


@main.command()
def schema() -> None:
    """Print the JSON Schema for agent.yaml (for editor validation)."""
    click.echo(json.dumps(agent_json_schema(), indent=2))


if __name__ == "__main__":
    main()
