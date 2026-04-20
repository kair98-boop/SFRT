"""TOPAS case rendering and execution helpers."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


PHYSICS_PROFILES = {
    "topas_default": [
        "g4em-standard_opt4",
        "g4h-phy_QGSP_BIC_HP",
        "g4decay",
        "g4ion-binarycascade",
        "g4h-elastic_HP",
        "g4stopping",
    ],
    "em_opt4_only": ["g4em-standard_opt4"],
    "em_opt0_only": ["g4em-standard_opt0"],
}


def fill_template(template_text: str, replacements: Mapping[str, str]) -> str:
    """Replace placeholder tokens in a TOPAS template string."""

    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(str(key), str(value))
    return rendered


def render_case_text(template_path: str | Path, replacements: Mapping[str, str]) -> str:
    """Read a template file and render it with replacement tokens."""

    template_text = Path(template_path).read_text(encoding="utf-8")
    return fill_template(template_text, replacements)


def format_physics_modules(profile_name: str, physics_profiles: Mapping[str, Sequence[str]] | None = None) -> str:
    """Format a named TOPAS modular-physics profile."""

    profiles = PHYSICS_PROFILES if physics_profiles is None else physics_profiles
    modules = list(profiles[profile_name])
    quoted = " ".join(f'"{module}"' for module in modules)
    return f"{len(modules)} {quoted}"


def discover_g4_data_env(g4_data_dir: str | Path) -> Dict[str, str]:
    """Resolve Geant4 data environment variables from a base install directory."""

    base = Path(g4_data_dir).expanduser()
    search_roots = [
        base,
        base / "G4DATA",
        base / "geant4-install" / "share" / "Geant4" / "data",
    ]
    available_roots = [root for root in search_roots if root.exists() and root.is_dir()]
    if not available_roots:
        return {}

    env_to_prefixes = {
        "G4NEUTRONHPDATA": ["G4NDL"],
        "G4PARTICLEXSDATA": ["G4PARTICLEXS"],
        "G4PIIDATA": ["G4PII"],
        "G4LEVELGAMMADATA": ["PhotonEvaporation", "G4PhotonEvaporation"],
        "G4RADIOACTIVEDATA": ["RadioactiveDecay", "G4RadioactiveDecay"],
        "G4LEDATA": ["G4EMLOW"],
        "G4SAIDXSDATA": ["G4SAIDDATA"],
        "G4REALSURFACEDATA": ["RealSurface"],
        "G4ABLADATA": ["G4ABLA"],
        "G4INCLDATA": ["G4INCL"],
        "G4ENSDFSTATEDATA": ["G4ENSDFSTATE"],
        "G4CHANNELINGDATA": ["G4CHANNELING"],
    }

    resolved: Dict[str, str] = {}
    for env_var, prefixes in env_to_prefixes.items():
        for root in available_roots:
            matches = sorted(
                [
                    child
                    for child in root.iterdir()
                    if child.is_dir() and any(child.name.startswith(prefix) for prefix in prefixes)
                ],
                key=lambda p: p.name,
                reverse=True,
            )
            if matches:
                resolved[env_var] = str(matches[0].resolve())
                break

    if "G4NEUTRONXSDATA" not in resolved:
        if "G4PARTICLEXSDATA" in resolved:
            resolved["G4NEUTRONXSDATA"] = resolved["G4PARTICLEXSDATA"]
        elif "G4NEUTRONHPDATA" in resolved:
            resolved["G4NEUTRONXSDATA"] = resolved["G4NEUTRONHPDATA"]

    return resolved


def build_topas_env(g4_data_dir: str | Path) -> Dict[str, str]:
    """Build the environment used for TOPAS runs."""

    env = os.environ.copy()
    env["TOPAS_G4_DATA_DIR"] = str(Path(g4_data_dir).expanduser())
    env.update(discover_g4_data_env(g4_data_dir))
    return env


def write_text_with_retries(
    path: str | Path,
    content: str,
    retries: int = 8,
    retry_delay_sec: float = 0.75,
) -> None:
    """Atomically write a text file with simple retry handling."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay_sec))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(out_path)
            return
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise OSError(f"Failed to write {out_path}")


def has_nonempty_output(path: str | Path) -> bool:
    """Return True when a file exists and is non-empty."""

    out_path = Path(path)
    try:
        return out_path.exists() and out_path.stat().st_size > 0
    except OSError:
        return False


def run_topas_case(
    *,
    topas_bin: str | Path,
    case_dir: str | Path,
    parameter_file: str | Path,
    g4_data_dir: str | Path,
    expected_outputs: Iterable[str | Path] = (),
    log_file: str | Path | None = None,
    log_tail_lines: int = 80,
    failure_context: str = "TOPAS run failed.",
) -> subprocess.CompletedProcess[str]:
    """Run a TOPAS case and optionally validate expected outputs."""

    case_path = Path(case_dir)
    parameter_path = Path(parameter_file)
    result = subprocess.run(
        [str(topas_bin), parameter_path.name],
        cwd=str(case_path),
        capture_output=True,
        text=True,
        env=build_topas_env(g4_data_dir),
    )

    combined_log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if log_file is not None:
        write_text_with_retries(log_file, combined_log)

    output_paths = [Path(path) for path in expected_outputs]
    missing_outputs = [path for path in output_paths if not has_nonempty_output(case_path / path)]
    if result.returncode != 0 or missing_outputs:
        tail = "\n".join(combined_log.strip().splitlines()[-int(log_tail_lines) :])
        missing_display = ", ".join(str(path) for path in missing_outputs) if missing_outputs else "none"
        raise RuntimeError(
            f"{failure_context}\n"
            f"Return code: {result.returncode}\n"
            f"Missing outputs: {missing_display}\n"
            f"Recent log:\n{tail}"
        )

    return result
