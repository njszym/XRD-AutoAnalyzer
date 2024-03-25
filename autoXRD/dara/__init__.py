from __future__ import annotations

import datetime
import json
import os
import platform
import re
import shutil
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from subprocess import run
from typing import Optional, Literal, Any, Union

import numpy as np
import requests
from asteval import Interpreter
from pymatgen.core import Element, DummySpecie, Lattice, Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from tqdm import tqdm

with open(Path(__file__).parent / "data" / "possible_species.txt") as f:
    POSSIBLE_SPECIES = {sp.strip() for sp in f}


def do_refinement(
        pattern_path: Path | str,
        phase_paths: list[Path | str],
        instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
        working_dir: Optional[Path | str] = None,
        phase_params=None,
        refinement_params=None,
):
    pattern_path = Path(pattern_path)
    phase_paths = [Path(cif_path) for cif_path in phase_paths]
    working_dir = Path(working_dir) if working_dir is not None else None

    if working_dir is None:
        working_dir = pattern_path.parent / f"refinement_{pattern_path.stem}"

    if not working_dir.exists():
        working_dir.mkdir(exist_ok=True, parents=True)

    if phase_params is None:
        phase_params = {}
    if refinement_params is None:
        refinement_params = {}

    if not pattern_path.suffix == ".xy":
        raise ValueError("Only xy files are supported for now")

    str_paths = []
    for cif_path in phase_paths:
        str_path = cif2str(cif_path, phase_name_suffix="", working_dir=working_dir, **phase_params)
        str_paths.append(str_path)

    control_file_path = generate_control_file(
        pattern_path,
        str_paths,
        instrument_name,
        working_dir=working_dir,
        **refinement_params,
    )

    bgmn_worker = BGMNWorker()
    bgmn_worker.run_refinement_cmd(control_file_path)
    try:
        return get_result(control_file_path)
    except Exception as e:
        raise RuntimeError(f"Error in BGMN refinement for {control_file_path}") from e


def do_refinement_no_saving(
        pattern_path: Path | str,
        phase_paths: list[Path | str],
        instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
        phase_params=None,
        refinement_params=None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        return do_refinement(
            pattern_path,
            phase_paths,
            instrument_name,
            working_dir=working_dir,
            phase_params=phase_params,
            refinement_params=refinement_params,
        )


class BGMNWorker:
    def __init__(self):
        self.bgmn_folder = (
                Path(__file__).parent / "data" / "BGMNwin"
        ).absolute()

        self.bgmn_path = self.bgmn_folder / "bgmn"

        if not self.bgmn_path.exists() and not self.bgmn_path.with_suffix(".exe").exists():
            download_bgmn()

        if not self.bgmn_path.exists():
            self.bgmn_path = self.bgmn_folder / "bgmn.exe"

        if not self.bgmn_path.exists():
            raise FileNotFoundError(
                f"Cannot find BGMN executable at {self.bgmn_path}"
            )

        os.environ["EFLECH"] = self.bgmn_folder.as_posix()
        os.environ["PATH"] += os.pathsep + self.bgmn_folder.as_posix()

    def run_refinement_cmd(self, control_file: Path):
        cp = run(
            [self.bgmn_path.as_posix(), control_file.absolute().as_posix()],
            cwd=control_file.parent.absolute().as_posix(),
            capture_output=True,
            check=False,
            timeout=600,
        )
        if cp.returncode:
            raise RuntimeError(
                f"Error in BGMN refinement for {control_file}. The exit code is {cp.returncode}\n"
                f"{cp.stdout}\n"
                f"{cp.stderr}"
            )


class CIF2StrError(Exception):
    pass


def process_specie_string(sp: str | Specie | Element | DummySpecie) -> str:
    specie = re.sub(r"(\d+)([+-])", r"\2\1", str(sp))
    if specie.endswith(("+", "-")):
        specie += "1"
    specie = specie.upper()

    if specie not in POSSIBLE_SPECIES:

        specie = re.search(r"[A-Z]+", specie).group(0)
        if specie not in POSSIBLE_SPECIES:
            raise CIF2StrError(
                f"Unknown species {specie}, the original specie string is {sp}"
            )
    return specie


def get_lattice_parameters_from_lattice(
        lattice: Lattice,
        crystal_system: Literal[
            "Monoclinic",
            "Cubic",
            "Hexagonal",
            "Trigonal",
            "Orthorhombic",
            "Triclinic",
            "Tetragonal",
            "Rhombohedral",
        ],
) -> dict[str, float]:
    if crystal_system == "Triclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "ALPHA": lattice.alpha,
            "BETA": lattice.beta,
            "GAMMA": lattice.gamma,
        }
    if crystal_system == "Monoclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "BETA": lattice.beta,
        }
    if crystal_system == "Orthorhombic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Tetragonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Rhombohedral":
        return {
            "A": lattice.a / 10,
            "GAMMA": lattice.alpha,
        }
    if crystal_system == "Hexagonal" or crystal_system == "Trigonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Cubic":
        return {
            "A": lattice.a / 10,
        }

    raise CIF2StrError(f"Unknown crystal system {crystal_system}")


def get_std_position(
        spacegroup_setting: dict[str, Any],
        wyckoff_letter: str,
        positions: list[list[float]],
) -> tuple[list[float], bool]:
    wyckoff = spacegroup_setting["wyckoffs"].get(wyckoff_letter, {})

    if not wyckoff:
        raise CIF2StrError(f"Cannot find the wyckoff letter {wyckoff_letter}")

    std_notations = wyckoff["std_notations"]

    positions = [standardize_coords(*position) for position in positions]

    for position in positions:
        variable_dict = {
            "x": position[0],
            "y": position[1],
            "z": position[2],
        }
        for std_notation in std_notations:
            constraints = std_notation.split(" ")

            aeval = Interpreter(use_numpy=False, symtable=variable_dict)
            wx, wy, wz = (aeval.eval(constraint) for constraint in constraints)
            if (
                    fuzzy_compare(wx, position[0])
                    and fuzzy_compare(wy, position[1])
                    and fuzzy_compare(wz, position[2])
            ):
                return position, True
    return positions[0], False


def check_wyckoff(
        spacegroup_setting: dict[str, Any], structure: SymmetrizedStructure
) -> tuple[list[dict[str, Any]], int]:
    element_settings = []
    error_count = 0

    for site_idx in structure.equivalent_indices:
        idx = site_idx[0]
        site = structure[idx]
        wyckoff_letter = structure.wyckoff_letters[idx]
        if wyckoff_letter == "A":
            wyckoff_letter = "alpha"

        std_position, ok = get_std_position(
            spacegroup_setting,
            wyckoff_letter,
            [structure[idx].frac_coords for idx in site_idx],
        )

        if not ok:
            error_count += 1

        if site.is_ordered:
            species_string = process_specie_string(str(next(iter(site.species))))
        else:
            sorted_species = sorted(site.species)
            species_string = ",".join(
                f"{process_specie_string(ssp)}({site.species[ssp]:.6f})"
                for ssp in sorted_species
            )
            species_string = f"({species_string})"

        element_setting = {
            "E": species_string,
            "Wyckoff": wyckoff_letter,
            "x": f"{std_position[0]:.6f}",
            "y": f"{std_position[1]:.6f}",
            "z": f"{std_position[2]:.6f}",
            "TDS": f"{0.01:.6f}",
        }
        element_settings.append(element_setting)

    return element_settings, error_count


def make_spacegroup_setting_str(spacegroup_setting: dict[str, Any]) -> str:
    return (
            " ".join([f"{k}={v}" for k, v in spacegroup_setting["setting"].items()]) + " //"
    )


def make_lattice_parameters_str(
        spacegroup_setting: dict[str, Any],
        structure: SymmetrizedStructure,
        lattice_range: float,
) -> str:
    crystal_system = spacegroup_setting["setting"]["Lattice"]
    lattice_parameters = get_lattice_parameters_from_lattice(
        structure.lattice, crystal_system
    )

    lattice_parameters_str = " ".join(
        [
            f"PARAM={k}={v:.5f}_{v * (1 - lattice_range):.5f}^{v * (1 + lattice_range):.5f}"
            for k, v in lattice_parameters.items()
        ]
    )
    lattice_parameters_str += " //"
    return lattice_parameters_str


def make_peak_parameter_str(k1: str, k2: str, b1: str, gewicht: str, rp: int) -> str:
    return (
            f"RP={rp} "
            + (f"PARAM=k1={k1} " if k1 != "fixed" else "k1=0 ")
            + (f"PARAM=k2={k2} " if k2 != "fixed" else "k2=0 ")
            + (f"PARAM=B1={b1} " if b1 != "fixed" else "B1=0 ")
            + (f"GEWICHT={gewicht} //" if gewicht != "0_0" else "PARAM=GEWICHT=0_0 //")
    )


def cif2str(
        cif_path: Path,
        phase_name_suffix: str = "",
        working_dir: Path | None = None,
        *,
        lattice_range: float = 0.1,
        gewicht: str = "0_0",
        rp: int = 4,
        k1: str = "0_0^0.01",
        k2: str = "0_0^0.01",
        b1: str = "0_0^0.01",
) -> Path:
    str_path = (
        cif_path.parent / f"{cif_path.stem}.str"
        if working_dir is None
        else working_dir / f"{cif_path.stem}.str"
    )

    structure, spg = load_symmetrized_structure(cif_path)

    hall_number = str(spg.get_symmetry_dataset()["hall_number"])
    with (Path(__file__).parent / "data" / "spglib_db" / "spg.json").open(
            "r", encoding="utf-8"
    ) as f:
        spg_group_db = json.load(f)
    settings = spg_group_db[hall_number]["settings"]

    best_setting = None
    for spacegroup_setting in settings:
        element_settings, error_count = check_wyckoff(spacegroup_setting, structure)
        if best_setting is None or error_count < best_setting[2]:
            best_setting = (spacegroup_setting, element_settings, error_count)

        if error_count == 0:
            break

    spacegroup_setting, element_settings, error_count = best_setting

    if error_count > 0:
        raise CIF2StrError(
            f"Cannot find a valid lattice symmetry setting for {cif_path}."
        )

    str_text = ""

    phase_name = process_phase_name(cif_path.stem + phase_name_suffix)
    str_text += f"PHASE={phase_name} // generated by pymatgen {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    formula = structure.composition.reduced_formula
    str_text += f"FORMULA={formula} //\n"

    str_text += make_spacegroup_setting_str(spacegroup_setting) + "\n"

    str_text += (
            make_lattice_parameters_str(
                spacegroup_setting, structure, lattice_range=lattice_range
            )
            + "\n"
    )

    str_text += make_peak_parameter_str(k1, k2, b1, gewicht, rp) + "\n"

    str_text += f"GOAL:{phase_name}=GEWICHT*ifthenelse(ifdef(d),exp(my*d*3/4),1) //\nGOAL=GrainSize(1,1,1) //\n"

    element_settings_str = [
        " ".join([f"{k}={v}" for k, v in element_setting.items()])
        for element_setting in element_settings
    ]
    str_text += "\n".join(element_settings_str)

    with open(str_path, "w") as f:
        f.write(str_text)

    return str_path


def copy_instrument_files(instrument_name: str, working_dir: Path) -> None:
    instrument_path = Path(__file__).parent / "data" / "BGMN-Templates" / "Devices"

    for file in instrument_path.glob(f"{instrument_name}*"):
        shutil.copy(file, working_dir)


def copy_xy_pattern(pattern_path: Path, working_dir: Path) -> Path:
    if pattern_path.parent != working_dir:
        shutil.copy(pattern_path, working_dir)
    return working_dir / pattern_path.name


def generate_control_file(
        pattern_path: Path,
        str_paths: list[Path],
        instrument_name: str,
        working_dir: Path | None = None,
        *,
        n_threads: int = 8,
        wmin: float | None = None,
        wmax: float | None = None,
        eps2: float | str = "0_-0.01^0.01",
) -> Path:
    if working_dir is None:
        control_file_path = pattern_path.parent / f"{pattern_path.stem}.sav"
    else:
        control_file_path = working_dir / f"{pattern_path.stem}.sav"

    copy_xy_pattern(pattern_path, control_file_path.parent)
    copy_instrument_files(instrument_name, control_file_path.parent)

    phases_str = "\n".join(
        [f"STRUC[{i}]={str_path.name}" for i, str_path in enumerate(str_paths, start=1)]
    )

    phase_names = [read_phase_name_from_str(str_path) for str_path in str_paths]
    phase_fraction_str = "\n".join(
        [f"Q{phase_name}={phase_name}/sum" for phase_name in phase_names]
    )
    goal_str = "\n".join(
        [
            f"GOAL[{i}]=Q{phase_name}"
            for i, phase_name in enumerate(phase_names, start=1)
        ]
    )

    control_file = f"""
    % Theoretical instrumental function
    VERZERR={instrument_name}.geq
    % Wavelength
    LAMBDA=CU
    {f"WMIN={wmin}" if wmin is not None else ""}
    {f"WMAX={wmax}" if wmax is not None else ""}
    % Phases
    {phases_str}
    % Measured data
    VAL[1]={pattern_path.name}
    % Result list output
    LIST={pattern_path.stem}.lst
    % Peak list output
    OUTPUT={pattern_path.stem}.par
    % Diagram output
    DIAGRAMM={pattern_path.stem}.dia
    % Global parameters for zero point and sample displacement
    EPS1=0
    {f"PARAM[1]=EPS2={eps2}" if isinstance(eps2, str) else f"EPS2={eps2}"}
    NTHREADS={n_threads}
    PROTOKOLL=Y
    sum={"+".join(phase_name for phase_name in phase_names)}
    {phase_fraction_str}
    {goal_str}
    """
    control_file = re.sub(r"^\s+", "", control_file, flags=re.MULTILINE)

    with open(control_file_path, "w") as f:
        f.write(control_file)

    return control_file_path


def get_result(control_file: Path):
    lst_path = control_file.parent / f"{control_file.stem}.lst"
    dia_path = control_file.parent / f"{control_file.stem}.dia"

    sav_text = control_file.read_text()
    phase_names = re.findall(r"STRUC\[\d+]=(.+?)\.str", sav_text)

    result = {
        "lst_data": parse_lst(lst_path, phase_names),
        "plot_data": parse_dia(dia_path, phase_names),
    }

    return result


def parse_lst(lst_path: Path, phase_names: list[str]):
    def parse_values(v_: str) -> Union[float, tuple[float, float], None, str, int]:
        try:
            v_ = v_.strip("%")
            if v_ == "ERROR" or v_ == "UNDEF":
                return None
            if "+-" in v_:
                v_ = (float(v_.split("+-")[0]), float(v_.split("+-")[1]))
            else:
                v_ = float(v_)
                if v_.is_integer():
                    v_ = int(v_)
        except ValueError:
            pass
        return v_

    def parse_section(text: str) -> dict[str, Any]:
        section = dict(re.findall(r"^(\w+)=(.+?)$", text, re.MULTILINE))
        section = {k: parse_values(v) for k, v in section.items()}
        return section

    if not lst_path.exists():
        raise FileNotFoundError(f"Cannot find the .lst file from {lst_path}")

    with lst_path.open() as f:
        texts = f.read()

    pattern_name = re.search(r"Rietveld refinement to file\(s\) (.+?)\n", texts).group(
        1
    )
    result = {"raw_lst": texts, "pattern_name": pattern_name}

    num_steps = int(re.search(r"(\d+) iteration steps", texts).group(1))
    result["num_steps"] = num_steps

    for var in ["Rp", "Rpb", "R", "Rwp", "Rexp"]:
        result[var] = float(re.search(rf"{var}=(\d+(\.\d+)?)%", texts).group(1))
    result["d"] = (
        float(d.group(1))
        if (d := re.search(r"Durbin-Watson d=(\d+(\.\d+)?)", texts))
        else None
    )
    result["1-rho"] = (
        float(rho.group(1))
        if (rho := re.search(r"1-rho=(\d+(\.\d+)?)%", texts))
        else None
    )

    # global goals
    global_parameters_text = re.search(
        r"Global parameters and GOALs\n(.*?)\n(?:\n|\Z)", texts, re.DOTALL
    )
    if global_parameters_text:
        global_parameters_text = global_parameters_text.group(1)
        global_parameters = parse_section(global_parameters_text)
        result.update(global_parameters)

    phases_results = re.findall(
        r"Local parameters and GOALs for phase .+?\n(.*?)\n(?:\n|\Z)",
        texts,
        re.DOTALL,
    )

    result["phases_results"] = {
        phase_name: parse_section(phase_result)
        for phase_name, phase_result in zip(phase_names, phases_results)
    }
    return result


def parse_dia(dia_path: Path, phase_names: list[str]):
    if not dia_path.exists():
        raise FileNotFoundError(f"Cannot find the .dia file from {dia_path}")

    dia_text = dia_path.read_text().split("\n")

    raw_data = np.loadtxt(dia_text[1:])
    data = {
        "x": raw_data[:, 0],
        "y_obs": raw_data[:, 1],
        "y_calc": raw_data[:, 2],
        "y_bkg": raw_data[:, 3],
        "structs": {
            name: raw_data[:, i + 4].tolist() for i, name in enumerate(phase_names)
        },
    }

    return data


def get_phase_weights(result, normalize=True) -> dict[str, float]:
    weights = {}
    for phase, data in result["lst_data"]["phases_results"].items():
        weights[phase] = get_number(data["GEWICHT"])

    if normalize:
        tot = np.sum(list(weights.values()))
        weights = {k: v / tot for k, v in weights.items()}
    return dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))


def process_phase_name(phase_name: str) -> str:
    """Process the phase name to remove special characters."""
    return re.sub(r"[\s()_/\\+–\-*.]", "", phase_name)


def get_number(s: Union[float, None, tuple[float, float]]) -> Union[float, None]:
    """Get the number from a float or tuple of floats."""
    if isinstance(s, tuple):
        return s[0]
    elif isinstance(s, str):
        return float(re.search(r"(\d+\.\d+)", s).group(1))
    else:
        return s


def load_symmetrized_structure(
        cif_path: Path,
) -> tuple[SymmetrizedStructure, SpacegroupAnalyzer]:
    # suppress the warnings from pymatgen
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        structure = SpacegroupAnalyzer(
            Structure.from_file(cif_path.as_posix(), site_tolerance=1e-3)
        ).get_refined_structure()

    spg = SpacegroupAnalyzer(structure)
    structure: SymmetrizedStructure = spg.get_symmetrized_structure()
    return structure, spg


def read_phase_name_from_str(str_path: Path) -> str:
    """Get the phase name from the str file path.

    Example of str:
    PHASE=BaSnO3 // generated from pymatgen
    FORMULA=BaSnO3 //
    Lattice=Cubic HermannMauguin=P4/m-32/m Setting=1 SpacegroupNo=221 //
    PARAM=A=0.41168_0.40756^0.41580 //
    RP=4 PARAM=k1=0_0^1 k2=0 PARAM=B1=0_0^0.01 PARAM=GEWICHT=0_0 //
    GOAL:BaSnO3=GEWICHT //
    GOAL=GrainSize(1,1,1) //
    E=BA+2 Wyckoff=b x=0.500000 y=0.500000 z=0.500000 TDS=0.010000
    E=SN+4 Wyckoff=a x=0.000000 y=0.000000 z=0.000000 TDS=0.010000
    E=O-2 Wyckoff=d x=0.000000 y=0.000000 z=0.500000 TDS=0.010000
    """
    text = str_path.read_text()
    try:
        return re.search(r"PHASE=(\S*)", text).group(1)
    except AttributeError as e:
        raise ValueError(
            f"Could not find phase name in {str_path}. The content is: {text}"
        ) from e


def standardize_coords(x, y, z):
    # Adjust coordinates to specific fractional values if close
    fractions = {
        0.3333: 1 / 3,
        0.6667: 2 / 3,
        0.1667: 1 / 6,
        0.8333: 5 / 6,
        0.0833: 1 / 12,
        0.4167: 5 / 12,
        0.5833: 7 / 12,
        0.9167: 11 / 12,
    }

    for key, value in fractions.items():
        if abs(x - key) < 0.0001:
            x = value
        if abs(y - key) < 0.0001:
            y = value
        if abs(z - key) < 0.0001:
            z = value

    return x, y, z


def fuzzy_compare(a: float, b: float):
    fa = round(a, 6)
    fb = round(b, 6)

    # Normalizing the fractional parts to be within [0, 1]
    while fa < 0.0:
        fa += 1.0
    while fb < 0.0:
        fb += 1.0
    while fa >= 1.0:
        fa -= 1.0
    while fb >= 1.0:
        fb -= 1.0

    # Checking specific fractional values
    fractions = [
        (0.3333, 0.3334),  # 1/3
        (0.6666, 0.6667),  # 2/3
        (0.1666, 0.1667),  # 1/6
        (0.8333, 0.8334),  # 5/6
        (0.0833, 0.0834),  # 1/12
        (0.4166, 0.4167),  # 5/12
        (0.5833, 0.5834),  # 7/12
        (0.9166, 0.9167),  # 11/12
    ]

    for lower, upper in fractions:
        if lower <= fa <= upper and lower <= fb <= upper:
            return True

    # Fuzzy comparison for general case
    def is_close(_a, _b, rel_tol=0, abs_tol=1e-3):
        # Custom implementation of fuzzy comparison
        return abs(_a - _b) <= max(rel_tol * max(abs(_a), abs(_b)), abs_tol)

    return is_close(fa, fb)


def download_bgmn():
    bgmn_folder = Path(__file__).parent / "data"
    bgmn_folder.mkdir(parents=True, exist_ok=True)

    temp_file = bgmn_folder / ".bgmn-lock"

    while temp_file.exists():
        time.sleep(1)

        if (bgmn_folder / "BGMNwin").exists():
            return

    try:
        temp_file.touch()

        if (bgmn_folder / "BGMNwin").exists():
            return

        os_name = platform.system()

        if os_name not in ["Darwin", "Linux", "Windows"]:
            raise Exception("Unsupported OS: " + os_name + ".")

        URL = f"https://ocf.berkeley.edu/~yuxingfei/bgmn/bgmnwin_{os_name}.zip"

        r = requests.get(URL, stream=True)
        if r.status_code != 200:
            raise Exception(f"Cannot download from {URL}.")

        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with (bgmn_folder / "bgmnwin.zip").open("wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        with zipfile.ZipFile((bgmn_folder / "bgmnwin.zip").as_posix(), "r") as zip_ref:
            zip_ref.extractall(path=bgmn_folder.as_posix())

        os.remove((bgmn_folder / "bgmnwin.zip").as_posix())

        if os_name == "Linux" or os_name == "Darwin":
            os.system(f"chmod +x {bgmn_folder}/BGMNwin/bgmn")
    finally:
        temp_file.unlink(missing_ok=False)
