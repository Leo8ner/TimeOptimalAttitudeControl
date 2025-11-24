#!/usr/bin/env python3
"""
plot_states_controls.py

Reads a custom CSV file of the form

X
row0,row0,row0,...
row1,row1,...
...
U
row0,row0,...
...
T
2.50663
dt
0.0250663

and produces:
  • Fig 1 : X‑rows 0–2    (3 curves)
  • Fig 2 : X‑rows 3–6    (4 curves)
  • Fig 3 : X‑rows 7–9    (3 curves)
  • Fig 4 : every U‑row   (all control curves)

Usage
-----
$ python plot_states_controls.py path/to/data.csv
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 14


def load_custom_csv(path: str | Path) -> dict[str, list[list[float]]]:
    """
    Parse the bespoke CSV into a dict:
        {"X": [[...], [...], ...],
         "U": [[...], [...], ...],
         "T": float,
         "dt": float}
    """
    path = Path(path)
    out: dict[str, list[list[float]] | float] = {}

    with path.open(newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader]

    i = 0
    n = len(rows)
    while i < n:
        if not rows[i] or not rows[i][0].strip():
            i += 1
            continue

        header = rows[i][0].strip()
        i += 1

        # scalar headers ------------------------------------------------------
        if header.lower() in {"t"}:
            # next non‑empty line is the value
            while i < n and (not rows[i] or not rows[i][0].strip()):
                i += 1
            if i >= n:
                raise ValueError(f"Missing value after header '{header}'.")
            out[header] = float(rows[i][0])
            i += 1
            continue

        # vector / matrix headers ---------------------------------------------
        data_rows: list[list[float]] = []
        while i < n and rows[i] and rows[i][0].strip() \
                and rows[i][0] not in {"X", "U", "T", "dt"}:
            data_rows.append([float(x) for x in rows[i] if x])
            i += 1

        out[header] = data_rows

    return out  # type: ignore[return-value]


def main(csv_path: str | Path) -> None:
    data = load_custom_csv(csv_path)

    X_rows: list[list[float]] = data["X"]  # type: ignore[index]
    U_rows: list[list[float]] = data["U"]  # type: ignore[index]
    T: float = data["T"]  # type: ignore[assignment]
    dt_rows: list[list[float]] = data["dt"]  # type: ignore[index]
    n_steps = len(X_rows[0])  # number of time steps
    if len(dt_rows) == 1:
        # Fixed dt case
        dt = dt_rows[0][0]
        time = np.linspace(0, T, n_steps)
    else:
        # Variable dt case
        dt = np.array(dt_rows).flatten()
        time = np.insert(np.cumsum(dt), 0, 0.0)  # insert t=0 at start

    # Convert to arrays for convenience
    X = np.vstack(X_rows)   # shape (10, N)
    U = np.vstack(U_rows)   # shape (m,  N)

    # ------------------------- plotting --------------------------------------
    plt.figure()
    plt.plot(time, X[0], label=r"$\phi$")
    plt.plot(time, X[1], label=r"$\theta$")
    plt.plot(time, X[2], label=r"$\psi$")
    plt.xlabel("Time  [s]", fontsize=FONTSIZE)
    plt.ylabel("Euler Angles [deg]", fontsize=FONTSIZE)
    plt.title("Attitude with time")
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig("../output/euler_angles.pdf" , format='pdf', dpi=600, bbox_inches='tight')

    plt.figure()
    plt.plot(time, X[3], label=r"$q_0$")
    plt.plot(time, X[4], label=r"$q_1$")
    plt.plot(time, X[5], label=r"$q_2$")
    plt.plot(time, X[6], label=r"$q_3$")
    plt.xlabel("Time  [s]", fontsize=FONTSIZE)
    plt.ylabel("Quaternions [-]", fontsize=FONTSIZE)
    plt.title("Quaternions with time")
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig("../output/quaternions.pdf" , format='pdf', dpi=600, bbox_inches='tight')


    plt.figure()
    plt.plot(time, X[7]*180/np.pi, label=r'$\omega_x$')
    plt.plot(time, X[8]*180/np.pi, label=r"$\omega_y$")
    plt.plot(time, X[9]*180/np.pi, label=r"$\omega_z$")
    plt.xlabel("Time  [s]", fontsize=FONTSIZE)
    plt.ylabel("Angular Rates [deg/s]", fontsize=FONTSIZE)
    plt.title("Angular rates with time")
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig("../output/angular_rates.pdf" , format='pdf', dpi=600, bbox_inches='tight')



    # Controls ---------------------------------------------------------------
    plt.figure()
    plt.plot(time, np.insert(U[0], 0, 0), label=r"$\tau_x$")
    plt.plot(time, np.insert(U[1], 0, 0), label=r"$\tau_y$")
    plt.plot(time, np.insert(U[2], 0, 0), label=r"$\tau_z$")
    plt.xlabel("Time  [s]", fontsize=FONTSIZE)
    plt.ylabel("Torque [Nm]", fontsize=FONTSIZE)
    plt.title("Control Inputs")
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig("../output/control_inputs.pdf" , format='pdf', dpi=600, bbox_inches='tight')

    print("Plotting done. Check the output directory for results.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_csv_data.py trajectory.csv")
        sys.exit(1)
    main("../output/" + sys.argv[1])
