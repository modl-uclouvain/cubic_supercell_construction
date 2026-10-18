#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = "1.2.0"
__author__ = "Weiguo Jing"

from math import ceil, acos, degrees

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange
from pymatgen.core.structure import Structure


@njit(cache=True, fastmath=True)
def matmul3x3(A, B):
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]
    return C


@njit(cache=True, fastmath=True)
def get_supercell_length_and_angle(supercell_matrix):
    a, b, c = supercell_matrix
    length_a = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])**0.5
    length_b = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) ** 0.5
    length_c = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) ** 0.5
    alpha = (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) / length_b / length_c
    beta = (c[0] * a[0] + c[1] * a[1] + c[2] * a[2]) / length_c / length_a
    gamma = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / length_a / length_b
    alpha = degrees(acos(alpha))
    beta = degrees(acos(beta))
    gamma = degrees(acos(gamma))

    bcx = b[1] * c[2] - b[2] * c[1]
    bcy = b[2] * c[0] - b[0] * c[2]
    bcz = b[0] * c[1] - b[1] * c[0]
    l0 = abs(a[0] * bcx + a[1] * bcy + a[2] * bcz) ** (1/3)

    relative_delta_l = (((length_a / l0 - 1) ** 2 + (length_b / l0 - 1) ** 2 + (length_c / l0 - 1) ** 2) / 3) ** 0.5
    relative_delta_angle = (((alpha / 90 - 1) ** 2 + (beta / 90 - 1) ** 2 + (gamma / 90 - 1) ** 2) / 3) ** 0.5

    return length_a, length_b, length_c, alpha, beta, gamma, relative_delta_l, relative_delta_angle


@njit(cache=True, fastmath=True)
def deviation_from_components(
    a0, a1, a2,
    b0, b1, b2,
    c0, c1, c2,
):
    """Return the normalized surface-area deviation from a cube."""
    abx = a1 * b2 - a2 * b1
    aby = a2 * b0 - a0 * b2
    abz = a0 * b1 - a1 * b0
    ab_norm = (abx * abx + aby * aby + abz * abz) ** 0.5

    acx = a1 * c2 - a2 * c1
    acy = a2 * c0 - a0 * c2
    acz = a0 * c1 - a1 * c0
    ac_norm = (acx * acx + acy * acy + acz * acz) ** 0.5

    bcx = b1 * c2 - b2 * c1
    bcy = b2 * c0 - b0 * c2
    bcz = b0 * c1 - b1 * c0
    bc_norm = (bcx * bcx + bcy * bcy + bcz * bcz) ** 0.5

    surface = 2.0 * (ab_norm + ac_norm + bc_norm)
    volume = abs(a0 * bcx + a1 * bcy + a2 * bcz)
    return surface * volume ** (-2.0 / 3.0) / 6.0 - 1.0


@njit(fastmath=True, parallel=True)
def search_matrices_parallel(cell, r1, r2, r3, start, stop):
    """Search candidate matrices using Numba threads.

    Each thread keeps private minima and counts. This avoids data races; the
    private results are reduced after the parallel loop.
    """
    n_sizes = stop - start + 1
    n_threads = get_num_threads()

    thread_bias = np.full((n_threads, n_sizes), np.inf, dtype=np.float64)
    thread_count = np.zeros((n_threads, n_sizes), dtype=np.int64)
    thread_matrices = np.zeros(
        (n_threads, n_sizes, 3, 3), dtype=np.int64
    )

    n_diagonal = (r1 + 1) * (r2 + 1) * (r3 + 1)

    for diagonal_index in prange(n_diagonal):
        thread_id = get_thread_id()

        # Do not mutate a variable copied from the prange index. Some Numba
        # versions regard that as overwriting the parallel loop index.
        p33 = diagonal_index % (r3 + 1)
        p22 = (diagonal_index // (r3 + 1)) % (r2 + 1)
        p11 = diagonal_index // ((r3 + 1) * (r2 + 1))

        for p21 in range(-r1, r1 + 1):
            for p31 in range(-r1, r1 + 1):
                for p12 in range(-r2, r2 + 1):
                    for p32 in range(-r2, r2 + 1):
                        for p13 in range(-r3, r3 + 1):
                            for p23 in range(-r3, r3 + 1):
                                determinant = (
                                    p11 * (p22 * p33 - p23 * p32)
                                    - p12 * (p21 * p33 - p23 * p31)
                                    + p13 * (p21 * p32 - p22 * p31)
                                )
                                n = abs(determinant)

                                if n < start or n > stop:
                                    continue

                                idx = n - start
                                thread_count[thread_id, idx] += 1

                                # Rows of super_cell = P @ cell. Computing the
                                # nine components directly avoids allocating a
                                # small NumPy array for every candidate.
                                a0 = (
                                    p11 * cell[0, 0]
                                    + p12 * cell[1, 0]
                                    + p13 * cell[2, 0]
                                )
                                a1 = (
                                    p11 * cell[0, 1]
                                    + p12 * cell[1, 1]
                                    + p13 * cell[2, 1]
                                )
                                a2 = (
                                    p11 * cell[0, 2]
                                    + p12 * cell[1, 2]
                                    + p13 * cell[2, 2]
                                )
                                b0 = (
                                    p21 * cell[0, 0]
                                    + p22 * cell[1, 0]
                                    + p23 * cell[2, 0]
                                )
                                b1 = (
                                    p21 * cell[0, 1]
                                    + p22 * cell[1, 1]
                                    + p23 * cell[2, 1]
                                )
                                b2 = (
                                    p21 * cell[0, 2]
                                    + p22 * cell[1, 2]
                                    + p23 * cell[2, 2]
                                )
                                c0 = (
                                    p31 * cell[0, 0]
                                    + p32 * cell[1, 0]
                                    + p33 * cell[2, 0]
                                )
                                c1 = (
                                    p31 * cell[0, 1]
                                    + p32 * cell[1, 1]
                                    + p33 * cell[2, 1]
                                )
                                c2 = (
                                    p31 * cell[0, 2]
                                    + p32 * cell[1, 2]
                                    + p33 * cell[2, 2]
                                )

                                result = deviation_from_components(
                                    a0, a1, a2,
                                    b0, b1, b2,
                                    c0, c1, c2,
                                )

                                if result < thread_bias[thread_id, idx]:
                                    thread_bias[thread_id, idx] = result
                                    thread_matrices[thread_id, idx, 0, 0] = p11
                                    thread_matrices[thread_id, idx, 0, 1] = p12
                                    thread_matrices[thread_id, idx, 0, 2] = p13
                                    thread_matrices[thread_id, idx, 1, 0] = p21
                                    thread_matrices[thread_id, idx, 1, 1] = p22
                                    thread_matrices[thread_id, idx, 1, 2] = p23
                                    thread_matrices[thread_id, idx, 2, 0] = p31
                                    thread_matrices[thread_id, idx, 2, 1] = p32
                                    thread_matrices[thread_id, idx, 2, 2] = p33

    bias = np.full(n_sizes, np.inf, dtype=np.float64)
    count = np.zeros(n_sizes, dtype=np.int64)
    matrices = np.zeros((n_sizes, 3, 3), dtype=np.int64)

    for idx in range(n_sizes):
        for thread_id in range(n_threads):
            count[idx] += thread_count[thread_id, idx]
            if thread_bias[thread_id, idx] < bias[idx]:
                bias[idx] = thread_bias[thread_id, idx]
                matrices[idx, :, :] = thread_matrices[thread_id, idx, :, :]

    return bias, matrices, count


def find_transform_matrix(
    structure,
    num=10,
    start=1,
    stop=None,
    tolerance=1e-5,
    fast_search=False,
    write_log=False,
):
    """Find the transformation matrix producing the most cubic supercell."""
    if stop is None:
        stop = num
    if num < 1:
        raise ValueError("num must be at least 1")
    if start < 1:
        raise ValueError("start must be at least 1")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")

    cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
    volume = float(structure.volume)

    lengths = np.linalg.norm(cell, axis=1)
    max_length = num ** (1.0 / 3.0) * float(np.max(lengths))
    r1 = ceil(max_length / (volume / np.linalg.norm(np.cross(cell[1], cell[2]))))
    r2 = ceil(max_length / (volume / np.linalg.norm(np.cross(cell[2], cell[0]))))
    r3 = ceil(max_length / (volume / np.linalg.norm(np.cross(cell[0], cell[1]))))

    total = (
        (r1 + 1)
        * (r2 + 1)
        * (r3 + 1)
        * (2 * r1 + 1) ** 2
        * (2 * r2 + 1) ** 2
        * (2 * r3 + 1) ** 2
    )

    print(
        f"Searching {total} candidate matrices with "
        f"{get_num_threads()} Numba threads...",
        flush=True,
    )

    if write_log:
        with open("log.txt", "w", encoding="utf-8") as file:
            file.write("0.00%\n")

    bias, matrix_array, count_array = search_matrices_parallel(
        cell, r1, r2, r3, start, stop
    )

    print("Search completed.", flush=True)
    with open("log.txt", "w", encoding="utf-8") as file:
        file.write("100%\n")

    found = np.isfinite(bias)
    if not np.any(found):
        raise RuntimeError(
            f"No transformation matrix with |det(P)| in [{start}, {stop}] "
            "was found. Increase num."
        )

    matrixes = [
        matrix_array[i].copy() if found[i] else None
        for i in range(stop - start + 1)
    ]
    count = count_array.tolist()
    valid = int(np.sum(count_array))

    with open("p_matrix.txt", "w", encoding="utf-8") as file:
        file.write(f"{total} matrix are generated.\n\n")
        file.write(
            f"{valid} supercells are constructed, the abs(det) of which "
            f"is in the range [{start}, {stop}].\n\n"
        )
        file.write(
            "IMPORTANT: all p_matrix here corresponds to the case that all "
            "three lattice vectors are in ROW vector format (for example, "
            "POSCAR in VASP). If you use VESTA to construct the supercell "
            "(which needs COLUMN vector format), you NEED TO TRANSPOSE the "
            "p_matrix!\n\n"
        )
        file.write(
            "The bias = S * V ** (-2/3) / 6 - 1. Parameters S and V "
            "correspond to the surface area and volume of the supercell, "
            "respectively.\n\n"
        )

        for i in range(len(bias)):
            n = i + start
            file.write(f"{n}\n")
            file.write(f"total supercells P({n}): {count[i]}\n")
            if matrixes[i] is None:
                file.write("bias: not found\n")
                file.write("matrix: not found\n\n")
            else:
                file.write(f"bias: {bias[i]:.8f}\n")
                file.write(f"{matrixes[i]}\n\n")

        file.write('supercell size, bias, total P matrix, a, b, c, alpha, beta, gamma, ΔL/V**(1/3), Δθ/90°\n\n')
        for i in range(len(bias)):
            if found[i]:
                p_matrix = np.array(matrixes[i])
                super_cell = matmul3x3(p_matrix, cell)
                supercell_information = get_supercell_length_and_angle(super_cell)
                file.write(f'{i + start}, {bias[i]:.7f}, {count[i]}, '
                           f'{supercell_information[0]:.5f}, {supercell_information[1]:.5f}, {supercell_information[2]:.5f}, '
                           f'{supercell_information[3]:.4f}, {supercell_information[4]:.4f}, {supercell_information[5]:.4f}, '
                           f'{supercell_information[6]:.7f}, {supercell_information[7]:.7f}\n')
            else:
                file.write(f"{i + start} not_found\n")

    if fast_search:
        qualified = np.flatnonzero(bias < tolerance)
        if qualified.size:
            # Parallel execution has no deterministic serial "first" result.
            # Return the smallest determinant that satisfies the tolerance.
            return matrixes[int(qualified[0])]

    index = int(np.argmin(bias))
    return matrixes[index]


def build_cubic_supercell(
    primitive: Structure,
    num: int = 10,
    start: int = 1,
    stop: int | None = None,
    fast_search: bool = False,
    tolerance: float = 1e-5,
    write_log: bool = False,
):
    """Build and write the searched cubic-like supercell."""
    transform = find_transform_matrix(
        primitive,
        num=num,
        start=start,
        stop=stop,
        tolerance=tolerance,
        fast_search=fast_search,
        write_log=write_log,
    )

    result = transform @ primitive.lattice.matrix
    if result[0, 0] < 0:
        transform[0] *= -1
    if result[1, 1] < 0:
        transform[1] *= -1
    if result[2, 2] < 0:
        transform[2] *= -1

    transform = np.asarray(transform, dtype=int)
    cubic_supercell = primitive.copy()
    cubic_supercell.make_supercell(transform)
    n = int(round(abs(np.linalg.det(transform))))
    cubic_supercell.to(
        filename=f"POSCAR_supercell_P{n}.vasp",
        fmt="poscar",
    )
    return cubic_supercell


if __name__ == "__main__":
    filename = "CaMg(CO3)2.vasp"
    primitive_cell = Structure.from_file(filename)
    supercell = build_cubic_supercell(primitive_cell, num=100)
