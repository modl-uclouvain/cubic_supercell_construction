#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = '1.1.3'
__author__ = 'Weiguo Jing'

import numpy as np
from itertools import product
from math import ceil, acos, degrees
from numba import njit
from pymatgen.core.structure import Structure


@njit(cache=True, fastmath=True)
def det3(m):
    return (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


@njit(cache=True, fastmath=True)
def matmul3x3(A, B):
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]
    return C


def find_transform_matrix(structure, num=10, start=1, stop=None, tolerance=1e-5, fast_search=False, write_log=False):
    from numpy import cross
    from numpy.linalg import norm

    if stop is None:
        stop = num

    cell = structure.lattice.matrix
    volume = structure.volume

    # Precompute projection lengths
    max_length = pow(num, 1.0 / 3) * max(norm(cell[0]), norm(cell[1]), norm(cell[2]))
    r1 = ceil(max_length / (volume / norm(cross(cell[1], cell[2]))))
    r2 = ceil(max_length / (volume / norm(cross(cell[2], cell[0]))))
    r3 = ceil(max_length / (volume / norm(cross(cell[0], cell[1]))))

    bias = [np.inf] * (stop - start + 1)
    matrixes = [None] * (stop - start + 1)
    count = [0] * (stop - start + 1)

    # Cache indices
    range_r1 = range(-r1, r1 + 1)
    range_r2 = range(-r2, r2 + 1)
    range_r3 = range(-r3, r3 + 1)

    total = (2 * r1 + 1) ** 2 * (2 * r2 + 1) ** 2 * (2 * r3 + 1) ** 2 * (r1 + 1) * (r2 + 1) * (r3 + 1)
    i = 0
    valid = 0

    for P11 in range(0, r1 + 1):
        for P22 in range(0, r2 + 1):
            for P33 in range(0, r3 + 1):
                for P21, P31, P12, P32, P13, P23 in product(range_r1, range_r1, range_r2, range_r2, range_r3, range_r3):
                    pmatrix = np.array([
                        [P11, P12, P13],
                        [P21, P22, P23],
                        [P31, P32, P33]
                    ])
                    n = int(abs(det3(pmatrix)))
                    i += 1
                    if n > stop or n < start:
                        continue
                    # super_cell = np.dot(pmatrix, cell)
                    super_cell = matmul3x3(pmatrix, cell)

                    result = deviation(super_cell)

                    if fast_search and result < tolerance:
                        with open('log.txt', 'w') as file:
                            file.write('100%')
                        return pmatrix

                    idx = n - start
                    if result < bias[idx]:
                        bias[idx] = result
                        matrixes[idx] = pmatrix
                    count[idx] += 1

            percent = 100 * i / total
            if write_log:
                with open('log.txt', 'w') as file:
                    file.write(f'{percent:.2f}%\n')
            else:
                print(f'{percent:.2f}%', end='\r')

    with open('log.txt', 'w') as file:
        file.write('100%\n')

    valid = sum(count)
    with open('p_matrix.txt', 'w') as file:
        file.write(f'{total} matrix are generated.\n\n')
        file.write(f"{valid} supercells are constructed, the abs(det) of which is in the range [{start}, {stop}].\n\n")
        file.write("IMPORTANT: all p_matrix here corresponds to the case that all three lattice vectors are "
                   "in ROW vector format (For example, POSCAR in VASP). If you use VESTA to construct the supercell "
                   "(need COLUMN vector format), you NEED TO TRANSPOSE the p_matrix!\n\n")
        file.write("The bias = S * V ** (-2/3) / 6 - 1. Parameters S and V correspond to "
                   "the surface and the volume of supercells(N), respectively.")
        for i in range(len(bias)):
            file.write(f'{i + start}\n')
            file.write(f'total supercells P({i + start}): {count[i]}\n')
            file.write(f'bias: {bias[i]:.8f}\n')
            file.write(f'{matrixes[i]}\n\n')
        file.write('supercell size, bias, total P matrix, a, b, c, alpha, beta, gamma, ΔL/V**(1/3), Δθ/90°\n\n')
        for i in range(len(bias)):
            p_matrix = np.array(matrixes[i])
            super_cell = matmul3x3(p_matrix, cell)
            supercell_information = get_supercell_length_and_angle(super_cell)
            file.write(f'{i + start}, {bias[i]:.7f}, {count[i]}, '
                       f'{supercell_information[0]:.5f}, {supercell_information[1]:.5f}, {supercell_information[2]:.5f}, '
                       f'{supercell_information[3]:.4f}, {supercell_information[4]:.4f}, {supercell_information[5]:.4f}, '
                       f'{supercell_information[6]:.7f}, {supercell_information[7]:.7f}\n')

    index = np.argmin(bias)
    return matrixes[index]


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
def deviation(cell):
    a, b, c = cell

    abx = a[1]*b[2] - a[2]*b[1]
    aby = a[2]*b[0] - a[0]*b[2]
    abz = a[0]*b[1] - a[1]*b[0]
    ab_norm = (abx**2 + aby**2 + abz**2)**0.5

    acx = a[1]*c[2] - a[2]*c[1]
    acy = a[2]*c[0] - a[0]*c[2]
    acz = a[0]*c[1] - a[1]*c[0]
    ac_norm = (acx**2 + acy**2 + acz**2)**0.5

    bcx = b[1]*c[2] - b[2]*c[1]
    bcy = b[2]*c[0] - b[0]*c[2]
    bcz = b[0]*c[1] - b[1]*c[0]
    bc_norm = (bcx**2 + bcy**2 + bcz**2)**0.5

    surface = 2 * (ab_norm + ac_norm + bc_norm)

    # Volume = |a . (b x c)|
    vol = abs(a[0]*bcx + a[1]*bcy + a[2]*bcz)

    return surface * vol ** (-2/3) / 6 - 1


def build_cubic_supercell(
        primitive: Structure,
        num: int = 10,
        start: int = 1,
        stop: int | None = None,
        fast_search: bool = False,
        tolerance: float = 1e-5,
        write_log: bool = False,
):
    """
    To transform a non-cubic primitive to a supercell as close to cube as possible. A geometric properties is used to \
    judge whether a parallel hexahedron is close to a cubic that, in given volume of parallel hexahedron, cubic has the\
     smallest surface area.

    :param primitive: 'pymatgen.core.Structure'
    The structure of primitive cell. The lattice vectors are stored as three *row* vectors.
    :param num: 'int'
    It decides the element range in loop matrix, whose value is proportional to the maximum size of supercell. When \
    stop para is default, the maximum size of supercell will be chosen as the value of num para. It is extremly useful \
    when you want to search the supercell in a large range, because you can choose a smaller value of num for smaller \
    loop times.
    :param start: 'int'
    The minimun sizes of supercell to primitive, default is 1, i.e. same volume as primitive cell.
    :param stop: 'int'
    The maximum sizes of supercell to primitive cell. Default value is None. When this value is default, this program \
    will choose the value of num as the maximum size of primitive cell.
    :param fast_search: 'bool'
    When the switch of fast search scheme is on, the program will finish when find a cell satisfies the judge tolerance.
    :param tolerance: 'float'
    The tolerance to stop the loop of find transform matrix. The supercell will be considered as cube when the \
    deviation is smaller than tolerance.
    :param write_log: 'bool'
    If set True, the program will generate a log file to show the progress bar.
    :return: 'Pylada.crystal.Structure'
    The structure of supercell.
    """
    from numpy import dot
    from numpy.linalg import det

    transform = find_transform_matrix(primitive, num=num, start=start, stop=stop, tolerance=tolerance,
                                      fast_search=fast_search, write_log=write_log)
    result = dot(transform, primitive.lattice.matrix)
    if result[0][0] < 0:
        transform[0] = -1 * transform[0]
    if result[1][1] < 0:
        transform[1] = -1 * transform[1]
    if result[2][2] < 0:
        transform[2] = -1 * transform[2]
    transform = np.array(transform, dtype=int)
    cubic_supercell = primitive.copy()
    cubic_supercell.make_supercell(transform)
    n = int(0.01 + abs(det(transform)))
    cubic_supercell.to(filename=f"POSCAR_supercell_P{n}.vasp", fmt="poscar")

    return cubic_supercell


if __name__ == "__main__":
    filename = "CaO.vasp"
    primitive_cell = Structure.from_file(filename)
    supercell = build_cubic_supercell(primitive_cell, num=10)

