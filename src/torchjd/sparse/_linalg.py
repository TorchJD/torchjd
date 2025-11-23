import torch
from torch import Tensor

# TODO: Implement in C everything in this file.


def solve_int(A: Tensor, B: Tensor, tol=1e-9) -> Tensor | None:
    """
    Solve A X = B where A, B and X have integer dtype.
    Return X if such a matrix exists and otherwise None.
    """

    A_ = A.to(torch.float64)
    B_ = B.to(torch.float64)

    try:
        X = torch.linalg.solve(A_, B_)
    except RuntimeError:
        return None

    X_rounded = X.round()
    if not torch.all(torch.isclose(X, X_rounded, atol=tol)):
        return None

    # TODO: Verify that the round operation cannot fail
    return X_rounded.to(torch.int64)


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean Algorithm (Python integers).
    Returns (g, x, y) such that a*x + b*y = g.
    """
    # We perform the logic in standard Python int for speed on scalars
    # then cast back to torch tensors if needed, or return python ints.
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y


def hnf_decomposition(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes the reduced Hermite Normal Form decomposition using PyTorch. For a matrix A (m x n) of
    rank r, computes the matrices H (m x r), U (n x r) and V (r x n) such that
        V U = I_r
        A = H V
        H = A U

    Args:
        A: (m x n) torch.Tensor (dtype=torch.long)

    Returns:
        H: (m x r) Canonical Lower Triangular HNF
        U: (n x r) Unimodular transform (A @ U = H)
        V: (r x n) Right inverse Unimodular transform (H @ V = A)
    """

    H = A.clone().to(dtype=torch.long)
    m, n = H.shape

    U = torch.eye(n, dtype=torch.long)
    V = torch.eye(n, dtype=torch.long)

    row = 0
    col = 0

    while row < m and col < n:
        # --- 1. Pivot Selection ---
        # Find first non-zero entry in current row from col onwards
        pivot_idx = -1

        # We extract the row slice to CPU for faster scalar checks if on GPU
        # or just iterate. For HNF, strictly sequential loop is often easiest.
        for j in range(col, n):
            if H[row, j] != 0:
                pivot_idx = j
                break

        if pivot_idx == -1:
            row += 1
            continue

        # Swap to current column
        if pivot_idx != col:
            # Swap Columns in H and U
            H[:, [col, pivot_idx]] = H[:, [pivot_idx, col]]
            U[:, [col, pivot_idx]] = U[:, [pivot_idx, col]]
            # Swap ROWS in V
            V[[col, pivot_idx], :] = V[[pivot_idx, col], :]

        # --- 2. Gaussian Elimination via GCD ---
        for j in range(col + 1, n):
            if H[row, j] != 0:
                # Extract values as python ints for GCD logic
                a_val = H[row, col].item()
                b_val = H[row, j].item()

                g, x, y = extended_gcd(a_val, b_val)

                # Bezout: a*x + b*y = g
                # c1 = -b // g, c2 = a // g
                c1 = -b_val // g
                c2 = a_val // g

                # --- Update H (Column Ops) ---
                # Important: Clone columns to avoid in-place modification issues during calc
                col_c = H[:, col].clone()
                col_j = H[:, j].clone()

                H[:, col] = col_c * x + col_j * y
                H[:, j] = col_c * c1 + col_j * c2

                # --- Update U (Column Ops) ---
                u_c = U[:, col].clone()
                u_j = U[:, j].clone()
                U[:, col] = u_c * x + u_j * y
                U[:, j] = u_c * c1 + u_j * c2

                # --- Update V (Inverse Row Ops) ---
                # Inverse of [[x, c1], [y, c2]] is [[c2, -c1], [-y, x]]
                v_r_c = V[col, :].clone()
                v_r_j = V[j, :].clone()
                V[col, :] = v_r_c * c2 - v_r_j * c1
                V[j, :] = v_r_c * (-y) + v_r_j * x

        # --- 3. Enforce Positive Diagonal ---
        if H[row, col] < 0:
            H[:, col] *= -1
            U[:, col] *= -1
            V[col, :] *= -1

        # --- 4. Canonical Reduction (Modulo) ---
        # Ensure 0 <= H[row, k] < H[row, col] for k < col
        pivot_val = H[row, col].clone()
        if pivot_val != 0:
            for j in range(col):
                # floor division
                factor = torch.div(H[row, j], pivot_val, rounding_mode="floor")

                if factor != 0:
                    H[:, j] -= factor * H[:, col]
                    U[:, j] -= factor * U[:, col]
                    V[col, :] += factor * V[j, :]

        row += 1
        col += 1

    col_magnitudes = torch.sum(torch.abs(H), dim=0)
    non_zero_indices = torch.nonzero(col_magnitudes, as_tuple=True)[0]

    if len(non_zero_indices) == 0:
        rank = 0
    else:
        rank = non_zero_indices.max().item() + 1

    reduced_H = H[:, :rank]
    reduced_U = U[:, :rank]
    reduced_V = V[:rank, :]

    return reduced_H, reduced_U, reduced_V


def compute_gcd(S1: Tensor, S2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes the GCD and the projection factors. i.e.
    S1 = G @ K1
    S2 = G @ K2
    with G having minimal rank.

    Args:
        S1, S2: torch.Tensors (m x n1), (m x n2)

    Returns:
        G: (m x r) The Matrix GCD (Canonical Base)
        K1: (r x n1) Factors for S1
        K2: (r x n2) Factors for S2
    """
    assert S1.shape[0] == S2.shape[0], "Virtual dimension mismatch"
    m, n1 = S1.shape

    A = torch.cat([S1, S2], dim=1)
    G, U, V = hnf_decomposition(A)

    # H = [S1 | S2] @ U
    # [S1 | S2] = H @ V
    #
    # S1 = H @ V[:, :m1]
    # S2 = H @ V[:, m1:]
    #
    # K1 = V[:, :m1]
    # K2 = V[:, m1:]
    # G = H
    #
    # S1 = G @ K1
    # S2 = G @ K2
    #
    # SLT(p1, S1) = SLT(SLT(p1, K1), G)
    # SLT(p2, S2) = SLT(SLT(p2, K2), G)

    K1 = V[:, :n1]
    K2 = V[:, n1:]

    return G, K1, K2


def compute_lcm(S1, S2):
    """
    Computes the Matrix LCM (L) and the Multiples (M1, M2), i.e.
    L = S1 @ M1 = S2 @ M2

    Returns:
        L:  (m x m)  The Matrix LCM
        M1: (n1 x m) Factor such that L = S1 @ M1
        M2: (n2 x m) Factor such that L = S2 @ M2
    """
    m = S1.shape[0]
    n1 = S1.shape[1]

    # 1. Kernel Setup: [S1 | -S2]
    B = torch.cat([S1, -S2], dim=1)

    # 2. Decompose to find Kernel
    H_B, U_B, _ = hnf_decomposition(B)

    # 3. Find Zero Columns in H_B (Kernel basis)
    # Sum abs values down columns
    col_mags = torch.sum(torch.abs(H_B), dim=0)
    zero_indices = torch.nonzero(col_mags == 0, as_tuple=True)[0]

    if len(zero_indices) == 0:
        return torch.zeros((m, m), dtype=torch.long)

    # 4. Extract Kernel Basis
    # U_B columns corresponding to H_B zeros are the kernel generators
    kernel_basis = U_B[:, zero_indices]

    # 5. Map back to Image Space
    # The kernel vector is [u; v]. We need u (top n1 rows).
    # Intersection = S1 @ u
    u_parts = kernel_basis[:n1, :]
    L_generators = S1 @ u_parts

    # 6. Canonicalize L
    # The generators might be redundant or non-square.
    # Run HNF one last time to get the unique square LCM matrix.
    L, _, _ = hnf_decomposition(L_generators)

    return L[:, :m]
