//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/TridiagonalSolver.cc
//---------------------------------------------------------------------------//
#include "TridiagonalSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Contruct with coefficients.
 */
TridiagonalSolver::TridiagonalSolver(Coefficients coeffs) : coeffs_{coeffs}
{
    CELER_EXPECT(coeffs_.a.size() >= 2);
    CELER_EXPECT(coeffs_.a.size() == coeffs_.b.size()
                 && coeffs_.b.size() == coeffs_.c.size()
                 && coeffs_.c.size() == coeffs_.d.size());
}

//---------------------------------------------------------------------------//
/*!
 * Solve the tridiagonal system.
 */
auto TridiagonalSolver::operator()() const -> VecReal
{
    size_type num_rows = coeffs_.a.size();
    VecReal result(num_rows);
    VecReal c_prime(num_rows);

    c_prime[0] = coeffs_.c[0] / coeffs_.b[0];
    result[0] = coeffs_.d[0] / coeffs_.b[0];

    // Forward sweep
    for (size_type i = 1; i < num_rows; ++i)
    {
        real_type factor = 1 / (coeffs_.b[i] - coeffs_.a[i] * c_prime[i - 1]);
        c_prime[i] = coeffs_.c[i] * factor;
        result[i] = (coeffs_.d[i] - coeffs_.a[i] * result[i - 1]) * factor;
    }

    // Back substitution
    for (int i = num_rows - 2; i >= 0; --i)
    {
        result[i] -= c_prime[i] * result[i + 1];
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
