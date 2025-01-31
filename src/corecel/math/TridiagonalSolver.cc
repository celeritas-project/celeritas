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
 *
 * The first three coefficients are the bands of the tridiagonal matrix and the
 * last is the right-hand side.
 */
TridiagonalSolver::TridiagonalSolver(Coeffs&& coeffs) : coeffs_{coeffs}
{
    CELER_EXPECT(coeffs_.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Solve the tridiagonal system.
 */
void TridiagonalSolver::operator()(SpanReal dst) const
{
    CELER_EXPECT(dst.size() == coeffs_.size());

    std::vector<real_type> c_prime(coeffs_.size());
    c_prime[0] = coeffs_[0][2] / coeffs_[0][1];
    dst[0] = coeffs_[0][3] / coeffs_[0][1];

    // Forward sweep
    for (size_type i = 1; i < coeffs_.size(); ++i)
    {
        auto const& a = coeffs_[i];
        real_type factor = 1 / (a[1] - a[0] * c_prime[i - 1]);
        c_prime[i] = a[2] * factor;
        dst[i] = (a[3] - a[0] * dst[i - 1]) * factor;
    }

    // Back substitution
    for (size_type i = coeffs_.size() - 2; i != size_type(-1); --i)
    {
        dst[i] -= c_prime[i] * dst[i + 1];
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
