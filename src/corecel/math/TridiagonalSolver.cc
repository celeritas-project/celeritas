//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/TridiagonalSolver.cc
//---------------------------------------------------------------------------//
#include "TridiagonalSolver.hh"

#include <utility>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the tridiagonal matrix coefficients.
 */
TridiagonalSolver::TridiagonalSolver(Coeffs&& tridiag)
    : tridiag_{std::move(tridiag)}
{
    CELER_EXPECT(tridiag_.size() >= 2);
}

//---------------------------------------------------------------------------//
/*!
 * Solve the tridiagonal system.
 */
void TridiagonalSolver::operator()(SpanConstReal rhs, SpanReal x) const
{
    CELER_EXPECT(x.size() == tridiag_.size());

    std::vector<real_type> c_prime(x.size());
    c_prime[0] = tridiag_[0][2] / tridiag_[0][1];
    x[0] = rhs[0] / tridiag_[0][1];

    // Forward sweep
    for (size_type i = 1; i < x.size(); ++i)
    {
        auto const& a = tridiag_[i];
        real_type factor = 1 / (a[1] - a[0] * c_prime[i - 1]);
        c_prime[i] = a[2] * factor;
        x[i] = (rhs[i] - a[0] * x[i - 1]) * factor;
    }

    // Back substitution
    for (size_type ip = x.size() - 1; ip > 0; --ip)
    {
        x[ip - 1] -= c_prime[ip - 1] * x[ip];
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
