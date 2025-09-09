//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/TridiagonalSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Solve a tridiagonal system of equations using the Thomas algorithm.
 *
 * This is a simplified form of Gaussian elimination that can solve a
 * tridiagonal system \f$ \mathbf{T} \mathbf{x} = \mathbf{b} \f$ in O(n) time.
 */
class TridiagonalSolver
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<real_type, 3>;
    using Coeffs = std::vector<Real3>;
    using SpanConstReal = Span<real_type const>;
    using SpanReal = Span<real_type>;
    //!@}

  public:
    // Construct with coefficients
    explicit TridiagonalSolver(Coeffs&& tridiag);

    // Solve the tridiagonal system
    void operator()(SpanConstReal rhs, SpanReal x) const;

  private:
    Coeffs tridiag_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
