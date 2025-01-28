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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Solve a tridiagonal system of equations using the Thomas algorithm.
 *
 * This is a simplified form of Gaussian elimination that can solve a
 * tridiagonal system \f$ a_i x_{i - 1} + b_i x_i + c_i x_{i + 1} = d_i \f$
 * with \f$ n \f$ unknowns where \f$ a_1 = 0 \f$ and \f$ c_n = 0 \f$ in O(n)
 * time.
 */
class TridiagonalSolver
{
  public:
    //!@{
    //! \name Type aliases
    using VecReal = std::vector<real_type>;
    //!@}

    struct Coefficients
    {
        VecReal a;  //!< Subdiagonal
        VecReal b;  //!< Diagonal
        VecReal c;  //!< Superdiagonal
        VecReal d;  //!< RHS
    };

  public:
    // Contruct with coefficients
    explicit TridiagonalSolver(Coefficients coeffs);

    // Solve the tridiagonal system
    VecReal operator()() const;

  private:
    Coefficients coeffs_;
};

//---------------------------------------------------------------------------//
// Resize the coefficient vectors
void resize(TridiagonalSolver::Coefficients&, size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
