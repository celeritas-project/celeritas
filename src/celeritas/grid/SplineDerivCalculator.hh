//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineDerivCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/grid/UniformGrid.hh"

#include "XsGridData.hh"

#include "detail/GridAccessor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the second derivatives of an interpolating cubic spline.
 *
 * This uses not-a-knot boundary conditions. TODO: description
 */
class SplineDerivCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using UPGridAccessor = std::unique_ptr<detail::GridAccessor>;
    using SpanConstReal = detail::SpanGridAccessor::SpanConstReal;
    using Values = detail::XsGridAccessor::Values;
    using VecReal = std::vector<real_type>;
    //!@}

    //! Cubic spline interpolation boundary conditions
    enum class BoundaryCondition
    {
        natural = 0,
        not_a_knot,
        geant,  //!< Not not-a-knot
        size_
    };

  public:
    // Contruct with x and y grids and boundary type
    SplineDerivCalculator(SpanConstReal, SpanConstReal, BoundaryCondition);

    // Contruct with grid data and boundary type
    SplineDerivCalculator(XsGridData const&, Values const&, BoundaryCondition);

    // Calculate the second derivatives
    VecReal operator()() const;

  private:
    //// TYPES ////

    using Real4 = Array<real_type, 4>;

    //// DATA ////

    UPGridAccessor grid_;
    BoundaryCondition bc_;

    //// HELPER FUNCTIONS ////

    void calc_initial_row(Real4&) const;
    void calc_final_row(Real4&) const;
    void calc_boundaries(VecReal&) const;
    VecReal calc_geant_derivatives() const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
