//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineDerivativeCalculator.hh
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
 * Calculate the first derivatives of an interpolating cubic spline.
 *
 * This uses not-a-knot boundary conditions. TODO: description
 */
class SplineDerivativeCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using UPGridAccessor = std::unique_ptr<detail::GridAccessor>;
    using SpanConstReal = detail::SpanGridAccessor::SpanConstReal;
    using Values = detail::XsGridAccessor::Values;
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Contruct with x and y grids
    SplineDerivativeCalculator(SpanConstReal x_values, SpanConstReal y_values);

    // Contruct with cross section grid
    SplineDerivativeCalculator(XsGridData const& grid, Values const& values);

    // Calculate the first derivatives
    VecReal operator()() const;

  private:
    UPGridAccessor grid_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
