//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CalculatorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "celeritas/grid/XsGridData.hh"

#include "Test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for interpolating values on grids.
 */
class CalculatorTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using BC = SplineDerivCalculator::BoundaryCondition;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using Data
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;
    using SpanReal = Span<real_type>;
    using VecReal = std::vector<real_type>;
    //!@}

    struct GridInput
    {
        real_type emin{0};
        real_type emax{0};
        VecReal value;
        size_type spline_order{1};
    };

  public:
    // Construct from grid bounds and cross section values
    void build(GridInput grid, GridInput grid_scaled);
    void build_spline(GridInput grid, GridInput grid_scaled, BC bc);

    // Construct without scaled values
    void build(GridInput grid);
    void build_spline(GridInput grid, BC bc);

    XsGridRecord const& data() const { return data_; }
    Data const& values() const { return value_ref_; }

  private:
    XsGridRecord data_;
    Values value_storage_;
    Data value_ref_;

    void build_impl(GridInput grid, GridInput grid_scaled, BC bc);
    void build_impl(GridInput grid, BC bc);
    void build_grid(UniformGridRecord& data, GridInput const& grid, BC bc);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
