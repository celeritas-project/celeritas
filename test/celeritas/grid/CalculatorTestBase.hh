//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CalculatorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/grid/SplineDerivCalculator.hh"
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
    using XsFunc = std::function<real_type(real_type)>;
    using Real2 = Array<real_type, 2>;
    //!@}

  public:
    //!@{
    //! Deprecated: use the "build" (with function) and "convert_to_prime"
    // Construct linear cross sections
    void build(real_type emin, real_type emax, size_type count);
    void set_prime_index(size_type i);
    SpanReal mutable_values();
    //!@}

    // Construct from an arbitrary function
    void build(Real2 bounds, size_type count, XsFunc calc_xs)
    {
        return this->build_impl(bounds, count, std::move(calc_xs), BC::size_);
    }
    void build_spline(Real2 bounds, size_type count, XsFunc calc_xs, BC bc)
    {
        CELER_EXPECT(bc != BC::size_);
        return this->build_impl(bounds, count, std::move(calc_xs), bc);
    }

    // Scale cross sections at or above this index by a factor of E
    void convert_to_prime(size_type i);

    XsGridData const& data() const { return data_; }
    Data const& values() const { return value_ref_; }

  private:
    XsGridData data_;
    Values value_storage_;
    Data value_ref_;

    void build_impl(Real2 bounds, size_type count, XsFunc calc_xs, BC bc);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
