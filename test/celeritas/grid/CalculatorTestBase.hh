//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CalculatorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
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
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using Data
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;
    using SpanReal = Span<real_type>;
    //!@}

  public:
    // Construct linear cross sections
    void build(real_type emin, real_type emax, size_type count);
    void set_prime_index(size_type i);
    SpanReal mutable_values();

    XsGridData const& data() const { return data_; }
    Data const& values() const { return value_ref_; }

  private:
    XsGridData data_;
    Values value_storage_;
    Data value_ref_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
