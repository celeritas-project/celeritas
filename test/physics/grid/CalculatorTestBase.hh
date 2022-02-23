//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CalculatorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "physics/grid/XsGridData.hh"

#include "gtest/Test.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for interpolating values on grids.
 */
class CalculatorTestBase : public celeritas::Test
{
  public:
    //!@{
    //! Type aliases
    using real_type  = celeritas::real_type;
    using size_type  = celeritas::size_type;
    using XsGridData = celeritas::XsGridData;
    using Values     = celeritas::Collection<real_type,
                                         celeritas::Ownership::value,
                                         celeritas::MemSpace::host>;
    using Data       = celeritas::Collection<real_type,
                                       celeritas::Ownership::const_reference,
                                       celeritas::MemSpace::host>;
    using SpanReal   = celeritas::Span<real_type>;
    //!@}

  public:
    // Construct linear cross sections
    void     build(real_type emin, real_type emax, size_type count);
    void     set_prime_index(size_type i);
    SpanReal mutable_values();

    const XsGridData& data() const { return data_; }
    const Data&       values() const { return value_ref_; }

  private:
    XsGridData data_;
    Values     value_storage_;
    Data       value_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
