//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CalculatorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "gtest/Test.hh"
#include "base/Collection.hh"
#include "physics/grid/XsGridInterface.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    CalculatorTestBase ...;
   \endcode
 */
class CalculatorTestBase : public celeritas::Test
{
  public:
    //!@{
    //! Type aliases
    using real_type  = celeritas::real_type;
    using size_type  = celeritas::size_type;
    using XsGridData = celeritas::XsGridData;
    using Pointers
        = celeritas::Collection<real_type,
                                celeritas::Ownership::const_reference,
                                celeritas::MemSpace::host>;
    using SpanReal = celeritas::Span<real_type>;
    //!@}

  public:
    // Construct linear cross sections
    void     build(real_type emin, real_type emax, size_type count);
    void     set_prime_index(size_type i);
    SpanReal mutable_values();

    const XsGridData& data() const { return data_; }
    const Pointers&   values() const { return value_ref_; }

  private:
    XsGridData data_;
    celeritas::Collection<real_type,
                          celeritas::Ownership::value,
                          celeritas::MemSpace::host>
             value_storage_;
    Pointers value_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
