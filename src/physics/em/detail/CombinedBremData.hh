//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

#include "SeltzerBergerData.hh"
#include "RelativisticBremData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for sampling CombinedBremInteractor.
 */
template<Ownership W, MemSpace M>
struct CombinedBremData
{
    // Differential cross section data for SeltzerBerger
    SeltzerBergerTableData<W, M> sb_differential_xs;

    // Device data for RelativisticBrem
    RelativisticBremData<W, M> rb_data;

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        sb_differential_xs&& rb_data;
        return true;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CombinedBremData& operator=(const CombinedBremData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        sb_differential_xs = other.sb_differential_xs;
        rb_data            = other.rb_data;
        return *this;
    }
};

using CombinedBremDeviceRef
    = CombinedBremData<Ownership::const_reference, MemSpace::device>;
using CombinedBremHostRef
    = CombinedBremData<Ownership::const_reference, MemSpace::host>;
using CombinedBremNativeRef
    = CombinedBremData<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
