//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

#include "RelativisticBremData.hh"
#include "SeltzerBergerData.hh"

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
    // Hack for having an "ids" field: same as model in rb_data
    struct
    {
        ActionId action;
    } ids;

    // Differential cross section data for SeltzerBerger
    SeltzerBergerTableData<W, M> sb_differential_xs;

    // Device data for RelativisticBrem
    RelativisticBremData<W, M> rb_data;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ids.action && sb_differential_xs && rb_data;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CombinedBremData& operator=(const CombinedBremData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        ids.action         = other.ids.action;
        sb_differential_xs = other.sb_differential_xs;
        rb_data            = other.rb_data;
        return *this;
    }
};

using CombinedBremDeviceRef
    = CombinedBremData<Ownership::const_reference, MemSpace::device>;
using CombinedBremHostRef
    = CombinedBremData<Ownership::const_reference, MemSpace::host>;
using CombinedBremRef
    = CombinedBremData<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
