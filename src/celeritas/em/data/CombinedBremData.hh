//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/CombinedBremData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

#include "RelativisticBremData.hh"
#include "SeltzerBergerData.hh"

namespace celeritas
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

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return sb_differential_xs && rb_data;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CombinedBremData& operator=(CombinedBremData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        sb_differential_xs = other.sb_differential_xs;
        rb_data = other.rb_data;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
