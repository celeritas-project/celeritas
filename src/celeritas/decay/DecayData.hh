//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/data/DecayData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data for a decay interactor.
 *
 * This stores the particle IDs of the daughters for a specific decay channel.
 */
struct DecayChannelData
{
    //! Daughter particle IDs
    ItemRange<ParticleId> daughters;

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !daughters.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Decay channels for a particle type.
 */
struct DecayTableData
{
    //! Decay channels
    ItemRange<DecayChannelId> channel_ids;
    //! Branching ratio of each decay channel
    ItemRange<real_type> branching_ratios;

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !channel_ids.empty()
               && branching_ratios.size() == channel_ids.size();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
