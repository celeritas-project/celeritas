//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/MiniStepTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "GlobalTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Run one or more tracks with the same starting conditions for a single step.
 *
 * This high-level test *only* executes on the host so we can extract detailed
 * information from the states.
 */
class MiniStepTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using MevEnergy   = units::MevEnergy;
    using CoreRefHost = CoreRef<MemSpace::host>;
    //!@}

    struct Input
    {
        ParticleId particle_id;
        MevEnergy  energy{0};
        Real3      position{0, 0, 0};
        Real3      direction{0, 0, 1};
        real_type  time{0};
        real_type  phys_mfp{1}; //!< Number of MFP to collision

        explicit operator bool() const
        {
            return particle_id && energy >= zero_quantity() && time >= 0
                   && phys_mfp > 0;
        }
    };

    void init(const Input& inp, size_type num_tracks);

    const CoreRefHost& core_ref() const
    {
        CELER_EXPECT(states_);
        CELER_ENSURE(core_ref_);
        return core_ref_;
    }

  private:
    CoreRef<MemSpace::host>                             core_ref_;
    CollectionStateStore<CoreStateData, MemSpace::host> states_;
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
