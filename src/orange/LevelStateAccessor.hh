//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/LevelStateAccessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Accesss the 2D fields (i.e., {thread, level}) of OrangeStateData
 */
class LevelStateAccessor
{
  public:
    //!@{
    //! Type aliases
    using StateRef = NativeRef<OrangeStateData>;
    //!@}

  public:
    // Construct from states and indices
    explicit inline CELER_FUNCTION LevelStateAccessor(const StateRef* states,
                                                      ThreadId thread_id,
                                                      LevelId  level_id);

    CELER_FUNCTION VolumeId vol()
    {
        return states_->vol[OpaqueId<VolumeId>{index_}];
    }

    CELER_FUNCTION void set_vol(VolumeId id)
    {
        states_->vol[OpaqueId<VolumeId>{index_}] = id;
    }

    CELER_FUNCTION Real3 pos()
    {
        return states_->pos[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION void set_pos(Real3 pos)
    {
        states_->pos[OpaqueId<Real3>{index_}] = pos;
    }

    CELER_FUNCTION Real3 dir()
    {
        return states_->dir[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION void set_dir(Real3 dir)
    {
        states_->dir[OpaqueId<Real3>{index_}] = dir;
    }

    CELER_FUNCTION UniverseId universe()
    {
        return states_->universe[OpaqueId<UniverseId>{index_}];
    }

    CELER_FUNCTION void set_universe(UniverseId id)
    {
        states_->universe[OpaqueId<UniverseId>{index_}] = id;
    }

  private:
    const StateRef* states_;
    const size_type index_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from states and indices
 */
CELER_FUNCTION
LevelStateAccessor::LevelStateAccessor(const StateRef* states,
                                       ThreadId        thread_id,
                                       LevelId         level_id)
    : states_(states)
    , index_(thread_id.get() * OrangeParamsScalars::max_level + level_id.get())
{
}

//---------------------------------------------------------------------------//
} // namespace celeritas
