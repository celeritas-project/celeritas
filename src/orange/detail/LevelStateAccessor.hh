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
    explicit inline CELER_FUNCTION LevelStateAccessor(StateRef const* states,
                                                      ThreadId thread_id,
                                                      LevelId level_id);

    //// GETTERS ////

    CELER_FUNCTION VolumeId vol() const
    {
        return states_->vol[OpaqueId<VolumeId>{index_}];
    }

    CELER_FUNCTION Real3 pos() const
    {
        return states_->pos[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION Real3 dir() const
    {
        return states_->dir[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION UniverseId universe() const
    {
        return states_->universe[OpaqueId<UniverseId>{index_}];
    }

    CELER_FUNCTION SurfaceId surf() const
    {
        return states_->surf[OpaqueId<SurfaceId>{index_}];
    }

    CELER_FUNCTION Sense sense() const
    {
        return states_->sense[OpaqueId<Sense>{index_}];
    }

    CELER_FUNCTION BoundaryResult boundary() const
    {
        return states_->boundary[OpaqueId<BoundaryResult>{index_}];
    }

    //// SETTERS ////

    CELER_FUNCTION void set_vol(VolumeId id)
    {
        states_->vol[OpaqueId<VolumeId>{index_}] = id;
    }

    CELER_FUNCTION void set_pos(Real3 pos)
    {
        states_->pos[OpaqueId<Real3>{index_}] = pos;
    }

    CELER_FUNCTION void set_dir(Real3 dir)
    {
        states_->dir[OpaqueId<Real3>{index_}] = dir;
    }

    CELER_FUNCTION void set_universe(UniverseId id)
    {
        states_->universe[OpaqueId<UniverseId>{index_}] = id;
    }

    CELER_FUNCTION void set_surf(SurfaceId id)
    {
        states_->surf[OpaqueId<SurfaceId>{index_}] = id;
    }

    CELER_FUNCTION void set_sense(Sense sense)
    {
        states_->sense[OpaqueId<Sense>{index_}] = sense;
    }

    CELER_FUNCTION void set_boundary(BoundaryResult br)
    {
        states_->boundary[OpaqueId<BoundaryResult>{index_}] = br;
    }

  private:
    StateRef const* states_;
    const size_type index_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from states and indices
 */
CELER_FUNCTION
LevelStateAccessor::LevelStateAccessor(StateRef const* states,
                                       ThreadId thread_id,
                                       LevelId level_id)
    : states_(states)
    , index_(thread_id.get() * states_->max_level + level_id.get())
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
