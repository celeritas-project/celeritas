//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LevelStateAccessor.hh
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
    inline CELER_FUNCTION LevelStateAccessor(StateRef const* states,
                                             TrackSlotId tid,
                                             LevelId level_id);

    // Copy data from another LSA
    inline CELER_FUNCTION LevelStateAccessor&
    operator=(LevelStateAccessor const& other);

    //// ACCESSORS ////

    CELER_FUNCTION LocalVolumeId& vol()
    {
        return states_->vol[OpaqueId<LocalVolumeId>{index_}];
    }

    CELER_FUNCTION Real3& pos()
    {
        return states_->pos[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION Real3& dir()
    {
        return states_->dir[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION UniverseId& universe()
    {
        return states_->universe[OpaqueId<UniverseId>{index_}];
    }

    //// CONST ACCESSORS ////

    CELER_FUNCTION LocalVolumeId const& vol() const
    {
        return states_->vol[OpaqueId<LocalVolumeId>{index_}];
    }

    CELER_FUNCTION Real3 const& pos() const
    {
        return states_->pos[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION Real3 const& dir() const
    {
        return states_->dir[OpaqueId<Real3>{index_}];
    }

    CELER_FUNCTION UniverseId const& universe() const
    {
        return states_->universe[OpaqueId<UniverseId>{index_}];
    }

  private:
    StateRef const* const states_;
    size_type const index_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from states and indices
 */
CELER_FUNCTION
LevelStateAccessor::LevelStateAccessor(StateRef const* states,
                                       TrackSlotId tid,
                                       LevelId level_id)
    : states_(states), index_(tid.get() * states_->max_depth + level_id.get())
{
    CELER_EXPECT(level_id < states->max_depth);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from another LSA
 */
CELER_FUNCTION LevelStateAccessor&
LevelStateAccessor::operator=(LevelStateAccessor const& other)
{
    this->vol() = other.vol();
    this->pos() = other.pos();
    this->dir() = other.dir();
    this->universe() = other.universe();

    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
