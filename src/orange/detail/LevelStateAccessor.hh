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
    inline CELER_FUNCTION LevelStateAccessor(StateRef const* states,
                                             ThreadId thread_id,
                                             LevelId level_id);

    // Copy data from another LSA
    inline CELER_FUNCTION LevelStateAccessor&
    operator=(LevelStateAccessor const& other);

    //// ACCESSORS ////

    CELER_FUNCTION VolumeId& vol()
    {
        return states_->vol[OpaqueId<VolumeId>{index_}];
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

    CELER_FUNCTION SurfaceId& surf()
    {
        return states_->surf[OpaqueId<SurfaceId>{index_}];
    }

    CELER_FUNCTION Sense& sense()
    {
        return states_->sense[OpaqueId<Sense>{index_}];
    }

    CELER_FUNCTION BoundaryResult& boundary()
    {
        return states_->boundary[OpaqueId<BoundaryResult>{index_}];
    }

    //// CONST ACCESSORS ////

    CELER_FUNCTION VolumeId const& vol() const
    {
        return states_->vol[OpaqueId<VolumeId>{index_}];
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

    CELER_FUNCTION SurfaceId const& surf() const
    {
        return states_->surf[OpaqueId<SurfaceId>{index_}];
    }

    CELER_FUNCTION Sense const& sense() const
    {
        return states_->sense[OpaqueId<Sense>{index_}];
    }

    CELER_FUNCTION BoundaryResult const& boundary() const
    {
        return states_->boundary[OpaqueId<BoundaryResult>{index_}];
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
                                       ThreadId thread_id,
                                       LevelId level_id)
    : states_(states)
    , index_(thread_id.get() * states_->max_level + level_id.get())
{
    CELER_EXPECT(level_id < states->max_level);
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
    this->surf() = other.surf();
    this->sense() = other.sense();
    this->boundary() = other.boundary();

    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
