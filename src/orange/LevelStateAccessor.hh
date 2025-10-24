//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/LevelStateAccessor.hh
// NOTE: this file is used by SCALE ORANGE; leave it public
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "OrangeData.hh"
#include "OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access the 2D fields (i.e., {track slot, udepth}) of OrangeStateData.
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
    inline CELER_FUNCTION LevelStateAccessor(OrangeParamsScalars const& scalars,
                                             StateRef const* states,
                                             TrackSlotId tid,
                                             UnivLevelId ulev_id);

    LevelStateAccessor(LevelStateAccessor const&) = default;
    LevelStateAccessor(LevelStateAccessor&&) = default;
    // Copy data from another LSA
    inline CELER_FUNCTION LevelStateAccessor&
    operator=(LevelStateAccessor const& other);
    ~LevelStateAccessor() = default;

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

    CELER_FUNCTION UnivId& universe()
    {
        return states_->universe[OpaqueId<UnivId>{index_}];
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

    CELER_FUNCTION UnivId const& universe() const
    {
        return states_->universe[OpaqueId<UnivId>{index_}];
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
LevelStateAccessor::LevelStateAccessor(OrangeParamsScalars const& scalars,
                                       StateRef const* states,
                                       TrackSlotId tid,
                                       UnivLevelId ulev_id)
    : states_(states)
    , index_(tid.get() * scalars.num_univ_levels + ulev_id.get())
{
    CELER_EXPECT(ulev_id < scalars.num_univ_levels);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from another LSA
 */
CELER_FUNCTION LevelStateAccessor&
LevelStateAccessor::operator=(LevelStateAccessor const& other)
{
    if (this == &other)
    {
        return *this;
    }
    this->vol() = other.vol();
    this->pos() = other.pos();
    this->dir() = other.dir();
    this->universe() = other.universe();

    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
