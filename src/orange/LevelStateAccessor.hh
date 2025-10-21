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
 * Access the 2D fields (i.e., {track slot, level}) of OrangeStateData.
 */
class LevelStateAccessor
{
  public:
    //!@{
    //! Type aliases
    using StateRef = NativeRef<OrangeStateData>;
    using LVolId = LocalVolumeId;
    using UnivId = UniverseId;
    //!@}

  public:
    // Construct from states and indices
    inline CELER_FUNCTION LevelStateAccessor(StateRef const* states,
                                             TrackSlotId tid,
                                             ImplLevelId level_id);

    // Copy data from another LSA
    inline CELER_FUNCTION LevelStateAccessor&
    operator=(LevelStateAccessor const& other);

    // Default move/copy
    LevelStateAccessor(LevelStateAccessor const&) = default;
    LevelStateAccessor(LevelStateAccessor&&) = default;
    ~LevelStateAccessor() = default;

    //// ACCESSORS ////

    CELER_FIF LVolId& vol() { return this->get(s_.vol); }
    CELER_FIF Real3& pos() { return this->get(s_.pos); }
    CELER_FIF Real3& dir() { return this->get(s_.dir); }
    CELER_FIF UnivId& universe() { return this->get(s_.universe); }

    //// CONST ACCESSORS ////

    CELER_FIF LVolId const& vol() const { return this->get(s_.vol); }
    CELER_FIF Real3 const& pos() const { return this->get(s_.pos); }
    CELER_FIF Real3 const& dir() const { return this->get(s_.dir); }
    CELER_FIF UnivId const& universe() const { return this->get(s_.universe); }

  private:
    //// TYPES ////

    template<class T>
    using Items = Collection<T, Ownership::reference, MemSpace::native>;

    //// DATA ////

    StateRef const& s_;
    size_type index_;

    //// HELPER FUNCTIONS ////
    template<class T>
    CELER_FIF T& get(Items<T> const& items)
    {
        return items[OpaqueId<T>{index_}];
    }

    template<class T>
    CELER_FIF T const& get(Items<T> const& items) const
    {
        return items[OpaqueId<T>{index_}];
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from states and indices.
 */
CELER_FIF
LevelStateAccessor::LevelStateAccessor(StateRef const* states,
                                       TrackSlotId tid,
                                       ImplLevelId level_id)
    : s_(*states), index_(tid.get() * s_.max_depth + level_id.get())
{
    CELER_EXPECT(level_id < s_.max_depth);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from another LSA.
 */
CELER_FIF LevelStateAccessor&
LevelStateAccessor::operator=(LevelStateAccessor const& other)
{
    CELER_EXPECT(index_ != other.index_);

    this->vol() = other.vol();
    this->pos() = other.pos();
    this->dir() = other.dir();
    this->universe() = other.universe();

    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
