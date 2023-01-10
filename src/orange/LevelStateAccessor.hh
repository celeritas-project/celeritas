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
template<Ownership W, MemSpace M>
class LevelStateAccessor
{
  public:
    // Construct from states and indices
    explicit inline CELER_FUNCTION
    LevelStateAccessor(OrangeStateData<W, M>* states,
                       ThreadId               thread_id,
                       LevelId                level_id);

    CELER_FUNCTION VolumeId volume_id()
    {
        return states_->volume_id[OpaqueId<VolumeId>{offset_}];
    }

    CELER_FUNCTION void set_volume_id(VolumeId id)
    {
        &states_->volume_id[OpaqueId<VolumeId>{offset_}] = id;
    }

  private:
    const OrangeStateData<W, M>* states_;
    const size_type              offset_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from states and indices
 */
template<Ownership W, MemSpace M>
CELER_FUNCTION
LevelStateAccessor<W, M>::LevelStateAccessor(OrangeStateData<W, M>* states,
                                             ThreadId               thread_id,
                                             LevelId                level_id)
    : states_(states)
    , offset_(thread_id.unchecked_get() * OrangeParamsScalars::max_level
              + level_id.unchecked_get())
{
}

//---------------------------------------------------------------------------//
} // namespace celeritas
