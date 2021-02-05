//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "base/Types.hh"
#include "MaterialInterface.hh"

namespace celeritas
{
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Manage on-device material states for tracks.
 *
 * Each track has:
 * - The material ID of the volume in which it's located
 * - Scratch space for storing cross sections (one per element) for sampling
 *
 * \note Construction on a build without a GPU (or CUDA) will raise an error.
 *
 * \todo We don't need a whole class for this anymore. Delete and let the
 * mega-track-state or app state handle it.
 */
class MaterialStateStore
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t = MaterialTrackState;
    using MaterialStatePointers
        = MaterialStateData<Ownership::reference, MemSpace::device>;
    //!@}

  public:
    // Construct from material params and number of track states
    MaterialStateStore(const MaterialParams& mats, size_type size);

    //// ACCESSORS ////

    //! Number of states
    size_type size() const { return device_.size(); }

    //! Get a reference to on-device states
    const MaterialStatePointers& device_pointers() const
    {
        return device_ref_;
    };

  private:
    MaterialStateData<Ownership::value, MemSpace::device> device_;
    MaterialStatePointers                                 device_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
