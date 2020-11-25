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
#include "MaterialStatePointers.hh"

namespace celeritas
{
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Manage on-device VecGeom states.
 *
 * \note Construction on a build without a GPU (or CUDA) will raise an error.
 */
class MaterialStateStore
{
  public:
    // Construct from material params and number of track states
    MaterialStateStore(const MaterialParams& mats, size_type size);

    /// ACCESSORS ///

    // Number of states
    size_type size() const { return states_.size(); }

    // View on-device states
    MaterialStatePointers device_pointers();

  private:
    size_type                        max_el_;
    DeviceVector<MaterialTrackState> states_;
    DeviceVector<real_type>          element_scratch_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
