//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/GeantSimpleCaloStorage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

class G4LogicalVolume;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Shared storage for a single GeantSimpleCalo.
 *
 * This is created by the \c GeantSimpleCalo and passed to all the \c
 * G4VSensitiveDetector instances (one for each thread) for the calo.
 */
struct GeantSimpleCaloStorage
{
    using VecReal = std::vector<double>;
    using VecVecReal = std::vector<VecReal>;
    using MapVolumeIdx = std::unordered_map<G4LogicalVolume*, size_type>;

    //! SD name
    std::string name;
    //! Number of threads
    size_type num_threads{};
    //! Map of logical volume to "detector ID" index
    MapVolumeIdx volume_to_index;
    //! Accumulated energy deposition [thread][volume]
    VecVecReal data;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
