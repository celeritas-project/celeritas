//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/LogicalVolumeConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "geocel/Types.hh"
#include "orange/orangeinp/ObjectInterface.hh"

//---------------------------------------------------------------------------//
// Forward declarations
//---------------------------------------------------------------------------//

class G4LogicalVolume;

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
struct LogicalVolume;
class SolidConverter;

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 base LV to a VecGeom LV.
 *
 * This does not convert or add any of the daughters, which must be placed as
 * physical volumes.
 */
class LogicalVolumeConverter
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = G4LogicalVolume const&;
    using result_type = std::shared_ptr<LogicalVolume>;
    using MapLvVolId = std::unordered_map<G4LogicalVolume const*, VolumeId>;
    //!@}

  public:
    explicit LogicalVolumeConverter(SolidConverter& convert_solid);

    // Convert a volume
    result_type operator()(arg_type);

  private:
    //// DATA ////

    SolidConverter& convert_solid_;
    std::unordered_map<G4LogicalVolume const*, result_type> cache_;

    //// HELPER FUNCTIONS ////

    // Convert an LV that's not in the cache
    result_type construct_impl(arg_type);
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
