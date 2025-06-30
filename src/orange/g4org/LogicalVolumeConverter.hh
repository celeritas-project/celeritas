//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/LogicalVolumeConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "corecel/io/Label.hh"
#include "geocel/Types.hh"
#include "orange/orangeinp/ObjectInterface.hh"

//---------------------------------------------------------------------------//
// Forward declarations
//---------------------------------------------------------------------------//

class G4LogicalVolume;

namespace celeritas
{
class GeantGeoParams;
namespace g4org
{
//---------------------------------------------------------------------------//
struct LogicalVolume;
class SolidConverter;

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 base LV to an ORANGE temporary LV.
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
    using VecLabel = std::vector<Label>;
    using SPLV = std::shared_ptr<LogicalVolume>;
    using result_type = std::pair<SPLV, bool>;
    //!@}

  public:
    LogicalVolumeConverter(GeantGeoParams const& geo,
                           SolidConverter& convert_solid);

    // Convert a volume, return result plus insertion
    result_type operator()(arg_type);

  private:
    using WPLV = std::weak_ptr<LogicalVolume>;

    //// DATA ////

    GeantGeoParams const& geo_;
    SolidConverter& convert_solid_;
    std::unordered_map<G4LogicalVolume const*, WPLV> cache_;

    //// HELPER FUNCTIONS ////

    // Convert an LV that's not in the cache
    SPLV construct_impl(arg_type);
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
