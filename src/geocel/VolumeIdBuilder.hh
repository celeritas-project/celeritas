//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeIdBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "geocel/Types.hh"

class G4LogicalVolume;

namespace celeritas
{
class GeantGeoParams;
class VolumeParams;
struct Label;

//---------------------------------------------------------------------------//
/*!
 * Map a string or Geant4 volume pointer to a volume ID.
 *
 * This \c std::visit -compatible class will convert input types to a canonical
 * volume ID depending on what metadata (constructed Geant4 geometry, volume
 * parameters) are available. A "null" ID can be returned (and warning/error
 * message emitted) if the mapping fails.
 *
 * This helper class should only have *temporary* scope.
 */
class VolumeIdBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using G4LV = G4LogicalVolume;
    //!@}

  public:
    // Construct using "global" values (NOT PREFERRED)
    VolumeIdBuilder();

    // Construct using geant4 params and/or volume params
    inline VolumeIdBuilder(VolumeParams const*, GeantGeoParams const*);

    // Map from a string using VolumeParams
    VolumeId operator()(std::string const& s) const;

    // Map from a label using VolumeParams
    VolumeId operator()(Label const& lab) const;

    // Map from Geant4 volume pointer using geant_geo
    VolumeId operator()(G4LogicalVolume const* lv) const;

  private:
    VolumeParams const* volumes_{nullptr};
    GeantGeoParams const* geant_geo_{nullptr};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct using geant4 params and/or volume params.
 *
 * If both are null, this class will be nonfunctional.
 */
VolumeIdBuilder::VolumeIdBuilder(VolumeParams const* v,
                                 GeantGeoParams const* gg)
    : volumes_{v}, geant_geo_{gg}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
