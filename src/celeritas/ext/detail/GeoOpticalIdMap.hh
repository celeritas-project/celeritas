//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeoOpticalIdMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <G4MaterialTable.hh>

#include "geocel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct optical material IDs and map from a geometry material ID.
 *
 * This construct a material -> optical material mapping based on whether the
 * \c RINDEX table is present on a Geant4 material.
 *
 * As a reminder, \em geometry materials correspond to \c G4Material and
 * \em physics materials correspond to \c G4MaterialCutsCouple .
 */
class GeoOpticalIdMap
{
  public:
    //! Construct without optical materials
    GeoOpticalIdMap() = default;

    // Construct from underlying Geant4 objects
    explicit GeoOpticalIdMap(G4MaterialTable const&);

    // Return the optical ID corresponding to a geo ID
    inline OpticalMaterialId operator[](GeoMaterialId) const;

    //! True if no optical materials are present
    bool empty() const { return geo_to_opt_.empty(); }

    //! Number of geometry materials
    GeoMaterialId::size_type num_geo() const { return geo_to_opt_.size(); }

    //! Number of optical materials
    OpticalMaterialId::size_type num_optical() const { return num_optical_; }

  private:
    std::vector<OpticalMaterialId> geo_to_opt_;
    OpticalMaterialId::size_type num_optical_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return the optical ID corresponding to a geo ID.
 *
 * The result \em may be a "null" ID if there's no associated optical physics.
 */
OpticalMaterialId GeoOpticalIdMap::operator[](GeoMaterialId m) const
{
    CELER_EXPECT(!this->empty());
    CELER_EXPECT(m < this->num_geo());

    return geo_to_opt_[m.get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
