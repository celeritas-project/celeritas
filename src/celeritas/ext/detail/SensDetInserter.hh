//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/SensDetInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <unordered_set>
#include <vector>

#include "corecel/Assert.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeoParamsInterface.hh"

#include "../GeantVolumeMapper.hh"

class G4LogicalVolume;
class G4VSensitiveDetector;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Map logical volumes to native geometry and log potential issues.
 *
 * This is an implementation detail of \c GeantSd .
 */
class SensDetInserter
{
  public:
    //!@{
    //! \name Type aliases
    using SetLV = std::unordered_set<G4LogicalVolume const*>;
    using VecLV = std::vector<G4LogicalVolume const*>;
    using MapIdLv = std::map<ImplVolumeId, G4LogicalVolume const*>;
    //!@}

  public:
    // Construct with defaults
    inline SensDetInserter(GeoParamsInterface const& geo,
                           SetLV const& skip_volumes,
                           MapIdLv* found,
                           VecLV* missing);

    // Save a sensitive detector
    void operator()(G4LogicalVolume const* lv, G4VSensitiveDetector const* sd);

    // Forcibly add the given volume
    void operator()(G4LogicalVolume const* lv);

  private:
    GeoParamsInterface const& geo_;
    GeantVolumeMapper g4_to_celer_;
    SetLV const& skip_volumes_;
    MapIdLv* found_;
    VecLV* missing_;

    ImplVolumeId insert_impl(G4LogicalVolume const* lv);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with references to the inserted data.
 */
SensDetInserter::SensDetInserter(GeoParamsInterface const& geo,
                                 SetLV const& skip_volumes,
                                 MapIdLv* found,
                                 VecLV* missing)
    : geo_(geo)
    , g4_to_celer_(geo)
    , skip_volumes_{skip_volumes}
    , found_{found}
    , missing_{missing}
{
    CELER_EXPECT(found_);
    CELER_EXPECT(missing_);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
