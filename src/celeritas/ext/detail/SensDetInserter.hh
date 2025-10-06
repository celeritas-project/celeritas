//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/SensDetInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

#include "geocel/Types.hh"
#include "geocel/VolumeIdBuilder.hh"

class G4LogicalVolume;
class G4VSensitiveDetector;

namespace celeritas
{
class GeantGeoParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Map logical volumes to canonical volumes, logging potential issues.
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
    using MapIdLv = std::map<VolumeId, G4LogicalVolume const*>;
    //!@}

  public:
    // Construct from interfaces, loading geant geo
    SensDetInserter(SetLV const& skip_volumes, MapIdLv* found, VecLV* missing);

    // Save a sensitive detector
    void operator()(G4LogicalVolume const* lv, G4VSensitiveDetector const* sd);

    // Forcibly add the given volume
    void operator()(G4LogicalVolume const* lv);

  private:
    VolumeIdBuilder to_vol_id_;
    SetLV const& skip_volumes_;
    MapIdLv* found_;
    VecLV* missing_;

    VolumeId insert_impl(G4LogicalVolume const* lv);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
