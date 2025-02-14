//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantImportVolumeResult.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "geocel/GeoParamsInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test importing volume names for consistency.
 */
struct GeantImportVolumeResult
{
    static constexpr int empty = -1;
    static constexpr int missing = -2;

    static GeantImportVolumeResult from_import(GeoParamsInterface const& geom,
                                               G4VPhysicalVolume const* world);

    static GeantImportVolumeResult
    from_pointers(GeoParamsInterface const& geom,
                  G4VPhysicalVolume const* world);

    std::vector<int> volumes;  //!< Volume ID for each Geant4 instance ID
    std::vector<std::string> missing_labels;  //!< G4LV names without a match

    void print_expected() const;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
