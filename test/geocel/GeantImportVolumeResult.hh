//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantImportVolumeResult.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "geocel/GeoParamsInterface.hh"

namespace celeritas
{
namespace test
{
class AssertionHelper;
//---------------------------------------------------------------------------//
/*!
 * Test importing volume names for consistency.
 */
struct GeantImportVolumeResult
{
    static constexpr char class_str[] = "GeantImportVolumeResult";

    static constexpr int empty = -1;
    static constexpr int missing = -2;

    static GeantImportVolumeResult from_import(GeoParamsInterface const& geom);

    std::vector<int> volumes;  //!< Volume ID for each Geant4 instance ID
    std::vector<std::string> missing_labels;  //!< G4LV names without a match

    friend std::ostream& operator<<(std::ostream&,
                                    GeantImportVolumeResult const&);
    friend bool is_ref_eq(AssertionHelper&,
                          GeantImportVolumeResult const&,
                          GeantImportVolumeResult const&);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
