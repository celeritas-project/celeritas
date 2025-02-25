//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/SDTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "celeritas/GeantTestBase.hh"

class G4VLogicalVolume;

namespace celeritas
{
namespace test
{
class SimpleSensitiveDetector;
//---------------------------------------------------------------------------//
/*!
 * Attach "debug" sensitive detectors to a Geant4 geometry.
 */
class SDTestBase : virtual public GeantTestBase
{
    using Base = GeantTestBase;

  public:
    //!@{
    //! \name Type aliases
    using SetStr = std::set<std::string>;
    using MapStrSD = std::map<std::string, SimpleSensitiveDetector*>;
    //!@}

  public:
    //! Return a set of volume names to be turned into detectors
    virtual SetStr detector_volumes() const = 0;

    //! Access constructed sensitive detectors
    MapStrSD const& detectors() const { return detectors_; }

  protected:
    // Attach SDs when building geometry
    SPConstGeoI build_fresh_geometry(std::string_view) override;

    // Restore SD map and reset SDs when rebuilding geometry
    SPConstGeo build_geometry() override;

  private:
    // Constructed detectors
    MapStrSD detectors_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
