//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGdmlLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load a GDML file into memory.
 *
 * The \c detectors option reads \c auxiliary tags in the \c structure that
 * have \c auxtype=SensDet and returns a multimap of strings to volume
 * pointers.
 */
class GeantGdmlLoader
{
  public:
    //!@{
    //! \name Type aliases
    using MapDetectors = std::multimap<std::string, G4LogicalVolume*>;
    //!@}

    //! How to handle pointers in volume names
    enum class PointerTreatment
    {
        ignore,  //!< Pointers will remain in the volume name
        amputate,  //!< All text after '0x' is removed
        excise,  //!< Only pointers are carefully removed
    };

    struct Options
    {
        //! Strip pointer extensions from solids/volumes
        PointerTreatment pointers{PointerTreatment::amputate};
        //! Load sensitive detector map
        bool detectors{false};
    };

    struct Result
    {
        //! Self-owning pointer to the loaded top-level volume
        G4VPhysicalVolume* world{nullptr};
        //! If requested, load a sensitive detector map
        MapDetectors detectors;
    };

  public:
    //! Construct with options
    explicit GeantGdmlLoader(Options const& opts) : opts_{opts} {}

    //! Construct with defaults
    GeantGdmlLoader() : GeantGdmlLoader{Options{}} {}

    // Load a GDML file
    Result operator()(std::string const& filename) const;

  private:
    Options opts_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4

auto GeantGdmlLoader::operator()(std::string const& filename) const -> Result
{
    CELER_DISCARD(opts_);
    CELER_NOT_CONFIGURED("Geant4");
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
