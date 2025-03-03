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
 * The pointer treatment gives three options:
 * - \c ignore leaves names as they are imported by Geant4's GDML reader, which
 *   strips them from material/region names but leaves solid/logical/physical
 *   pointers in place.
 * - \c truncate lets the Geant4 GDML remove the pointers, which cuts
 *   everything after \c 0x including suffixes like \c _refl added during
 *   volume construction.
 * - \c remove uses a regular expression to remove pointers from volume names.
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
        truncate,  //!< All text after '0x' is removed
        remove,  //!< Only pointers are carefully removed
    };

    struct Options
    {
        //! Strip pointer extensions from solids/volumes
        PointerTreatment pointers{PointerTreatment::remove};
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
// Load a Geant4 geometry, excising pointers
G4VPhysicalVolume* load_gdml(std::string const& filename);

// Save a Geant4 geometry to a file
void save_gdml(G4VPhysicalVolume const* world, std::string const& out_filename);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Load a Geant4 geometry, excising pointers.
 *
 * This provides a good default for using GDML in Celeritas.
 *
 * \return Geant4-owned world volume
 */
inline G4VPhysicalVolume* load_gdml(std::string const& filename)
{
    return GeantGdmlLoader()(filename).world;
}

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4

inline auto GeantGdmlLoader::operator()(std::string const&) const -> Result
{
    CELER_DISCARD(opts_);
    CELER_NOT_CONFIGURED("Geant4");
}

inline void save_gdml(G4VPhysicalVolume const*, std::string const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
