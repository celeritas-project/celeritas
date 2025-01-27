//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImporterInterface.hh"

#include "GeantSetup.hh"

// Geant4 forward declaration
class G4VPhysicalVolume;  // IWYU pragma: keep
class G4ParticleDefinition;  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Only import a subset of available Geant4 data
struct GeantImportDataSelection
{
    //! Bit flags for selecting particles and process types
    using Flags = unsigned int;
    enum : unsigned int
    {
        none = 0x0,
        dummy = 0x1,  //!< Dummy particles+processes
        em_basic = 0x2,  //!< Electron, positron, gamma
        em_ex = 0x4,  //!< Extended EM particles
        optical = 0x8,  //!< Optical particles and processes
        em = em_basic | em_ex,  //!< Any EM
        hadron = 0x10,  //!< Hadronic particles and processes
    };

    Flags particles = em | optical;
    bool materials = true;
    Flags processes = em | optical;

    //! Change volume names to match exported GDML file
    bool unique_volumes = false;

    // TODO expand/set reader flags automatically based on loaded processes
    bool reader_data = true;
};

//---------------------------------------------------------------------------//
/*!
 * Load problem data directly from Geant4.
 *
 * This can be used to circumvent ROOT as a serialization tool, whether to
 * simplify the toolchain or to integrate better with user frameworks. As much
 * data as possible is imported (subject to the data selection); downstream
 * Celeritas classes will validate the imported data as needed.
 *
 * \code
    GeantImporter import(GeantSetup("blah.gdml"));
    ImportData data = import();
   \endcode
 * or to import from an existing, initialized Geant4 state:
 * \code
 *  GeantImport import(world_volume);
    ImportData data = import();
 *  \endcode
 */
class GeantImporter final : public ImporterInterface
{
  public:
    //!@{
    //! \name Type aliases
    using DataSelection = GeantImportDataSelection;
    //!@}

  public:
    // Get the top-level geometry element from the run manager+navigator
    static G4VPhysicalVolume const* get_world_volume();

    // Construct from an existing Geant4 geometry, assuming physics is loaded
    explicit GeantImporter(G4VPhysicalVolume const* world);

    // Construct by capturing a GeantSetup object
    explicit GeantImporter(GeantSetup&& setup);

    // Fill data from Geant4
    ImportData operator()(DataSelection const& selection);

    //! Fill all available data from Geant4
    ImportData operator()() final { return (*this)(DataSelection{}); }

  private:
    // Optional setup if celeritas handles initialization
    GeantSetup setup_;
    // World physical volume
    G4VPhysicalVolume const* world_{nullptr};
};

//---------------------------------------------------------------------------//

std::vector<ImportVolume>
import_volumes(G4VPhysicalVolume const& world, bool unique_volumes);

ImportParticle import_particle(G4ParticleDefinition const& p);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
inline constexpr bool operator==(GeantImporter::DataSelection const& lhs,
                                 GeantImporter::DataSelection const& rhs)
{
    // clang-format off
    return    lhs.particles == rhs.particles
           && lhs.materials == rhs.materials
           && lhs.processes == rhs.processes
           && lhs.unique_volumes == rhs.unique_volumes
           && lhs.reader_data == rhs.reader_data;
    // clang-format on
}

inline bool operator!=(GeantImporter::DataSelection const& lhs,
                       GeantImporter::DataSelection const& rhs)
{
    return !(lhs == rhs);
}

#if !CELERITAS_USE_GEANT4
inline G4VPhysicalVolume const* GeantImporter::get_world_volume()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantImporter::GeantImporter(G4VPhysicalVolume const*)
{
    CELER_DISCARD(world_);
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantImporter::GeantImporter(GeantSetup&&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline ImportData GeantImporter::operator()(DataSelection const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
