//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

#include "GeantPhysicsOptions.hh"

// Geant4 forward declarations
class G4VPhysicalVolume;
class G4RunManager;

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Construct a Geant 4 run manager and populate internal Geant4 physics.
 *
 * This is usually passed directly into \c GeantImporter . It hides Geant4
 * implementation details (including header files) from the rest of the code.
 * It is safe to include even when Geant4 is unavailable!
 *
 * The setup is targeted specifically for physics that Celeritas supports.
 */
class GeantSetup
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Clear Geant4's signal handlers that get installed when linking 11+
    static void disable_signal_handler();

    // Construct from a GDML file and physics options
    GeantSetup(std::string const& gdml_filename, Options options);

    // Default constructor
    GeantSetup() = default;

    // Terminate run on destruction
    ~GeantSetup();

    //!@{
    //! Prevent copying but allow moving
    GeantSetup(GeantSetup const&) = delete;
    GeantSetup& operator=(GeantSetup const&) = delete;
    GeantSetup(GeantSetup&&) = default;
    GeantSetup& operator=(GeantSetup&&) = default;
    //!@}

    // Get the world detector volume
    inline G4VPhysicalVolume const* world() const;

    //! True if we own a run manager
    explicit operator bool() const { return static_cast<bool>(run_manager_); }

  private:
    struct RMDeleter
    {
        void operator()(G4RunManager*) const;
    };
    using RMUniquePtr = std::unique_ptr<G4RunManager, RMDeleter>;

    RMUniquePtr run_manager_{nullptr};
    G4VPhysicalVolume* world_{nullptr};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Get the number of threads in a version-portable way
int get_num_threads(G4RunManager const&);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the world detector volume.
 */
G4VPhysicalVolume const* GeantSetup::world() const
{
    CELER_EXPECT(*this);
    return world_;
}

#if !CELERITAS_USE_GEANT4
inline GeantSetup::GeantSetup(std::string const&, Options)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantSetup::~GeantSetup() = default;

inline void GeantSetup::RMDeleter::operator()(G4RunManager*) const
{
    CELER_ASSERT_UNREACHABLE();
}

inline int get_num_threads(G4RunManager const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
