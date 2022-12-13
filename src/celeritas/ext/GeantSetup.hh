//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
    //! Type aliases
    using Options = GeantPhysicsOptions;
    //!@}

  public:
    // Construct from a GDML file and physics options
    GeantSetup(const std::string& gdml_filename, Options options);

    // Default constructor
    GeantSetup() = default;

    // Terminate run on destruction
    ~GeantSetup();

    //!@{
    //! Default move assignment/construction
    GeantSetup(GeantSetup&&) = default;
    GeantSetup& operator=(GeantSetup&&) = default;
    //!@}

    // Get the world detector volume
    inline const G4VPhysicalVolume* world() const;

    //! True if we own a run manager
    explicit operator bool() const { return static_cast<bool>(run_manager_); }

  private:
    struct RMDeleter
    {
        void operator()(G4RunManager*) const;
    };
    using RMUniquePtr = std::unique_ptr<G4RunManager, RMDeleter>;

    RMUniquePtr              run_manager_{nullptr};
    const G4VPhysicalVolume* world_{nullptr};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the world detector volume.
 */
const G4VPhysicalVolume* GeantSetup::world() const
{
    CELER_EXPECT(*this);
    return world_;
}

#if !CELERITAS_USE_GEANT4
inline GeantSetup::GeantSetup(const std::string&, Options)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantSetup::~GeantSetup() = default;

inline void GeantSetup::RMDeleter::operator()(G4RunManager*) const
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
