//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "celeritas_config.h"
#include "base/Assert.hh"

// Geant4 forward declarations
class G4VPhysicalVolume;
class G4RunManager;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a Geant 4 run manager and populate internal Geant4 physics.
 *
 * This is usually passed directly into \c GeantImporter . It hides geant
 * includes from the rest of the code.
 */
class GeantSetup
{
  public:
    //!@{
    //! Type aliases
    //!@}

    //! Selection of physics processes
    enum class PhysicsList
    {
        em_basic,    //!< Celeritas demo loop
        em_standard, //!< G4EmStandardPhysics
        ftfp_bert,   //!< Full physics
    };

  public:
    // Construct from a GDML file and physics options
    GeantSetup(const std::string& gdml_filename, PhysicsList physics);

    // Default constructor
    GeantSetup() = default;

    //! Get world detector volume
    const G4VPhysicalVolume* world() const
    {
        CELER_EXPECT(*this);
        return world_;
    }

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
#if !CELERITAS_USE_GEANT4
inline GeantSetup::GeantSetup(const std::string&, PhysicsList)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline void GeantSetup::RMDeleter::operator()(G4RunManager*) const
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
