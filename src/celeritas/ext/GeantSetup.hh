//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetup.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"

#include "GeantPhysicsOptions.hh"

// Geant4 forward declarations
class G4VPhysicalVolume;
class G4RunManager;

namespace celeritas
{
class GeantGeoParams;

//---------------------------------------------------------------------------//
/*!
 * Construct a Geant 4 run manager and populate internal Geant4 physics.
 *
 * This is usually passed directly into \c GeantImporter . It hides Geant4
 * implementation details (including header files) from the rest of the code.
 * It is safe to include even when Geant4 is unavailable!
 *
 * The setup is targeted specifically for physics that Celeritas supports.
 *
 * \todo This is a hot mess; it needs to be unified with inp/setup and not
 * passed around by moving it.
 */
class GeantSetup
{
  public:
    //!@{
    //! \name Type aliases
    using Options = GeantPhysicsOptions;
    using SPGeantGeo = std::shared_ptr<GeantGeoParams>;
    //!@}

  public:
    // Construct from a GDML file and physics options
    GeantSetup(std::string const& gdml_filename, Options options);

    // Default constructor
    GeantSetup() = default;

    // Terminate run on destruction
    ~GeantSetup();

    //! Prevent copying but allow moving
    CELER_DEFAULT_MOVE_DELETE_COPY(GeantSetup);

    //! Get the constructed geometry
    SPGeantGeo const& geo_params() const { return geo_; }

    //! True if we own a run manager
    explicit operator bool() const { return static_cast<bool>(run_manager_); }

  private:
    struct RMDeleter
    {
        void operator()(G4RunManager*) const;
    };
    using RMUniquePtr = std::unique_ptr<G4RunManager, RMDeleter>;

    RMUniquePtr run_manager_{nullptr};
    SPGeantGeo geo_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
