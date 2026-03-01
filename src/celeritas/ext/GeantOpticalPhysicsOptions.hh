//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantOpticalPhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <utility>

#include "celeritas/optical/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Cherenkov process options (use \c std::nullopt to disable)
struct CherenkovPhysicsOptions
{
    //! Enable generation of Cherenkov photons
    bool stack_photons{true};
    //! Track generated photons before parent
    bool track_secondaries_first{true};
    //! Maximum number of photons that can be generated before limiting step
    int max_photons{100};
    //! Maximum percentage change in particle \f$\beta\f$  before limiting step
    double max_beta_change{10.0};
};

//! Equality operator, mainly for test harness
constexpr bool
operator==(CherenkovPhysicsOptions const& a, CherenkovPhysicsOptions const& b)
{
    // clang-format off
    return a.stack_photons == b.stack_photons
           && a.track_secondaries_first == b.track_secondaries_first
           && a.max_photons == b.max_photons
           && a.max_beta_change == b.max_beta_change;
    // clang-format on
}

//---------------------------------------------------------------------------//
//! Scintillation process options (use \c std::nullopt to disable)
struct ScintillationPhysicsOptions
{
    //! Enable generation of scintillation photons
    bool stack_photons{true};
    //! Track generated photons before parent
    bool track_secondaries_first{true};
    //! Use per-particle yield and time constants for photon generation
    bool by_particle_type{false};
    //! Use material properties for sampling photon generation time
    bool finite_rise_time{false};
    //! Attach scintillation interaction information to generated photon
    bool track_info{false};
};

//! Equality operator, mainly for test harness
constexpr bool operator==(ScintillationPhysicsOptions const& a,
                          ScintillationPhysicsOptions const& b)
{
    // clang-format off
    return a.stack_photons == b.stack_photons
           && a.track_secondaries_first == b.track_secondaries_first
           && a.by_particle_type == b.by_particle_type
           && a.finite_rise_time == b.finite_rise_time
           && a.track_info == b.track_info;
    // clang-format on
}

//---------------------------------------------------------------------------//
//! Optical wavelength shifting process options (use \c std::nullopt to
//! disable)
struct WavelengthShiftingOptions
{
    //! Select a model for sampling reemission time
    optical::WlsDistribution time_profile{optical::WlsDistribution::delta};
};

//! Equality operator, mainly for test harness
constexpr bool operator==(WavelengthShiftingOptions const& a,
                          WavelengthShiftingOptions const& b)
{
    return a.time_profile == b.time_profile;
}

//---------------------------------------------------------------------------//
//! Optical boundary process options (use \c std::nullopt to disable)
struct BoundaryPhysicsOptions
{
    //! Invoke Geant4 SD at post step point if photon deposits energy
    bool invoke_sd{false};
};

//! Equality operator, mainly for test harness
constexpr bool
operator==(BoundaryPhysicsOptions const& a, BoundaryPhysicsOptions const& b)
{
    return a.invoke_sd == b.invoke_sd;
}

//---------------------------------------------------------------------------//
/*!
 * Construction options for Geant optical physics.
 *
 * These options attempt to default to our closest match to \c
 * G4OpticalPhysics from Geant4 10.5 onwards. Sub-struct options use
 * \c std::nullopt to disable the corresponding process.
 *
 * \todo When we require C++20, use `friend bool operator==(...) = default;`
 * instead of manually writing the equality operators
 */
struct GeantOpticalPhysicsOptions
{
    //!@{
    //! \name Optical photon creation physics

    //! Cherenkov radiation options (null: disabled)
    std::optional<CherenkovPhysicsOptions> cherenkov{std::in_place};
    //! Scintillation options (null: disabled)
    std::optional<ScintillationPhysicsOptions> scintillation{std::in_place};
    //!@}

    //!@{
    //! \name Optical photon physics

    //! Wavelength shifting options (null: disabled)
    std::optional<WavelengthShiftingOptions> wavelength_shifting{std::in_place};
    //! Second wavelength shifting options (null: disabled)
    std::optional<WavelengthShiftingOptions> wavelength_shifting2{
        std::in_place};
    //! Boundary process options (null: disabled)
    std::optional<BoundaryPhysicsOptions> boundary{std::in_place};
    //! Enable absorption
    bool absorption{true};
    //! Enable Rayleigh scattering
    bool rayleigh_scattering{true};
    //! Enable Mie scattering
    bool mie_scattering{true};
    //!@}

    //! Print detailed Geant4 output
    bool verbose{false};
};

//! Equality operator, mainly for test harness
constexpr bool operator==(GeantOpticalPhysicsOptions const& a,
                          GeantOpticalPhysicsOptions const& b)
{
    // clang-format off
    return a.cherenkov == b.cherenkov
           && a.scintillation == b.scintillation
           && a.wavelength_shifting == b.wavelength_shifting
           && a.wavelength_shifting2 == b.wavelength_shifting2
           && a.boundary == b.boundary
           && a.absorption == b.absorption
           && a.rayleigh_scattering == b.rayleigh_scattering
           && a.mie_scattering == b.mie_scattering
           && a.verbose == b.verbose;
    // clang-format on
}

//! Inequality operator
constexpr bool operator!=(GeantOpticalPhysicsOptions const& a,
                          GeantOpticalPhysicsOptions const& b)
{
    return !(a == b);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
