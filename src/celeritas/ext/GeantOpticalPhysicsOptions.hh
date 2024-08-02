//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantOpticalPhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! WLS time model selection
enum class WLSTimeProfileSelection
{
    none,
    delta,  //!< Delta function
    exponential,  //!< Exponential decay
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Construction options for Geant optical physics.
 *
 * These options attempt to default to our closest match to \c
 * G4OpticalPhysics from Geant4 10.5 onwards.
 */
struct GeantOpticalPhysicsOptions
{
    //!@{
    //! \name Optical photon physics
    //!@}

    //!@{
    //! \name Optical photon creation physics
    //! Enable Cerenkov radiation
    bool cerenkov_radiation{true};
    //! Enable scintillation
    bool scintillation{true};
    //!@}

    //!@{
    //! \name Optical photon physics
    //! Enable wavelength shifting and select a time profile
    WLSTimeProfileSelection wavelength_shifting{WLSTimeProfileSelection::delta};
    //! Enable second wavelength shifting type and select a time profile (TODO:
    //! clarify)
    WLSTimeProfileSelection wavelength_shifting2{
        WLSTimeProfileSelection::delta};
    //! Enable boundary effects
    bool boundary{true};
    //! Enable absorption
    bool absorption{true};
    //! Enable Rayleigh scattering
    bool rayleigh_scattering{true};
    //! Enable Mie scattering
    bool mie_scattering{true};
    //!@}

    //!@{
    //! \name Cerenkov physics options
    //! Enable generation of Cerenkov photons
    bool cerenkov_stack_photons{true};
    //! Track generated photons before parent
    bool cerenkov_track_secondaries_first{true};
    //! Maximum number of photons that can be generated before limiting step
    int cerenkov_max_photons{100};
    //! Maximum percentage change in particle \beta before limiting step
    double cerenkov_max_beta_change{10.0};
    //!@}

    //!@{
    //! \name Scintillation physics options
    //! Enable generation of scintillation photons
    bool scint_stack_photons{true};
    //! Track generated photons before parent
    bool scint_track_secondaries_first{true};
    //! Use per-particle yield and time constants for photon generation
    bool scint_by_particle_type{false};
    //! Use material properties for sampling photon generation time
    bool scint_finite_rise_time{false};
    //! Attach scintillation interaction information to generated photon
    bool scint_track_info{false};
    //!@}

    //!@{
    //! \name Boundary physics options
    //! Invoke Geant4 SD at post step point if photon deposits energy
    bool invoke_sd{false};
    //!@}

    //! Print detailed Geant4 output
    bool verbose{false};
};

//! Equality operator, mainly for test harness
// TODO: when we require C++20, use `friend bool operator==(...) =
// default;`
constexpr bool operator==(GeantOpticalPhysicsOptions const& a,
                          GeantOpticalPhysicsOptions const& b)
{
    // clang-format off
    return a.cerenkov_radiation == b.cerenkov_radiation
           && a.scintillation == b.scintillation
           && a.wavelength_shifting == b.wavelength_shifting 
           && a.wavelength_shifting2 == b.wavelength_shifting2 
           && a.boundary == b.boundary 
           && a.absorption == b.absorption 
           && a.rayleigh_scattering == b.rayleigh_scattering 
           && a.mie_scattering == b.mie_scattering 
           && a.cerenkov_stack_photons == b.cerenkov_stack_photons 
           && a.cerenkov_track_secondaries_first == b.cerenkov_track_secondaries_first 
           && a.cerenkov_max_photons == b.cerenkov_max_photons 
           && a.cerenkov_max_beta_change == b.cerenkov_max_beta_change
           && a.scint_stack_photons == b.scint_stack_photons
           && a.scint_track_secondaries_first == b.scint_track_secondaries_first 
           && a.scint_by_particle_type == b.scint_by_particle_type
           && a.scint_finite_rise_time == b.scint_finite_rise_time
           && a.scint_track_info == b.scint_track_info
           && a.invoke_sd == b.invoke_sd
           && a.verbose == b.verbose;
    // clang-format on
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(WLSTimeProfileSelection value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
