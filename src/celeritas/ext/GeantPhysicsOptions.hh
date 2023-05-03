//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Brems selection (TODO: make bitset)
enum class BremsModelSelection
{
    none,
    seltzer_berger,
    relativistic,
    all,
    size_
};

//---------------------------------------------------------------------------//
//! MSC selection (TODO: make bitset?)
enum class MscModelSelection
{
    none,
    urban,
    urban_extended,  //!< Use 100 TeV as upper bound instead of 100 MeV
    wentzel_vi,
    urban_wentzel,  //!< Urban for low-E, Wentzel_VI for high-E
    goudsmit_saunderson,
    size_
};

//---------------------------------------------------------------------------//
//! Atomic relaxation options
enum class RelaxationSelection
{
    none,
    radiative,
    all,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Construction options for Geant physics.
 *
 * These options attempt to default to our closest match to
 * \c G4StandardEmPhysics.
 *
 * - \c coulomb_scattering: enable a discrete Coulomb scattering process
 * - \c compton_scattering: enable a discrete Compton scattering process
 * - \c photoelectric: enable photoelectric effect
 * - \c rayleigh_scattering: enable the Rayleigh scattering process
 * - \c gamma_conversion: enable e+e- pair production process
 * - \c gamma_general: load G4GammaGeneral instead of individual processes
 * - \c ionization: enable e- and e+ ionization process
 * - \c annihilation: enable e-e+ annihilation process
 * - \c brems: enable Bremsstrahlung process selection
 * - \c brems_selection: Bremsstrahlung model selection
 * - \c msc: Multiple scattering model selection
 * - \c relaxation: Atomic relaxation selection
 * - \c em_bins_per_decade: number of log-spaced energy bins per factor of 10
 * - \c eloss_fluctuation: enable universal energy fluctuations
 * - \c lpm: apply relativistic corrections for select models
 * - \c integral_approach: see \c PhysicsParamsOptions::disable_integral_xs
 * - \c min_energy: lowest energy of any EM physics process
 * - \c max_energy: highest energy of any EM physics process
 * - \c linear_loss_limit: see \c PhysicsParamsOptions::linear_loss_limit
 * - \c lowest_electron_energy: lowest e-/e+ kinetic energy
 * - \c msc_range_factor: e-/e+ range factor for MSC models
 * - \c msc_safety_factor: safety factor for MSC models
 * - \c msc_lambda_limit: lambda limit for MSC models [cm]
 * - \c apply_cuts: kill secondaries below the production cut
 * - \c verbose: print detailed Geant4 output
 */
struct GeantPhysicsOptions
{
    //// Physics list ////
    // Gammas
    bool coulomb_scattering{false};
    bool compton_scattering{true};
    bool photoelectric{true};
    bool rayleigh_scattering{true};
    bool gamma_conversion{true};
    bool gamma_general{false};

    // Electrons and positrons
    bool ionization{true};
    bool annihilation{true};
    BremsModelSelection brems_selection{BremsModelSelection::all};
    MscModelSelection msc{MscModelSelection::urban_extended};
    RelaxationSelection relaxation{RelaxationSelection::none};

    //// Physics options ////
    int em_bins_per_decade{7};
    bool eloss_fluctuation{true};
    bool lpm{true};
    bool integral_approach{true};

    units::MevEnergy min_energy{0.1 * 1e-3};  // 0.1 keV
    units::MevEnergy max_energy{100 * 1e6};  // 100 TeV
    real_type linear_loss_limit{0.01};
    units::MevEnergy lowest_electron_energy{0.001};  // 1 keV
    bool apply_cuts{false};

    // Multiple scattering parameters
    real_type msc_range_factor{0.04};
    real_type msc_safety_factor{0.6};
    real_type msc_lambda_limit{0.1};  // 1 mm

    bool verbose{false};
};

//! Equality operator, mainly for test harness
// TODO: when we require C++20, use `friend bool operator==(...) = default;`
constexpr bool
operator==(GeantPhysicsOptions const& a, GeantPhysicsOptions const& b)
{
    // clang-format off
    return a.coulomb_scattering == b.coulomb_scattering
           && a.compton_scattering == b.compton_scattering
           && a.photoelectric == b.photoelectric
           && a.rayleigh_scattering == b.rayleigh_scattering
           && a.gamma_conversion == b.gamma_conversion
           && a.gamma_general == b.gamma_general
           && a.ionization == b.ionization
           && a.annihilation == b.annihilation
           && a.brems_selection == b.brems_selection
           && a.msc == b.msc
           && a.relaxation == b.relaxation
           // Physics options
           && a.em_bins_per_decade == b.em_bins_per_decade
           && a.eloss_fluctuation == b.eloss_fluctuation
           && a.lpm == b.lpm
           && a.integral_approach == b.integral_approach
           && a.min_energy == b.min_energy
           && a.max_energy == b.max_energy
           && a.linear_loss_limit == b.linear_loss_limit
           && a.lowest_electron_energy == b.lowest_electron_energy
           && a.apply_cuts == b.apply_cuts
           && a.msc_range_factor == b.msc_range_factor
           && a.msc_safety_factor == b.msc_safety_factor
           && a.msc_lambda_limit == b.msc_lambda_limit
           && a.verbose == b.verbose;
    // clang-format on
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(BremsModelSelection value);
char const* to_cstring(MscModelSelection value);
char const* to_cstring(RelaxationSelection value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
