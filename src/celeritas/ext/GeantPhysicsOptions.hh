//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
 * These options attempt to default to our closest match to \c
 * G4StandardEmPhysics.
 */
struct GeantPhysicsOptions
{
    using MevEnergy = Quantity<units::Mev, double>;

    //!@{
    //! \name Gamma physics
    //! Enable discrete Coulomb
    bool coulomb_scattering{false};
    //! Enable Compton scattering
    bool compton_scattering{true};
    //! Enable the photoelectric effect
    bool photoelectric{true};
    //! Enable Rayleigh scattering
    bool rayleigh_scattering{true};
    //! Enable electron pair production
    bool gamma_conversion{true};
    //! Use G4GammaGeneral instead of individual gamma processes
    bool gamma_general{false};
    //!@}

    //!@{
    //! \name Electron and positron physics
    //! Enable e- and e+ ionization
    bool ionization{true};
    //! Enable positron annihilation
    bool annihilation{true};
    //! Enable bremsstrahlung and select a model
    BremsModelSelection brems{BremsModelSelection::all};
    //! Enable multiple coulomb scattering and select a model
    MscModelSelection msc{MscModelSelection::urban_extended};
    //! Enable atomic relaxation and select a model
    RelaxationSelection relaxation{RelaxationSelection::none};
    //!@}

    //!@{
    //! \name Physics options
    //! Number of log-spaced bins per factor of 10 in energy
    int em_bins_per_decade{7};
    //! Enable universal energy fluctuations
    bool eloss_fluctuation{true};
    //! Apply relativistic corrections for select models
    bool lpm{true};
    //! See \c PhysicsParamsOptions::disable_integral_xs
    bool integral_approach{true};
    //!@}

    //!@{
    //! \name Cutoff options
    //! Lowest energy of any EM physics process
    MevEnergy min_energy{0.1 * 1e-3};  // 0.1 keV
    //! Highest energy of any EM physics process
    MevEnergy max_energy{100 * 1e6};  // 100 TeV
    //! See \c PhysicsParamsOptions::linear_loss_limit
    double linear_loss_limit{0.01};
    //! Tracking cutoff kinetic energy for e-/e+
    MevEnergy lowest_electron_energy{0.001};  // 1 keV
    //! Kill secondaries below the production cut
    bool apply_cuts{false};
    //! Set the default production cut for all particle types [len]
    double default_cutoff{0.1 * units::centimeter};
    //!@}

    //!@{
    //! \name Multiple scattering configuration
    //! E-/e+ range factor for MSC models
    double msc_range_factor{0.04};
    //! Safety factor for MSC models
    double msc_safety_factor{0.6};
    //! Lambda limit for MSC models [len]
    double msc_lambda_limit{0.1 * units::centimeter};
    //! Step limit algorithm for MSC models
    MscStepLimitAlgorithm msc_step_algorithm{MscStepLimitAlgorithm::safety};
    //!@}

    //! Print detailed Geant4 output
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
           && a.brems == b.brems
           && a.msc == b.msc
           && a.relaxation == b.relaxation
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
           && a.msc_step_algorithm == b.msc_step_algorithm
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
