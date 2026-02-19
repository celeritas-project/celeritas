//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"

#include "GeantOpticalPhysicsOptions.hh"

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
    urban,  //!< Urban for all energies
    wentzelvi,  //!< Wentzel VI for all energies
    urban_wentzelvi,  //!< Urban below 100 MeV, Wentzel VI above
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
 * Construction options for Geant muon EM physics.
 */
struct GeantMuonPhysicsOptions
{
    //! Enable muon pair production
    bool pair_production{true};
    //! Enable muon ionization
    bool ionization{true};
    //! Enable muon bremsstrahlung
    bool bremsstrahlung{true};
    //! Enable muon single Coulomb scattering
    bool coulomb{false};
    //! Enable multiple coulomb scattering and select a model.
    //! Muon MSC currently requires MSC enabled for electrons and positrons
    MscModelSelection msc{MscModelSelection::urban};

    //! True if any process is activated
    explicit operator bool() const
    {
        return pair_production || ionization || bremsstrahlung || coulomb
               || msc != MscModelSelection::none;
    }

    //! Initialize with no physics
    static GeantMuonPhysicsOptions deactivated()
    {
        GeantMuonPhysicsOptions opt;
        opt.pair_production = false;
        opt.ionization = false;
        opt.bremsstrahlung = false;
        opt.coulomb = false;
        opt.msc = MscModelSelection::none;
        return opt;
    }
};

//! Equality operator
constexpr bool
operator==(GeantMuonPhysicsOptions const& a, GeantMuonPhysicsOptions const& b)
{
    // clang-format off
    return a.pair_production == b.pair_production
           && a.ionization == b.ionization
           && a.bremsstrahlung == b.bremsstrahlung
           && a.coulomb == b.coulomb
           && a.msc == b.msc;
    // clang-format on
}

//---------------------------------------------------------------------------//
/*!
 * Construction options for Geant physics.
 *
 * These options attempt to default to our closest match to \c
 * G4StandardEmPhysics. They are passed to the \c EmPhysicsList
 * and \c FtfpBert physics lists to provide an easy way to set up
 * physics options.
 *
 * \todo This will be moved to celeritas::inp
 * \todo Rename default_cutoff to be consistent (it's a production cut)
 */
struct GeantPhysicsOptions
{
    using MevEnergy = Quantity<units::Mev, double>;

    //!@{
    //! \name Gamma physics

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

    //! Enable discrete Coulomb
    bool coulomb_scattering{false};
    //! Enable e- and e+ ionization
    bool ionization{true};
    //! Enable positron annihilation
    bool annihilation{true};
    //! Enable bremsstrahlung and select a model
    BremsModelSelection brems{BremsModelSelection::all};
    //! Upper limit for the Seltzer-Berger bremsstrahlung model
    MevEnergy seltzer_berger_limit{1e3};  // 1 GeV
    //! Enable multiple coulomb scattering and select a model.
    //! Electron/positron MSC requires ionization
    MscModelSelection msc{MscModelSelection::urban};
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
    //! Tracking cutoff kinetic energy for muons/hadrons
    MevEnergy lowest_muhad_energy{0.001};  // 1 keV
    //! Kill secondaries below the production cut
    bool apply_cuts{false};
    //! Set the default production cut for all particle types [len]
    double default_cutoff{0.1 * units::centimeter};
    //!@}

    //!@{
    //! \name Multiple scattering configuration

    //! e-/e+ range factor for MSC models
    double msc_range_factor{0.04};
    //! Muon/hadron range factor for MSC models
    double msc_muhad_range_factor{0.2};
    //! Safety factor for MSC models
    double msc_safety_factor{0.6};
    //! Lambda limit for MSC models [len]
    double msc_lambda_limit{0.1 * units::centimeter};
    //! Polar angle limit between single and multiple Coulomb scattering
    double msc_theta_limit{constants::pi};
    //! Factor for dynamic computation of angular limit between SS and MSC
    double angle_limit_factor{1};
    //! Whether lateral displacement is enabled for e-/e+ MSC
    bool msc_displaced{true};
    //! Whether lateral displacement is enabled for muon/hadron MSC
    bool msc_muhad_displaced{false};
    //! Step limit algorithm for e-/e+ MSC models
    MscStepLimitAlgorithm msc_step_algorithm{MscStepLimitAlgorithm::safety};
    //! Step limit algorithm for muon/hadron MSC models
    MscStepLimitAlgorithm msc_muhad_step_algorithm{
        MscStepLimitAlgorithm::minimal};
    //! Nuclear form factor model for Coulomm scattering
    NuclearFormFactorType form_factor{NuclearFormFactorType::exponential};
    //!@}

    //! Print detailed Geant4 output
    bool verbose{false};

    //! Muon EM physics
    GeantMuonPhysicsOptions muon{GeantMuonPhysicsOptions::deactivated()};

    //! Muon-catalyzed fusion physics
    bool mucf_physics{false};

    //! Optical physics options
    GeantOpticalPhysicsOptions optical{
        GeantOpticalPhysicsOptions::deactivated()};

    //! True if any EM process is activated
    bool em() const
    {
        return compton_scattering || photoelectric || rayleigh_scattering
               || gamma_conversion || gamma_general || coulomb_scattering
               || ionization || annihilation
               || brems != BremsModelSelection::none
               || msc != MscModelSelection::none;
    }

    //! Initialize with no physics
    static GeantPhysicsOptions deactivated()
    {
        GeantPhysicsOptions opt;
        // Gamma
        opt.compton_scattering = false;
        opt.photoelectric = false;
        opt.rayleigh_scattering = false;
        opt.gamma_conversion = false;
        opt.gamma_general = false;
        // Electron/positron
        opt.coulomb_scattering = false;
        opt.ionization = false;
        opt.annihilation = false;
        opt.brems = BremsModelSelection::none;
        opt.msc = MscModelSelection::none;
        opt.relaxation = RelaxationSelection::none;
        // Muon
        opt.muon = GeantMuonPhysicsOptions::deactivated();
        opt.mucf_physics = false;
        // Optical
        opt.optical = GeantOpticalPhysicsOptions::deactivated();
        return opt;
    }
};

//! Equality operator, mainly for test harness
// TODO: when we require C++20, use `friend bool operator==(...) = default;`
constexpr bool
operator==(GeantPhysicsOptions const& a, GeantPhysicsOptions const& b)
{
    // clang-format off
    return a.compton_scattering == b.compton_scattering
        && a.photoelectric == b.photoelectric
        && a.rayleigh_scattering == b.rayleigh_scattering
        && a.gamma_conversion == b.gamma_conversion
        && a.gamma_general == b.gamma_general
        // Electron and positron
        && a.coulomb_scattering == b.coulomb_scattering
        && a.ionization == b.ionization
        && a.annihilation == b.annihilation
        && a.brems == b.brems
        && a.seltzer_berger_limit == b.seltzer_berger_limit
        && a.msc == b.msc
        && a.relaxation == b.relaxation
        // Muon EM physics
        && a.muon == b.muon
        // Muon-catalyzed fusion physics
        && a.mucf_physics == b.mucf_physics
        // Physics options
        && a.em_bins_per_decade == b.em_bins_per_decade
        && a.eloss_fluctuation == b.eloss_fluctuation
        && a.lpm == b.lpm
        && a.integral_approach == b.integral_approach
        // Cutoff options
        && a.min_energy == b.min_energy
        && a.max_energy == b.max_energy
        && a.linear_loss_limit == b.linear_loss_limit
        && a.lowest_electron_energy == b.lowest_electron_energy
        && a.lowest_muhad_energy == b.lowest_muhad_energy
        && a.apply_cuts == b.apply_cuts
        && a.default_cutoff == b.default_cutoff
        // Multiple scattering configuration
        && a.msc_range_factor == b.msc_range_factor
        && a.msc_muhad_range_factor == b.msc_muhad_range_factor
        && a.msc_safety_factor == b.msc_safety_factor
        && a.msc_lambda_limit == b.msc_lambda_limit
        && a.msc_theta_limit == b.msc_theta_limit
        && a.angle_limit_factor == b.angle_limit_factor
        && a.msc_displaced == b.msc_displaced
        && a.msc_muhad_displaced == b.msc_muhad_displaced
        && a.msc_step_algorithm == b.msc_step_algorithm
        && a.msc_muhad_step_algorithm == b.msc_muhad_step_algorithm
        && a.form_factor == b.form_factor
        && a.verbose == b.verbose
        && a.optical == b.optical;
    // clang-format on
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(BremsModelSelection value);
char const* to_cstring(MscModelSelection value);
char const* to_cstring(RelaxationSelection value);

//---------------------------------------------------------------------------//
// Helper to read the options from a file or stream.
std::istream& operator>>(std::istream& is, GeantPhysicsOptions&);

//---------------------------------------------------------------------------//
// Helper to write the options to a file or stream.
std::ostream& operator<<(std::ostream& os, GeantPhysicsOptions const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
