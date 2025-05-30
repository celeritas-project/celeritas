//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportParameters.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>

#include "celeritas/Constants.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

#include "ImportUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Common electromagnetic physics parameters (see G4EmParameters.hh).
 *
 * \note Geant4 v11 removed the Spline() option from G4EmParameters.hh.
 * \note The Geant4 MSC models use the values in \c G4EmParameters as the
 * defaults; however, the MSC parameters can also be set directly using the
 * model setter methods (there is no way to retrieve the values from the model
 * in that case).
 */
struct ImportEmParameters
{
    static constexpr auto energy_units{ImportUnits::mev};

    //! Energy loss fluctuation
    bool energy_loss_fluct{false};
    //! LPM effect for bremsstrahlung and pair production
    bool lpm{true};
    //! Integral cross section rejection
    bool integral_approach{true};
    //! Slowing down threshold for linearity assumption
    double linear_loss_limit{0.01};
    //! Lowest e-/e+ kinetic energy [MeV]
    double lowest_electron_energy{0.001};
    //! Lowest muon/hadron kinetic energy [MeV]
    double lowest_muhad_energy{0.001};
    //! Whether auger emission should be enabled (valid only for relaxation)
    bool auger{false};
    //! MSC step limit algorithm for e-/e+
    MscStepLimitAlgorithm msc_step_algorithm{MscStepLimitAlgorithm::safety};
    //! MSC step limit algorithm for muon/hadron
    MscStepLimitAlgorithm msc_muhad_step_algorithm{
        MscStepLimitAlgorithm::minimal};
    //! MSC lateral displacement for e-/e+
    double msc_displaced{true};
    //! MSC lateral displacement for muon/hadron
    double msc_muhad_displaced{false};
    //! MSC range factor for e-/e+
    double msc_range_factor{0.04};
    //! MSC range factor for muon/hadron
    double msc_muhad_range_factor{0.2};
    //! MSC safety factor
    double msc_safety_factor{0.6};
    //! MSC lambda limit [length]
    double msc_lambda_limit{1 * units::millimeter};
    //! Polar angle limit between single and multiple Coulomb scattering
    double msc_theta_limit{constants::pi};
    //! Kill secondaries below production cut
    bool apply_cuts{false};
    //! Nuclear screening factor for single/multiple Coulomb scattering
    double screening_factor{1};
    //! Factor for dynamic computation of angular limit between SS and MSC
    double angle_limit_factor{1};
    //! Nuclear form factor model for Coulomb scattering
    NuclearFormFactorType form_factor{NuclearFormFactorType::exponential};

    //! Whether parameters are assigned and valid
    explicit operator bool() const
    {
        return linear_loss_limit > 0 && lowest_electron_energy > 0
               && msc_step_algorithm != MscStepLimitAlgorithm::size_
               && msc_muhad_step_algorithm != MscStepLimitAlgorithm::size_
               && msc_range_factor > 0 && msc_range_factor < 1
               && msc_muhad_range_factor > 0 && msc_muhad_range_factor < 1
               && msc_safety_factor >= 0.1 && msc_lambda_limit > 0
               && msc_theta_limit >= 0 && msc_theta_limit <= constants::pi
               && screening_factor > 0 && angle_limit_factor > 0
               && form_factor != NuclearFormFactorType::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Particle-dependent parameters for killing looping tracks.
 */
struct ImportLoopingThreshold
{
    static constexpr auto energy_units{ImportUnits::mev};

    //! Number of steps a higher-energy looping track takes before it's killed
    int threshold_trials{10};
    //! Energy below which looping tracks are immediately killed [MeV]
    double important_energy{250};

    //! Whether parameters are assigned and valid
    explicit operator bool() const
    {
        return threshold_trials > 0 && important_energy >= 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Parameters related to transportation.
 *
 * The looping thresholds are particle-dependent and stored in a map where the
 * keys are the PDG number.
 */
struct ImportTransParameters
{
    //!@{
    //! \name Type aliases
    using PDGInt = int;
    using ImportLoopingMap = std::map<PDGInt, ImportLoopingThreshold>;
    //!@}

    //! Thresholds for killing looping tracks
    ImportLoopingMap looping;
    //! Maximum number of substeps in the field propagator
    int max_substeps{1000};

    //! Whether parameters are assigned and valid
    explicit operator bool() const
    {
        return max_substeps >= 0 && !looping.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Optical parameter options imported from Geant4.
 *
 * See \c G4OpticalParameters .
 */
struct ImportOpticalParameters
{
    bool scintillation_by_particle{false};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
