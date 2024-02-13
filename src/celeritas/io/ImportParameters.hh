//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportParameters.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

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
#ifndef SWIG
    static constexpr auto energy_units{ImportUnits::mev};
#endif

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
    //! Whether auger emission should be enabled (valid only for relaxation)
    bool auger{false};
    //! MSC range factor for e-/e+
    double msc_range_factor{0.04};
    //! MSC safety factor
    double msc_safety_factor{0.6};
    //! MSC lambda limit [length]
    double msc_lambda_limit{1 * units::millimeter};
    //! Kill secondaries below production cut
    bool apply_cuts{false};
    //! Nuclear screening factor for single/multiple Coulomb scattering
    double screening_factor{1};

    //! Whether parameters are assigned and valid
    explicit operator bool() const
    {
        return linear_loss_limit > 0 && lowest_electron_energy > 0
               && msc_range_factor > 0 && msc_range_factor < 1
               && msc_safety_factor >= 0.1 && msc_lambda_limit > 0
               && screening_factor > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Particle-dependent parameters for killing looping tracks.
 */
struct ImportLoopingThreshold
{
#ifndef SWIG
    static constexpr auto energy_units{ImportUnits::mev};
#endif

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
    using ImportLoopingMap = std::unordered_map<PDGInt, ImportLoopingThreshold>;
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
}  // namespace celeritas
