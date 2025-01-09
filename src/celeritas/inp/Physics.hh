//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Physics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/model/SeltzerBergerModel.hh"
#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ProcessBuilder.hh"

namespace celeritas
{
namespace inp
{
#if 0
//---------------------------------------------------------------------------//
/*!
 * Set up multiple scattering options.
 *
 * TODO: some of these are moved from \c ImportEmParameters : we should import
 * more and change them around a bit.
 */
struct MscOptions
{
    //! Algorithm for step behavior near boundaries
    MscStepLimitAlgorithm step_limit{MscStepLimitAlgorithm::safety};

    //! MSC range factor for e-/e+
    real_type range_factor{0.04};
    //! MSC safety factor
    real_type safety_factor{0.6};
    //! MSC lambda limit [length]
    real_type lambda_limit{1 * units::millimeter};
    //! Polar angle limit between single and multiple Coulomb scattering
    real_type theta_limit{constants::pi};

    // TODO: unit system support?
    static inline constexpr UnitSystem units{UnitSystem::native};
};
#endif

struct SeltzerBergerBremsModel
{
    using Table = std::map<AtomicNumber, ImportPhysics2DVector>;

    Table sb_tables;
    // microscopic cross sections
};

struct RelBremsModel
{
    bool enable_lpm{true};  //!> Account for LPM effect at very high energies
    MicroXsTable micros;
};

struct BremsstrahlungProcess
{
    std::optional<SeltzerBergerBremsModel> sb;
    std::optional<RelBremsModel> rel;
    bool combined_model{true};  //!> Use a unified relativistic/SB
                                //! interactor
    bool use_integral_xs{true};  //!> Use integral method for sampling
                                 //! discrete interaction length
    // macroscopic cross sections
};

//---------------------------------------------------------------------------//
/*!
 * Set up electromagnetic physics options.
 */
struct EmPhysicsOptions
{
    //!@{
    //! \name Energy loss and slowing down
    //! Second-order spline interpolation for energy loss
    bool eloss_spline{false};
    // TODO: //! Energy loss fluctuations
    // TODO: bool eloss_fluct{true};
    // TODO: //! Integral cross section rejection
    // TODO: bool integral_xs_rejection{true};
    //!@}

    //!@{
    //! \name Model options
    //! Use LPM corrections for high-energy bremsstrahlung and pair production
    bool lpm{true};
    //! Use combined SB/relativistic interactor for bremsstrahlung
    bool brem_combined{false};
    //!@}
};

struct EmPhysics
{
    std::optional<BremsstrahlungProcess> brems;
    std::optional<GammaConversionProcess> pp;
    std::optional<ThreeGamma> three_gamma;
    EmPhysicsOptions em_options;
};

struct EmExtraPhysics
{
};

struct OpticalPhysics
{
    // Optical material map
    // Properties per material
};

struct MuDecayPhysics
{
    // Optical material map
    // Properties per material
};
struct HadronicPhysics
{
    // ftfp_bert vs qgsp_bic
};

//---------------------------------------------------------------------------//
/*!
 * Set up physics options.
 *
 * TODO: refactor ignore_processes so it ties in the with IO classes.
 */
struct Physics
{
    // If false, use geant4 defaults
    std::optional<EmPhysics> em;
    std::optional<OpticalPhysics> optical;
    std::optional<HadronicPhysics> hadronic;

    //! Electromagnetic physics options

    //! Do not use Celeritas physics for the given Geant4 process names
    std::vector<std::string> ignore_processes;

    ProcessBuilder::UserBuildFunction user_processes;

    //! Import physics from a file instead of Geant4
    // TODO: external driver only std::string physics_file;

    // TODO: particle selection
    // TODO: user process builder
    // TODO: where/how should celer-g4 set up physics list?
    // TODO: read Geant4, then callback to update?
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
