//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Brems selection (TODO: make bitset)
enum class BremsModelSelection
{
    seltzer_berger,
    relativistic,
    all,
    size_
};

//---------------------------------------------------------------------------//
//! MSC selection (TODO: make bitset)
enum class MscModelSelection
{
    none,
    urban,
    wentzel_vi,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Construction options for geant physics.
 *
 * Note that not all Geant physics choices are implemented in Celeritas.
 *
 * - \c coulomb_scattering: enable a discrete Coulomb scattering process
 * - \c rayleigh_scattering: enable the Rayleigh scattering process
 * - \c eloss_fluctuation: enable universal energy fluctuations
 * - \c lpm: apply relativistic corrections for select models
 * - \c integral_approach: see \c PhysicsParamsOptions::disable_integral_xs
 * - \c brems: Bremsstrahlung model selection
 * - \c msc: Multiple scattering model selection
 * - \c em_bins_per_decade: number of log-spaced energy bins per factor of 10
 * - \c min_energy: lowest energy of any EM physics process
 * - \c max_energy: highest energy of any EM physics process
 * - \c linear_loss_limit: see \c PhysicsParamsOptions::linear_loss_limit
 */
struct GeantPhysicsOptions
{
    bool coulomb_scattering{false};
    bool rayleigh_scattering{true};
    bool eloss_fluctuation{true};
    bool lpm{true};
    bool integral_approach{true};

    BremsModelSelection brems{BremsModelSelection::all};
    MscModelSelection   msc{MscModelSelection::urban};

    int              em_bins_per_decade{7};
    units::MevEnergy min_energy{0.1 * 1e-3}; // 0.1 keV
    units::MevEnergy max_energy{100 * 1e6};  // 100 TeV
    real_type        linear_loss_limit{0.01};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
