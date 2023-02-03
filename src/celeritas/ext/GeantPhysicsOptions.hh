//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
 * - \c relaxation: Atomic relaxation selection
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
    MscModelSelection msc{MscModelSelection::urban};
    RelaxationSelection relaxation{RelaxationSelection::none};

    int em_bins_per_decade{7};
    units::MevEnergy min_energy{0.1 * 1e-3};  // 0.1 keV
    units::MevEnergy max_energy{100 * 1e6};  // 100 TeV
    real_type linear_loss_limit{0.01};
};

//! Equality operator, mainly for test harness
// TODO: when we require C++20, use `friend bool operator==(...) = default;`
constexpr bool
operator==(GeantPhysicsOptions const& a, GeantPhysicsOptions const& b)
{
    return a.coulomb_scattering == b.coulomb_scattering
           && a.rayleigh_scattering == b.rayleigh_scattering
           && a.eloss_fluctuation == b.eloss_fluctuation && a.lpm == b.lpm
           && a.integral_approach == b.integral_approach && a.brems == b.brems
           && a.msc == b.msc && a.relaxation == b.relaxation
           && a.em_bins_per_decade == b.em_bins_per_decade
           && a.min_energy == b.min_energy && a.max_energy == b.max_energy
           && a.linear_loss_limit == b.linear_loss_limit;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
