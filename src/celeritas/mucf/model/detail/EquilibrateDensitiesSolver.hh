//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/EquilibrateDensitiesSolver.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "celeritas/Constants.hh"
#include "celeritas/mucf/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate dt mixture densities after reaching thermodynamic
 * equilibrium based isotopic fraction, density, and material temperature.
 *
 * Based on the theory from https://www.osti.gov/biblio/6205719.
 *
 * The equilibrated densities are needed to correctly calculate the cycle time
 * of dd, dt, and tt fusion cycles.
 *
 * \sa MucfMaterialInserter
 */
class EquilibrateDensitiesSolver
{
  public:
    //! Enum for safely accessing hydrogen isoprotologue molecules
    enum class MucfIsoprotologueMolecule
    {
        protium_protium,
        protium_deuterium,
        protium_tritium,
        deuterium_deuterium,
        deuterium_tritium,
        tritium_tritium,
        size_
    };

    //!@{
    //! \name Type aliases
    using LhdArray = EnumArray<MucfIsotope, real_type>;
    using EquilibriumArray = EnumArray<MucfIsoprotologueMolecule, real_type>;
    //!@}

    // Construct with material information
    EquilibrateDensitiesSolver(LhdArray const& lhd_densities);

    // Calculate equilibrated isoprotologue densities
    EquilibriumArray operator()(real_type const temperature);

  private:
    //// DATA ////

    LhdArray lhd_densities_;  //!< Number densities in units of LHD
    real_type total_density_;  //!< Total LHD density
    real_type inv_tot_density_;  //!< Inverse of total LHD density
    static constexpr Constant r_gas_ = constants::k_boltzmann
                                       * constants::na_avogadro;

    //// HELPER FUNCTIONS ////

    // {
    // Convergence limit parameters
    // Acceptance error between current and previous equilibrium iteration
    static real_type constexpr convergence_err() { return 1e-6; }

    // Maximum number of iterations to reach convergence
    static size_type constexpr max_iterations() { return 1000; }
    // }

    // Calculate equilibrium constant: \f$ H_2 + D_2 \rightleftharpoons 2HD \f$
    real_type calc_hd_equilibrium_constant(real_type temperature);

    // Calculate equilibrium constant: \f$ H_2 + T_2 \rightleftharpoons 2HT \f$
    real_type calc_ht_equilibrium_constant(real_type temperature);

    // Calculate equilibrium constant: \f$ D_2 + T_2 \rightleftharpoons 2DT \f$
    real_type calc_dt_equilibrium_constant(real_type temperature);

    // Equilibrate a pair of isotopes and write the new density in the array
    void equilibrate_pair(MucfIsoprotologueMolecule molecule_aa,
                          MucfIsoprotologueMolecule molecule_bb,
                          MucfIsoprotologueMolecule molecule_ab,
                          real_type eq_constant_ab,
                          EquilibriumArray& input);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
