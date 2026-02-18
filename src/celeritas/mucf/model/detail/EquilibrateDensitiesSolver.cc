//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/EquilibrateDensitiesSolver.cc
//---------------------------------------------------------------------------//
#include "EquilibrateDensitiesSolver.hh"

#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the infinity norm (\f$ ||x^{(k)} - x^{(k-1)} ||_\infty \f$)
 * between two consecutive iterations of an \c EquilibriumArray data.
 */
real_type
calc_infinity_norm(EquilibrateDensitiesSolver::EquilibriumArray const& current,
                   EquilibrateDensitiesSolver::EquilibriumArray const& previous)
{
    using MIP = EquilibrateDensitiesSolver::MucfIsoprotologueMolecule;

    real_type max_norm{0};
    for (auto i : range(MIP::size_))
    {
        real_type const diff = std::fabs(current[i] - previous[i]);
        if (diff > max_norm)
        {
            max_norm = diff;
        }
    }
    return max_norm;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with material information.
 */
EquilibrateDensitiesSolver::EquilibrateDensitiesSolver(
    LhdArray const& lhd_densities)
    : lhd_densities_(lhd_densities)
{
    using Iso = MucfIsotope;

    total_density_ = lhd_densities_[Iso::protium]
                     + lhd_densities_[Iso::deuterium]
                     + lhd_densities_[Iso::tritium];
    CELER_ENSURE(total_density_ > 0);
    inv_tot_density_ = real_type{1} / total_density_;
}

//---------------------------------------------------------------------------//
/*!
 * Return equilibrated isoprotologue values.
 */
EquilibrateDensitiesSolver::EquilibriumArray
EquilibrateDensitiesSolver::operator()(real_type temperature)
{
    CELER_EXPECT(temperature > 0);

    using Iso = MucfIsotope;
    using IsoProt = MucfIsoprotologueMolecule;

    // Cache equilibrium constants for this temperature for the while loop
    real_type const k_hd = this->calc_hd_equilibrium_constant(temperature);
    real_type const k_ht = this->calc_ht_equilibrium_constant(temperature);
    real_type const k_dt = this->calc_dt_equilibrium_constant(temperature);

    // Initialize result and set homonuclear molecules values
    EquilibriumArray result;
    result[IsoProt::protium_protium] = lhd_densities_[Iso::protium]
                                       * inv_tot_density_;
    result[IsoProt::deuterium_deuterium] = lhd_densities_[Iso::deuterium]
                                           * inv_tot_density_;
    result[IsoProt::tritium_tritium] = lhd_densities_[Iso::tritium]
                                       * inv_tot_density_;

    EquilibriumArray previous_equilib_dens;
    real_type iter_diff{0};
    auto remaining_iters{this->max_iterations()};
    do
    {
        previous_equilib_dens = result;

        // The ordering (DT -> HT -> HD) in which `this->equilibrate_pair` is
        // called matters, as a different sequence changes the values that are
        // passed into the next call through the `result` array.

        // Equilibrate DT
        this->equilibrate_pair(IsoProt::deuterium_deuterium,
                               IsoProt::tritium_tritium,
                               IsoProt::deuterium_tritium,
                               k_dt,
                               result);
        // Equilibrate HT
        this->equilibrate_pair(IsoProt::protium_protium,
                               IsoProt::tritium_tritium,
                               IsoProt::protium_tritium,
                               k_ht,
                               result);
        // Equilibrate HD
        this->equilibrate_pair(IsoProt::protium_protium,
                               IsoProt::deuterium_deuterium,
                               IsoProt::protium_deuterium,
                               k_hd,
                               result);

        // Calculate infinity norm between current and previous iteration
        iter_diff = calc_infinity_norm(result, previous_equilib_dens);

        if (--remaining_iters == 0)
        {
            CELER_LOG(warning)
                << "Equilibration did not converge after "
                << this->max_iterations() << " iterations. Current error is "
                << iter_diff;
            break;
        }
    } while (iter_diff > this->convergence_err());

    for (auto& val : result)
    {
        val *= total_density_;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate equilibrium constant for the
 * \f$ H_2 + D_2 \rightleftharpoons 2HD \f$ reaction.
 */
real_type
EquilibrateDensitiesSolver::calc_hd_equilibrium_constant(real_type temperature)
{
    real_type result;

    if (temperature < 30)
    {
        result = 6.785 * std::exp(-654.3 / (r_gas_ * temperature));
    }
    else
    {
        // r_gas_ constant not included as it cancels out in the exponent
        real_type const c_hd = 30 * (std::log(4) - std::log(0.49));
        result = 4.0 * std::exp(-c_hd / temperature);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate equilibrium constant for the
 * \f$ H_2 + T_2 \rightleftharpoons 2HT \f$ reaction.
 */
real_type
EquilibrateDensitiesSolver::calc_ht_equilibrium_constant(real_type temperature)
{
    real_type result;

    if (temperature < 30)
    {
        result = 10.22 * std::exp(-1423 / (r_gas_ * temperature));
    }
    else
    {
        // r_gas_ constant not included as it cancels out in the exponent
        real_type const c_ht = 30 * (std::log(4) - std::log(0.034));
        result = 4.0 * std::exp(-c_ht / temperature);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate equilibrium constant for the
 * \f$ D_2 + T_2 \rightleftharpoons 2DT \f$ reaction.
 */
real_type
EquilibrateDensitiesSolver::calc_dt_equilibrium_constant(real_type temperature)
{
    real_type result;

    if (temperature < 15)
    {
        result = 5.924 * std::exp(-168.3 / (r_gas_ * temperature));
    }
    else if (temperature < 30)
    {
        result = 2.995 * std::exp(-89.96 / (r_gas_ * temperature));
    }
    else if (temperature < 100)
    {
        // r_gas_ constant not included as it cancels out in the exponent
        real_type const c_dt = 30 * (std::log(4) - std::log(2.09));
        result = 4.0 * std::exp(-c_dt / temperature);
    }
    else
    {
        // r_gas_ constant not included as it cancels out in the exponent
        real_type const c_dt = 100 * (std::log(4) - std::log(3.29));
        result = 4.0 * std::exp(-c_dt / temperature);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Equilibrate a pair of isotopes and return the new density.
 *
 * Since there are 3 isotopes (H, D, and T), and 6 molecular combinations, the
 * equilibrium cannot be solved at once and has to be done iteratively for each
 * pair until a convergence criterion is met.
 *
 * Therefore, this function takes 2 isotope combinations (e.g. DD, TT, and DT),
 * the equilibrium constant for this temperature, and calculates how much of
 * the homonuclear molecules (e.g. DD and TT) convert to the heteronuclear
 * molecule (e.g. DT).
 *
 * The new densities are written into the input array.
 */
void EquilibrateDensitiesSolver::equilibrate_pair(
    MucfIsoprotologueMolecule molecule_aa,
    MucfIsoprotologueMolecule molecule_bb,
    MucfIsoprotologueMolecule molecule_ab,
    real_type eq_constant_ab,
    EquilibriumArray& input)
{
    CELER_EXPECT(molecule_aa < MucfIsoprotologueMolecule::size_);
    CELER_EXPECT(molecule_bb < MucfIsoprotologueMolecule::size_);
    CELER_EXPECT(molecule_ab < MucfIsoprotologueMolecule::size_);
    CELER_EXPECT(eq_constant_ab > 0);

    // AA + AB / 2
    real_type const mix_a = input[molecule_aa]
                            + input[molecule_ab] * real_type{0.5};
    // BB + AB / 2
    real_type const mix_b = input[molecule_bb]
                            + input[molecule_ab] * real_type{0.5};

    real_type sigma
        = ((mix_a + mix_b)
           - std::sqrt(ipow<2>(mix_a - mix_b)
                       + 16 * mix_a * mix_b
                             / (eq_constant_ab - this->convergence_err())))
          / (2 * (1 - 4 / (eq_constant_ab - this->convergence_err())));

    // Write new density into the equilibrium array
    input[molecule_aa] = mix_a - sigma;
    input[molecule_ab] = 2 * sigma;
    input[molecule_bb] = mix_b - sigma;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
