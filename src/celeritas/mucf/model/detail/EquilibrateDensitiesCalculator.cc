//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/detail/EquilibrateDensitiesCalculator.cc
//---------------------------------------------------------------------------//
#include "EquilibrateDensitiesCalculator.hh"

#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with material information.
 */
EquilibrateDensitiesCalculator::EquilibrateDensitiesCalculator(
    LhdArray const& lhd_densities, real_type const temperature)
    : lhd_densities_(lhd_densities), temperature_(temperature)
{
    CELER_EXPECT(temperature > 0);
    CELER_EXPECT(lhd_densities[MucfIsotope::protium]
                     + lhd_densities[MucfIsotope::deuterium]
                     + lhd_densities[MucfIsotope::tritium]
                 > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Return equilibrated isoprotologue values.
 */
EquilibrateDensitiesCalculator::EquilibriumArray
EquilibrateDensitiesCalculator::operator()()
{
    using Iso = MucfIsotope;
    using IsoProt = MucfIsoprotologueMolecule;

    real_type const total_density = lhd_densities_[Iso::protium]
                                    + lhd_densities_[Iso::deuterium]
                                    + lhd_densities_[Iso::tritium];
    CELER_ASSERT(total_density > 0);
    real_type const inv_tot_density = real_type{1} / total_density;

    // Cache equilibrium constants for this temperature for the while loop
    real_type const k_hd = this->calc_hd_equilibrium_constant();
    real_type const k_dt = this->calc_dt_equilibrium_constant();
    real_type const k_ht = this->calc_ht_equilibrium_constant();

    // Initialize result and calculate equilibrium densities
    EquilibriumArray result;
    result[IsoProt::protium_protium] = lhd_densities_[Iso::protium]
                                       * inv_tot_density;
    result[IsoProt::deuterium_deuterium] = lhd_densities_[Iso::deuterium]
                                           * inv_tot_density;
    result[IsoProt::tritium_tritium] = lhd_densities_[Iso::tritium]
                                       * inv_tot_density;
    result[IsoProt::protium_deuterium] = 0;
    result[IsoProt::deuterium_tritium] = 0;
    result[IsoProt::protium_tritium] = 0;

    EquilibriumArray previous_equilib_dens = result;
    auto iter_diff = std::numeric_limits<real_type>::infinity();
    size_type iter{0};
    while (iter_diff > this->convergence_err() && iter < this->max_iterations())
    {
        // Equilibrate HD
        this->equilibrate_pair(IsoProt::protium_protium,
                               IsoProt::deuterium_deuterium,
                               IsoProt::protium_deuterium,
                               k_hd,
                               result);
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

        for (auto const& i : range(MucfIsoprotologueMolecule::size_))
        {
            // Calculate difference between current and previous densities
            real_type diff = std::abs(result[i] - previous_equilib_dens[i]);
            if (diff > iter_diff)
            {
                // Select maximum difference for convergence check
                iter_diff = diff;
            }
        }
        // Save current state to compare with next iteration
        previous_equilib_dens = result;
        iter++;
    }

    if (iter == this->max_iterations())
    {
        CELER_LOG(warning) << "Equilibration did not converge after "
                           << max_iterations()
                           << " iterations. Current error is " << iter_diff;
    }

    for (auto& dens : result)
    {
        dens *= total_density;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate equilibrium constant for the
 * \f$ H_2 + D_2 \rightleftharpoons 2HD \f$ reaction.
 */
real_type EquilibrateDensitiesCalculator::calc_hd_equilibrium_constant()
{
    real_type result;

    if (temperature_ < 30)
    {
        result = 6.785 * exp(-654.3 / (r_gas_constant_ * temperature_));
    }
    else
    {
        real_type const c_hd = r_gas_constant_ * 30 * (log(4) - log(0.49));
        result = (4.0 * exp(-c_hd / (r_gas_constant_ * temperature_)));
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate equilibrium constant for the
 * \f$ H_2 + T_2 \rightleftharpoons 2HT \f$ reaction.
 */
real_type EquilibrateDensitiesCalculator::calc_ht_equilibrium_constant()
{
    real_type result;

    if (temperature_ < 30)
    {
        result = 10.22 * exp(-1423 / (r_gas_constant_ * temperature_));
    }
    else
    {
        real_type const c_ht = r_gas_constant_ * 30 * (log(4) - log(0.034));
        result = (4.0 * exp(-c_ht / (r_gas_constant_ * temperature_)));
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate equilibrium constant for the
 * \f$ D_2 + T_2 \rightleftharpoons 2DT \f$ reaction.
 */
real_type EquilibrateDensitiesCalculator::calc_dt_equilibrium_constant()
{
    real_type result;

    if (temperature_ < 15)
    {
        result = 5.924 * exp(-168.3 / (r_gas_constant_ * temperature_));
    }
    else if (temperature_ < 30)
    {
        result = 2.995 * exp(-89.96 / (r_gas_constant_ * temperature_));
    }
    else if (temperature_ < 100)
    {
        real_type const c_dt = r_gas_constant_ * 30 * (log(4) - log(2.09));
        result = 4.0 * exp(-c_dt / (r_gas_constant_ * temperature_));
    }
    else
    {
        real_type const c_dt = r_gas_constant_ * 100 * (log(4) - log(3.29));
        result = 4.0 * exp(-c_dt / (r_gas_constant_ * temperature_));
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
void EquilibrateDensitiesCalculator::equilibrate_pair(
    MucfIsoprotologueMolecule molecule_aa,
    MucfIsoprotologueMolecule molecule_bb,
    MucfIsoprotologueMolecule molecule_ab,
    real_type eq_constant_ab,
    EquilibriumArray& input)
{
    auto const& dens_aa = input[molecule_aa];
    auto const& dens_bb = input[molecule_bb];
    auto const& dens_ab = input[molecule_ab];

    // (AA + AB) / 2
    real_type const mix_a = (dens_aa + dens_ab) * real_type{0.5};
    // (BB + AB) / 2
    real_type const mix_b = (dens_bb + dens_ab) * real_type{0.5};

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
