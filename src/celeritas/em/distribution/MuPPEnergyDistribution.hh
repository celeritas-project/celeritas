//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuPPEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/FindInterp.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/grid/TwodGridCalculator.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuPairProductionData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample electron-positron pair energy for muon pair production.
 */
class MuPPEnergyDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    //!@}

    //! Sampled secondary energies
    struct PairEnergy
    {
        Energy electron;
        Energy positron;
    };

  public:
    // Construct from shared and incident particle data
    inline CELER_FUNCTION
    MuPPEnergyDistribution(NativeCRef<MuPairProductionData> const& shared,
                           ParticleTrackView const& particle,
                           CutoffView const& cutoffs,
                           ElementView const& element);

    template<class Engine>
    inline CELER_FUNCTION PairEnergy operator()(Engine& rng);

    //! Minimum energy of the electron-positron pair [MeV].
    CELER_FUNCTION Energy min_pair_energy() const
    {
        return Energy(min_pair_energy_);
    }

    //! Maximum energy of the electron-positron pair [MeV].
    CELER_FUNCTION Energy max_pair_energy() const
    {
        return Energy(max_pair_energy_);
    }

    //! Minimum incident particle kinetic energy [MeV].
    CELER_FUNCTION Energy min_energy() const { return Energy(min_energy_); }

  private:
    //// DATA ////

    // Table for sampling the pair energy
    NativeCRef<MuPairProductionTableData> const& table_;
    // Incident particle energy [MeV]
    real_type inc_energy_;
    // Log of incident particle energy
    real_type log_energy_;
    // Square of the muon mass
    real_type inc_mass_sq_;
    // Incident particle total energy [MeV]
    real_type total_energy_;
    // Secondary mass
    real_type electron_mass_;
    // Minimum energy transfer to electron/positron pair [MeV]
    real_type min_pair_energy_;
    // Maximum energy transfer to electron/positron pair [MeV]
    real_type max_pair_energy_;
    // Minimum incident particle kinetic energy [MeV]
    real_type min_energy_;
    // Log Z grid interpolation for the target element
    FindInterp<real_type> logz_interp_;
    // Coefficient for calculating the pair energy
    real_type coeff_;
    // Lower bound on the ratio of the pair energy to the incident energy
    real_type y_min_;
    // Upper bound on the ratio of the pair energy to the incident energy
    real_type y_max_;

    //// HELPER FUNCTIONS ////

    // Sample the scaled energy and interpolate in log Z
    template<class Engine>
    inline CELER_FUNCTION real_type sample_scaled_energy(Engine& rng) const;

    // Sample the scaled energy for a given Z
    template<class Engine>
    inline CELER_FUNCTION real_type sample_scaled_energy(size_type z_idx,
                                                         Engine& rng) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and incident particle data.
 *
 * The incident energy *must* be within the bounds of the sampling table data,
 * so the model's applicability must be consistent with the table data.
 */
CELER_FUNCTION
MuPPEnergyDistribution::MuPPEnergyDistribution(
    NativeCRef<MuPairProductionData> const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    ElementView const& element)
    : table_(shared.table)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , log_energy_(std::log(inc_energy_))
    , inc_mass_sq_(ipow<2>(value_as<Mass>(particle.mass())))
    , total_energy_(value_as<Energy>(particle.total_energy()))
    , electron_mass_(value_as<Mass>(shared.electron_mass))
    , min_pair_energy_(4 * value_as<Mass>(shared.electron_mass))
    , max_pair_energy_(inc_energy_
                       + value_as<Mass>(particle.mass())
                             * (1
                                - real_type(0.75) * std::sqrt(constants::euler)
                                      * element.cbrt_z()))
    , min_energy_(max(value_as<Energy>(cutoffs.energy(shared.ids.positron)),
                      min_pair_energy_))
{
    CELER_EXPECT(max_pair_energy_ > min_pair_energy_);

    NonuniformGrid logz_grid(table_.logz_grid, table_.reals);
    logz_interp_ = find_interp(logz_grid, element.log_z());

    NonuniformGrid y_grid(
        table_.grids[ItemId<TwodGridData>(logz_interp_.index)].y, table_.reals);
    coeff_ = std::log(min_pair_energy_ / inc_energy_) / y_grid.front();

    // Compute the bounds on the ratio of the pair energy to incident energy
    y_min_ = std::log(min_energy_ / inc_energy_) / coeff_;
    y_max_ = std::log(max_pair_energy_ / inc_energy_) / coeff_;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the exiting pair energy.
 */
template<class Engine>
CELER_FUNCTION auto MuPPEnergyDistribution::operator()(Engine& rng)
    -> PairEnergy
{
    // Sample the energy transfer
    real_type pair_energy
        = inc_energy_ * std::exp(coeff_ * this->sample_scaled_energy(rng));
    CELER_ASSERT(pair_energy >= min_energy_ && pair_energy <= max_pair_energy_);

    // Sample the energy partition between the electron and positron
    real_type r_max = (1
                       - 6 * inc_mass_sq_
                             / (total_energy_ * (total_energy_ - pair_energy)))
                      * std::sqrt(1 - min_pair_energy_ / pair_energy);
    real_type r = UniformRealDistribution(-r_max, r_max)(rng);

    PairEnergy result;
    result.electron
        = Energy((1 - r) * pair_energy * real_type(0.5) - electron_mass_);
    result.positron
        = Energy((1 + r) * pair_energy * real_type(0.5) - electron_mass_);

    CELER_ENSURE(result.electron > zero_quantity());
    CELER_ENSURE(result.positron > zero_quantity());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scaled energy and interpolate in log Z.
 */
template<class Engine>
CELER_FUNCTION real_type
MuPPEnergyDistribution::sample_scaled_energy(Engine& rng) const
{
    real_type y_lower = this->sample_scaled_energy(logz_interp_.index, rng);
    real_type y_upper = this->sample_scaled_energy(logz_interp_.index + 1, rng);
    return y_lower + (y_upper - y_lower) * logz_interp_.fraction;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scaled energy for a given Z.
 */
template<class Engine>
CELER_FUNCTION real_type
MuPPEnergyDistribution::sample_scaled_energy(size_type z_idx, Engine& rng) const
{
    CELER_EXPECT(z_idx < table_.grids.size());

    TwodGridData const& cdf_grid = table_.grids[ItemId<TwodGridData>(z_idx)];
    auto calc_cdf = TwodGridCalculator(cdf_grid, table_.reals)(log_energy_);

    // Sample the CDF value between the y bounds
    UniformRealDistribution sample_cdf(calc_cdf(y_min_), calc_cdf(y_max_));
    real_type cdf = sample_cdf(rng);

    NonuniformGrid y_grid(cdf_grid.y, table_.reals);

    // Find the y value corresponding to the sampled CDF value
    // TODO: refactor as CDF sampler and use a binary search
    size_type idx = y_grid.size() - 2;
    real_type cdf_lower = 1;
    real_type cdf_upper;
    do
    {
        cdf_upper = cdf_lower;
        cdf_lower = calc_cdf(y_grid[idx]);
    } while (cdf_lower > cdf && idx-- > 0);

    real_type frac = (cdf - cdf_lower) / (cdf_upper - cdf_lower);
    return fma(frac, y_grid[idx + 1] - y_grid[idx], y_grid[idx]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
