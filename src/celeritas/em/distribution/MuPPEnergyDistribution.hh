//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuPPEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/FindInterp.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/grid/TwodGridCalculator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuPairProductionData.hh"
#include "celeritas/grid/InverseCdfFinder.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the electron and positron energies for muon pair production.
 *
 * The energy transfer to the electron-positron pair is sampled using inverse
 * transform sampling on a tabulated CDF. The CDF is calculated on a 2-D grid,
 * where the *x* axis is the log of the incident muon energy and the *y* axis
 is
 * the log of the ratio of the energy transfer to the incident particle energy.
 * Because the shape of the distribution depends only weakly on the atomic
 * number, the CDF is calculated for a hardcoded set of points equally spaced
 * in \f$ \log Z \f$ and linearly interpolated.
 *
 * The formula used for the differential cross section is valid when the
 * maximum energy transfer to the electron-positron pair lies between \f$
 * \epsilon_{\text{min}} = 4 m \f$, where \f$ m \f$ is the electron mass, and
 * \f[
   \epsilon_{\text{max}} = E + \frac{3 \sqrt{e}}{4} \mu Z^{1/3},
 * \f]
 * where \f$ E = T + \mu \f$ is the total muon energy, \f$ \mu \f$ is the muon
 * mass, \f$ e \f$ is Euler's number, and \f$ Z \f$ is the atomic number.
 *
 * The maximum energy partition between the electron and positron is calculated
 * as
 * \f[
   r_{\text{max}} = \left[1 - 6 \frac{\mu^2}{E (E - \epsilon)} \right] \sqrt{1
   - \epsilon_{\text{min}} / \epsilon}.
 * \f]
 * The partition \f$ r \f$ is then sampled uniformly in \f$ [-r_{\text{max}},
 * r_{\text{max}}) \f$.
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

  private:
    //// DATA ////

    // CDF table for sampling the pair energy
    NativeCRef<MuPairProductionTableData> const& table_;
    // Incident particle energy [MeV]
    real_type inc_energy_;
    // Incident particle total energy [MeV]
    real_type total_energy_;
    // Square of the muon mass
    real_type inc_mass_sq_;
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

    // Calculate the scaled energy for a given Z grid and sampled CDF
    inline CELER_FUNCTION real_type calc_scaled_energy(size_type z_idx,
                                                       real_type u) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and incident particle data.
 *
 * The incident energy *must* be within the bounds of the sampling table data.
 */
CELER_FUNCTION
MuPPEnergyDistribution::MuPPEnergyDistribution(
    NativeCRef<MuPairProductionData> const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    ElementView const& element)
    : table_(shared.table)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , total_energy_(value_as<Energy>(particle.total_energy()))
    , inc_mass_sq_(ipow<2>(value_as<Mass>(particle.mass())))
    , electron_mass_(value_as<Mass>(shared.electron_mass))
    , min_pair_energy_(4 * value_as<Mass>(shared.electron_mass))
    , max_pair_energy_(inc_energy_
                       + value_as<Mass>(particle.mass())
                             * (1
                                - real_type(0.75) * constants::sqrt_euler
                                      * element.cbrt_z()))
    , min_energy_(max(value_as<Energy>(cutoffs.energy(shared.ids.positron)),
                      min_pair_energy_))
{
    CELER_EXPECT(max_pair_energy_ > min_energy_);

    NonuniformGrid logz_grid(table_.logz_grid, table_.reals);
    logz_interp_ = find_interp(logz_grid, element.log_z());

    NonuniformGrid y_grid(
        table_.grids[ItemId<TwodGridData>(logz_interp_.index)].y, table_.reals);
    coeff_ = std::log(min_pair_energy_ / inc_energy_) / y_grid.front();

    // Compute the bounds on the ratio of the pair energy to incident energy
    y_min_ = std::log(min_energy_ / inc_energy_) / coeff_;
    y_max_ = std::log(max_pair_energy_ / inc_energy_) / coeff_;

    // Check that the bounds are within the grid bounds
    CELER_ASSERT(y_min_ >= y_grid.front()
                 || soft_equal(y_grid.front(), y_min_));
    CELER_ASSERT(y_max_ <= y_grid.back() || soft_equal(y_grid.back(), y_max_));
    y_min_ = max(y_min_, y_grid.front());
    y_max_ = min(y_max_, y_grid.back());
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

    // Calculate the electron and positron energies
    PairEnergy result;
    real_type half_energy = pair_energy * real_type(0.5);
    result.electron = Energy((1 - r) * half_energy - electron_mass_);
    result.positron = Energy((1 + r) * half_energy - electron_mass_);

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
    real_type u = generate_canonical(rng);
    LinearInterpolator<real_type> interp_energy{
        {0, this->calc_scaled_energy(logz_interp_.index, u)},
        {1, this->calc_scaled_energy(logz_interp_.index + 1, u)}};
    return interp_energy(logz_interp_.fraction);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the scaled energy for a given Z grid and sampled CDF value.
 */
CELER_FUNCTION real_type
MuPPEnergyDistribution::calc_scaled_energy(size_type z_idx, real_type u) const
{
    CELER_EXPECT(z_idx < table_.grids.size());
    CELER_EXPECT(u >= 0 && u < 1);

    TwodGridData const& cdf_grid = table_.grids[ItemId<TwodGridData>(z_idx)];
    auto calc_cdf
        = TwodGridCalculator(cdf_grid, table_.reals)(std::log(inc_energy_));

    // Get the sampled CDF value between the y bounds
    real_type cdf = LinearInterpolator<real_type>{{0, calc_cdf(y_min_)},
                                                  {1, calc_cdf(y_max_)}}(u);

    // Find the grid index of the sampled CDF value
    return InverseCdfFinder(NonuniformGrid(cdf_grid.y, table_.reals),
                            std::move(calc_cdf))(cdf);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
