//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/detail/MomentumTransferSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/IsotopeView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the momentum transfer (\f$ Q^{2} = -t \f$) of the neutron-nucleus
 * elastic scattering.
 *
 * \note This performs the same sampling routine as in Geant4's
 *  G4ChipsElasticModel and G4ChipsNeutronElasticXS where the differential
 *  cross section for quark-exchange and the amplitude of the scattering are
 *  parameterized in terms of the neutron momentum (GeV/c) in the lab frame.
 */
class MomentumTransferSampler
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    using AtomicMassNumber = AtomicNumber;
    //!@}

  public:
    // Construct with shared and target data, and the neutron momentum
    inline CELER_FUNCTION
    MomentumTransferSampler(NeutronElasticRef const& shared,
                            IsotopeView const& target,
                            Momentum neutron_p);

    // Sample the momentum transfer
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    ChipsDiffXsCoefficients::ChipsArray const& par_;

    // Mass of neutron and target
    Mass neutron_mass_;
    Mass target_mass_;

    // Atomic mass number (A) of the target
    AtomicMassNumber amass_;
    bool heavy_target_{false};
    // Momentum magnitude of the incident neutron [MeV/c]
    Momentum neutron_p_;
    // Maximum momentum transfer for the elastic scattering [GeV^2/c^2]
    real_type max_q_sq_;
    // Parameters for the CHIPS differential cross section
    ExchangeParameters par_q_sq_;

    //// HELPER FUNCTIONS ////

    // Sample the slope
    template<class Engine>
    inline CELER_FUNCTION real_type sample_q_sq(real_type radius, Engine& rng)
    {
        return -std::log(1 - radius * generate_canonical(rng));
    }

    // Calculate the maximum momentum transfer
    inline CELER_FUNCTION real_type calc_max_q_sq(Momentum p) const;

    // Calculate parameters used in the quark-exchange process
    inline CELER_FUNCTION ExchangeParameters calc_par_q_sq(Momentum p) const;

    //// COMMON PROPERTIES ////

    // Covert from clhep::MeV value to clhep::GeV value
    static CELER_CONSTEXPR_FUNCTION real_type to_gev() { return 1e-3; }

    // S-wave limit for neutron, log(p) < -4.3 (GeV/c) (kinetic energy < 0.1
    // MeV)
    static CELER_CONSTEXPR_FUNCTION Momentum s_wave_limit()
    {
        return Momentum{13.568559012200934};  // = exp(-4.3) * 1000
    }

    // Limit of the slope square
    static CELER_CONSTEXPR_FUNCTION real_type tolerance_slope_sq()
    {
        return 1e-7;
    }
};

//---------------------------------------------------------------------------//
// Inline DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the incident momentum in the lab frame and the target(A).
 *
 * \note The incident neutron momentum, and neutron and nucleus masses are
 *  converted to the GeV value .
 */
CELER_FUNCTION
MomentumTransferSampler::MomentumTransferSampler(NeutronElasticRef const& shared,
                                                 IsotopeView const& target,
                                                 Momentum neutron_p)
    : par_(shared.coeffs[target.isotope_id()].par)
    , neutron_mass_(shared.neutron_mass)
    , target_mass_(target.nuclear_mass())
    , amass_(target.atomic_mass_number())
    , heavy_target_(amass_.get() > 6)
    , neutron_p_(neutron_p)
    , max_q_sq_(calc_max_q_sq(neutron_p_))
    , par_q_sq_(calc_par_q_sq(neutron_p_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the momentum transfer of the neutron-nucleus elastic scattering
 * based on G4ChipsElasticModel and G4ChipsNeutronElasticXS of the Geant4
 * 11.2 release.
 *
 */
template<class Engine>
CELER_FUNCTION auto MomentumTransferSampler::operator()(Engine& rng)
    -> real_type
{
    // Sample \f$ Q^{2} \f$ below S-wave limit
    if (neutron_p_ < this->s_wave_limit())
    {
        return max_q_sq_ * generate_canonical(rng) / ipow<2>(this->to_gev());
    }

    // Sample \f$ Q^{2} \f$
    real_type q_sq = 0;

    if (amass_ == AtomicMassNumber{1})
    {
        // Special case for the \f$ n + p \rightarrow n + p \f$ channel
        real_type const r[2] = {-std::expm1(-max_q_sq_ * par_q_sq_.slope[0]),
                                -std::expm1(-max_q_sq_ * par_q_sq_.slope[1])};

        real_type const mi[2]
            = {r[0] * par_q_sq_.expnt[0],
               r[1] * par_q_sq_.expnt[1] / par_q_sq_.slope[1]};

        // Sample by t-channel and u-channel (charge exchange)
        q_sq = (BernoulliDistribution(mi[0], mi[1])(rng))
                   ? sample_q_sq(r[0], rng) / par_q_sq_.slope[0]
                   : max_q_sq_ - sample_q_sq(r[1], rng) / par_q_sq_.slope[1];
    }
    else
    {
        // Sample \f$ Q^{2} \f$ for \f$ n + A \rightarrow n + A \f$
        constexpr real_type one_third = 1 / real_type(3);
        constexpr real_type one_fifth{0.2};
        constexpr real_type one_seventh = 1 / real_type(7);

        real_type const r[4]
            = {-std::expm1(-max_q_sq_
                           * (par_q_sq_.slope[0] + max_q_sq_ * par_q_sq_.ss)),
               -std::expm1(
                   -(heavy_target_ ? ipow<5>(max_q_sq_) : ipow<3>(max_q_sq_))
                   * par_q_sq_.slope[1]),
               -std::expm1(-(heavy_target_ ? ipow<7>(max_q_sq_) : max_q_sq_)
                           * par_q_sq_.slope[2]),
               -std::expm1(-max_q_sq_ * par_q_sq_.slope[3])};

        real_type mi[6] = {};
        for (auto i : range(4))
        {
            mi[i] = r[i] * par_q_sq_.expnt[i];
        }
        mi[4] = mi[0] + mi[1];
        mi[5] = mi[2] + mi[4];

        real_type rand = (mi[3] + mi[5]) * generate_canonical(rng);
        if (rand < mi[0])
        {
            real_type tss = 2 * par_q_sq_.ss;
            q_sq = this->sample_q_sq(r[0], rng) / par_q_sq_.slope[0];
            if (std::fabs(tss) > this->tolerance_slope_sq())
            {
                q_sq = (std::sqrt(par_q_sq_.slope[0]
                                  * (par_q_sq_.slope[0] + 2 * tss * q_sq))
                        - par_q_sq_.slope[0])
                       / tss;
            }
        }
        else if (rand < mi[4])
        {
            q_sq = clamp_to_nonneg(this->sample_q_sq(r[1], rng)
                                   / par_q_sq_.slope[1]);
            q_sq = std::pow(q_sq, heavy_target_ ? one_fifth : one_third);
        }
        else if (rand < mi[5])
        {
            q_sq = clamp_to_nonneg(this->sample_q_sq(r[2], rng)
                                   / par_q_sq_.slope[2]);
            if (heavy_target_)
            {
                q_sq = std::pow(q_sq, one_seventh);
            }
        }
        else
        {
            q_sq = this->sample_q_sq(r[3], rng) / par_q_sq_.slope[3];
            // u reduced for light-A (starts from 0)
            if (!heavy_target_)
            {
                q_sq = max_q_sq_ - q_sq;
            }
        }
    }
    return clamp(q_sq, real_type{0}, max_q_sq_) / ipow<2>(this->to_gev());
}

//---------------------------------------------------------------------------//
/*!
 * Returns the maximum \f$ -t = Q^{2} \f$ (value in \f$ clhep::GeV^{2} \f$) of
 * the elastic scattering between the incident neutron of which momentum is
 * \c p (value in clhep::GeV) and the target nucleus (A = Z + N) where Z > 0.
 *
 * For the neutron-nucleus scattering, the maximum momentum transfer is
 * \f$ Q_{max}^{2} = 4 M_{A}^{2} * p^{2}/(mds) \f$ where the Mandelstam mds
 * is \f$ mds = 2 M_{A} E_{neutron} + M_{neutron}^{2} + M_{A}^{2} \f$ (value
 * in clhep::GeV square. The the neutron-neutron channel, the maximum momentum
 * is \f$ Q^{2} = E_{neutron} * M_{neutron} - M_{neutron}^{2} \f$ when the
 * collision angle is 90 degree in the center of mass system, is currently
 * excluded, but may be supported if there is a user case.
 */
CELER_FUNCTION real_type MomentumTransferSampler::calc_max_q_sq(Momentum p) const
{
    // Momentum and mass square of the incident neutron
    real_type target_mass = value_as<Mass>(target_mass_);

    real_type p_sq = ipow<2>(value_as<units::MevMomentum>(p));
    real_type m_sq = ipow<2>(value_as<Mass>(neutron_mass_));
    real_type a_sq = ipow<2>(target_mass);

    // Return the maximum momentum transfer in the value of clhep::GeV square
    return 4 * a_sq * p_sq * ipow<2>(this->to_gev())
           / (2 * target_mass * std::sqrt(p_sq + m_sq) + m_sq + a_sq);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate parameters of the neutron-nucleus elastic scattering.
 *
 * \param neutron_p the neutron momentum in the lab frame (value in clhep::GeV
 * unit).
 */
CELER_FUNCTION
auto MomentumTransferSampler::calc_par_q_sq(Momentum neutron_p) const
    -> ExchangeParameters
{
    // ExchangeParameters
    real_type p = value_as<units::MevMomentum>(neutron_p) * this->to_gev();
    real_type lp = std::log(p);
    real_type sp = std::sqrt(p);
    real_type p2 = ipow<2>(p);
    real_type p3 = p2 * p;
    real_type p4 = p3 * p;

    ExchangeParameters result;

    if (amass_ == AtomicMassNumber{1})
    {
        // Special case for the \f$ n + p \rightarrow n + p \f$ channel
        real_type dl1 = lp - par_[3];

        constexpr real_type np_el[15] = {
            6.75,
            0.14,
            13,
            0.14,
            0.6,
            0.00013,
            75,
            0.001,
            7.2,
            4.32,
            0.012,
            2.5,
            12,
            0.34,
            18,
        };

        result.expnt[0] = (np_el[0] + np_el[1] * ipow<2>(dl1) + np_el[2] / p)
                              / (1 + np_el[3] / p4)
                          + np_el[4] / (p4 + np_el[5]);
        result.slope[0] = (np_el[8] + np_el[9] / (ipow<2>(p4) + np_el[10] * p3))
                          / (1 + np_el[11] / p4);
        result.expnt[1] = (np_el[6] + np_el[7] / p4 / p) / p3;
        result.slope[1] = np_el[12] / (p * sp + np_el[13]);
        result.ss = np_el[14];
    }
    else
    {
        real_type p5 = p4 * p;
        real_type p6 = p5 * p;
        real_type p8 = p6 * p2;
        real_type p10 = p8 * p2;
        real_type p12 = p10 * p2;
        real_type p16 = ipow<2>(p8);
        real_type dl = lp - real_type(5);

        if (!heavy_target_)
        {
            real_type pah = std::pow(p, real_type(0.5) * amass_.get());
            real_type pa = ipow<2>(pah);
            real_type pa2 = ipow<2>(pa);
            result.expnt[0] = par_[0] / (1 + par_[1] * p4 * pa)
                              + par_[2] / (p4 + par_[3] * p4 / pa2)
                              + (par_[4] * ipow<2>(dl) + par_[5])
                                    / (1 + par_[6] / p2);
            result.slope[0] = (par_[7] + par_[8] * p2) / (p4 + par_[9] / pah)
                              + par_[10];
            result.ss = par_[11] / (1 + par_[12] / p2)
                        + par_[13] / (p6 / pa + par_[14] / p16);
            result.expnt[1] = par_[15] / (pa / p2 + par_[16] / p4) + par_[17];
            result.slope[1] = par_[18] * std::pow(p, par_[19])
                              + par_[20] / (p8 + par_[21] / p16);
            result.expnt[2] = par_[22] / (pa * p + par_[23] / pa) + par_[24];
            result.slope[2] = par_[25] / (p3 + par_[26] / p6)
                              + par_[27] / (1 + par_[28] / p2);
            result.expnt[3]
                = p2
                  * (pah * par_[29] * std::exp(-pah * par_[30])
                     + par_[31] / (1 + par_[32] * std::pow(p, par_[33])));
            result.slope[3] = par_[34] * pa / p2 / (1 + pa * par_[35]);
        }
        else
        {
            result.expnt[0] = par_[0] / (1 + par_[1] / p4)
                              + par_[2] / (p4 + par_[3] / p2)
                              + par_[4] / (p5 + par_[5] / p16);
            result.slope[0] = (par_[6] / p8 + par_[10])
                                  / (p + par_[7] / std::pow(p, par_[11]))
                              + par_[8] / (1 + par_[9] / p4);
            result.ss = par_[12] / (p4 / std::pow(p, par_[14]) + par_[13] / p4);
            result.expnt[1] = par_[15] / p4
                                  / (std::pow(p, par_[16]) + par_[17] / p12)
                              + par_[18];
            result.slope[1] = par_[19] / std::pow(p, par_[20])
                              + par_[21] / std::pow(p, par_[22]);
            result.expnt[2] = par_[23] / std::pow(p, par_[26])
                                  / (1 + par_[27] / p12)
                              + par_[24] / (1. + par_[25] / p6);
            result.slope[2] = par_[28] / p8 + par_[29] / p2
                              + par_[30] / (1 + par_[31] / p8);
            result.expnt[3]
                = (par_[32] / p4 + par_[37] / p) / (1 + par_[33] / p10)
                  + (par_[34] + par_[35] * dl * dl) / (1 + par_[36] / p12);
            result.slope[3] = par_[38] / (1 + par_[39] / p)
                              + par_[40] * p4 / (1 + par_[41] * p5);
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
