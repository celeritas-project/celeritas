//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/detail/InvariantQ2Sampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

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
class InvariantQ2Sampler
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
    inline CELER_FUNCTION InvariantQ2Sampler(NeutronElasticRef const& shared,
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
    AtomicMassNumber a_;
    bool heavy_target_{false};
    // Momentum magnitude of the incident neutron (GeV/c)
    Momentum neutron_p_;
    // Maximum momentum transfer for the elastic scattering
    real_type max_q2_;
    // Parameters for the CHIPS differential cross section
    ExchangeParameters par_q2_;

    //// HELPER FUNCTIONS ////

    // Sample the slope
    template<class Engine>
    inline CELER_FUNCTION real_type sample_q2(real_type radius, Engine& rng)
    {
        return -std::log(1 - radius * generate_canonical(rng));
    }

    // Calculate the maximum momentum transfer
    inline CELER_FUNCTION real_type calc_max_q2(Momentum p) const;

    // Calculate parameters used in the quark-exchange process
    inline CELER_FUNCTION ExchangeParameters calc_par_q2(Momentum p) const;

    //// COMMON PROPERTIES ////

    // Covert from clhep::MeV value to clhep::GeV value
    static CELER_CONSTEXPR_FUNCTION real_type to_gev() { return 1e-3; }

    // S-wave for neutron p < 14 MeV/c (kinetic energy < 0.1 MeV)
    static CELER_CONSTEXPR_FUNCTION Momentum s_wave_limit()
    {
        return Momentum{14};
    }
};

//---------------------------------------------------------------------------//
// Inline DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the incident momentum in the lab frame and the target(A).
 *
 * \note The incident neutron momentum, and neutron and nucleus masses are
 *  converted to the GeV value.
 */
CELER_FUNCTION
InvariantQ2Sampler::InvariantQ2Sampler(NeutronElasticRef const& shared,
                                       IsotopeView const& target,
                                       Momentum neutron_p)
    : par_(shared.diff_xs.coeffs[target.isotope_id()].par)
    , neutron_mass_(shared.neutron_mass)
    , target_mass_(target.nuclear_mass())
    , a_(target.atomic_mass_number())
    , heavy_target_(a_.get() > 6)
    , neutron_p_(neutron_p)
{
    max_q2_ = calc_max_q2(neutron_p_);
    par_q2_ = calc_par_q2(neutron_p_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the momentum transfer of the neutron-nucleus elastic scattering
 * based on G4ChipsElasticModel and G4ChipsNeutronElasticXS of the Geant4
 * 11.2 release.
 *
 */
template<class Engine>
CELER_FUNCTION auto InvariantQ2Sampler::operator()(Engine& rng) -> real_type
{
    // Sample \f$ Q^{2} \f$ below S-wave limit
    if (neutron_p_ < this->s_wave_limit())
    {
        return max_q2_ * generate_canonical(rng) / ipow<2>(this->to_gev());
    }

    // Sample \f$ Q^{2} \f$
    real_type q2 = 0;

    if (a_ == AtomicMassNumber{1})
    {
        // Special case for the \f$ n + p \rightarrow n + p \f$ channel
        real_type R1 = -std::expm1(-max_q2_ * par_q2_.slope[0]);
        real_type R2 = -std::expm1(-max_q2_ * par_q2_.slope[1]);

        real_type I1 = R1 * par_q2_.expnt[0];
        real_type I2 = R2 * par_q2_.expnt[1] / par_q2_.slope[1];

        // Sample by t-channel and u-channel (charge exchange)
        q2 = (BernoulliDistribution(I1 / (I1 + I2))(rng))
                 ? sample_q2(R1, rng) / par_q2_.slope[0]
                 : max_q2_ - sample_q2(R2, rng) / par_q2_.slope[1];
    }
    else
    {
        // Sample \f$ Q^{2} \f$ for \f$ n + A \rightarrow n + A \f$
        constexpr real_type one_third = 1 / real_type(3);
        constexpr real_type one_fifth{0.2};
        constexpr real_type one_seventh = 1 / real_type(7);

        real_type R1 = -std::expm1(
            -max_q2_ * (par_q2_.slope[0] + max_q2_ * par_q2_.ss));

        // power 3 for low-A and power 5 for high-A
        real_type factor = heavy_target_ ? ipow<5>(max_q2_) : ipow<3>(max_q2_);
        real_type R2 = -std::expm1(-factor * par_q2_.slope[1]);

        // power 1 for low-A and power 7 for high-A
        factor = heavy_target_ ? ipow<7>(max_q2_) : max_q2_;
        real_type R3 = -std::expm1(-factor * par_q2_.slope[2]);
        real_type R4 = -std::expm1(-max_q2_ * par_q2_.slope[3]);

        real_type I1 = R1 * par_q2_.expnt[0];
        real_type I2 = R2 * par_q2_.expnt[1];
        real_type I3 = R3 * par_q2_.expnt[2];
        real_type I4 = R4 * par_q2_.expnt[3];
        real_type I5 = I1 + I2;
        real_type I6 = I3 + I5;

        real_type rand = (I4 + I6) * generate_canonical(rng);
        if (rand < I1)
        {
            real_type tss = 2 * par_q2_.ss;
            q2 = this->sample_q2(R1, rng) / par_q2_.slope[0];
            if (std::fabs(tss) > 1e-7)
            {
                q2 = (std::sqrt(par_q2_.slope[0]
                                * (par_q2_.slope[0] + 2 * tss * q2))
                      - par_q2_.slope[0])
                     / tss;
            }
        }
        else if (rand < I5)
        {
            q2 = clamp_to_nonneg(this->sample_q2(R2, rng) / par_q2_.slope[1]);
            q2 = (heavy_target_) ? std::pow(q2, one_fifth)
                                 : std::pow(q2, one_third);
        }
        else if (rand < I6)
        {
            q2 = clamp_to_nonneg(this->sample_q2(R3, rng) / par_q2_.slope[2]);
            if (heavy_target_)
            {
                q2 = std::pow(q2, one_seventh);
            }
        }
        else
        {
            q2 = this->sample_q2(R4, rng) / par_q2_.slope[3];
            // u reduced for light-A (starts from 0)
            if (!heavy_target_)
            {
                q2 = max_q2_ - q2;
            }
        }
    }
    return clamp(q2, real_type{0}, max_q2_) / ipow<2>(this->to_gev());
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
CELER_FUNCTION real_type InvariantQ2Sampler::calc_max_q2(Momentum p) const
{
    // Momentum and mass square of the incident neutron
    real_type target_mass = value_as<Mass>(target_mass_);

    real_type p2 = ipow<2>(value_as<units::MevMomentum>(p));
    real_type m2 = ipow<2>(value_as<Mass>(neutron_mass_));
    real_type a2 = ipow<2>(target_mass);

    // Return the maximum momentum transfer in the value of clhep::GeV square
    return 4 * a2 * p2 * ipow<2>(this->to_gev())
           / (2 * target_mass * std::sqrt(p2 + m2) + m2 + a2);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate parameters of the neutron-nucleus elastic scattering.
 *
 * \param p the neutron momentum in the lab frame (value in clhep::GeV unit).
 */
CELER_FUNCTION
auto InvariantQ2Sampler::calc_par_q2(Momentum neutron_p) const
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

    if (a_ == AtomicMassNumber{1})
    {
        // Special case for the \f$ n + p \rightarrow n + p \f$ channel
        real_type dl1 = lp - par_[3];

        constexpr real_type np_el[15] = {6.75,
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
                                         18};

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
            real_type pah = std::pow(p, real_type(0.5) * a_.get());
            real_type pa = ipow<2>(pah);
            real_type pa2 = ipow<2>(pa);
            result.expnt[0] = par_[0] / (1. + par_[1] * p4 * pa)
                              + par_[2] / (p4 + par_[3] * p4 / pa2)
                              + (par_[4] * ipow<2>(dl) + par_[5])
                                    / (1. + par_[6] / p2);
            result.slope[0] = (par_[7] + par_[8] * p2) / (p4 + par_[9] / pah)
                              + par_[10];
            result.ss = par_[11] / (1. + par_[12] / p2)
                        + par_[13] / (p6 / pa + par_[14] / p16);
            result.expnt[1] = par_[15] / (pa / p2 + par_[16] / p4) + par_[17];
            result.slope[1] = par_[18] * std::pow(p, par_[19])
                              + par_[20] / (p8 + par_[21] / p16);
            result.expnt[2] = par_[22] / (pa * p + par_[23] / pa) + par_[24];
            result.slope[2] = par_[25] / (p3 + par_[26] / p6)
                              + par_[27] / (1. + par_[28] / p2);
            result.expnt[3]
                = p2
                  * (pah * par_[29] * std::exp(-pah * par_[30])
                     + par_[31] / (1. + par_[32] * std::pow(p, par_[33])));
            result.slope[3] = par_[34] * pa / p2 / (1. + pa * par_[35]);
        }
        else
        {
            result.expnt[0] = par_[0] / (1. + par_[1] / p4)
                              + par_[2] / (p4 + par_[3] / p2)
                              + par_[4] / (p5 + par_[5] / p16);
            result.slope[0] = (par_[6] / p8 + par_[10])
                                  / (p + par_[7] / std::pow(p, par_[11]))
                              + par_[8] / (1. + par_[9] / p4);
            result.ss = par_[12] / (p4 / std::pow(p, par_[14]) + par_[13] / p4);
            result.expnt[1] = par_[15] / p4
                                  / (std::pow(p, par_[16]) + par_[17] / p12)
                              + par_[18];
            result.slope[1] = par_[19] / std::pow(p, par_[20])
                              + par_[21] / std::pow(p, par_[22]);
            result.expnt[2] = par_[23] / std::pow(p, par_[26])
                                  / (1. + par_[27] / p12)
                              + par_[24] / (1. + par_[25] / p6);
            result.slope[2] = par_[28] / p8 + par_[29] / p2
                              + par_[30] / (1. + par_[31] / p8);
            result.expnt[3]
                = (par_[32] / p4 + par_[37] / p) / (1. + par_[33] / p10)
                  + (par_[34] + par_[35] * dl * dl) / (1. + par_[36] / p12);
            result.slope[3] = par_[38] / (1. + par_[39] / p)
                              + par_[40] * p4 / (1. + par_[41] * p5);
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas