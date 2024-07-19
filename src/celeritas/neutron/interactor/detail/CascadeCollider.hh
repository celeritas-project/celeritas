//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/detail/CascadeCollider.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/grid/TwodGridCalculator.hh"
#include "corecel/grid/TwodGridData.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/FourVector.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "CascadeParticle.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for the nucleon-nucleon intra-nucleus cascade collision.
 */
class CascadeCollider
{
  public:
    //!@{
    //! \name Type aliases
    using FinalState = Array<CascadeParticle, 2>;
    using MevMass = units::MevMass;
    //!@}

  public:
    // Construct with shared data and colliding particles
    inline CELER_FUNCTION CascadeCollider(NeutronInelasticRef const& shared,
                                          CascadeParticle const& bullet,
                                          CascadeParticle const& target);

    // Sample the final state of the two-body intra-nucleus cascade collision
    template<class Engine>
    inline CELER_FUNCTION FinalState operator()(Engine& rng);

  private:
    //// TYPES ////

    using Grid = NonuniformGrid<real_type>;
    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    // Shared constant data
    NeutronInelasticRef const& shared_;
    // Participating cascade particles
    CascadeParticle const& bullet_;
    CascadeParticle const& target_;

    // Id for intra-nuclear channels
    ChannelId ch_id_;
    // Booster vector in the center of mass frame
    Real3 cm_velocity_;
    // Momentum magnitude in the center of mass frame
    real_type cm_p_;
    // Kinetic energy in the target rest frame
    real_type kin_energy_;

    // Sampler
    UniformRealDist sample_phi_;

    //// CONSTANTS ////

    //! A criteria for coplanarity in the Lorentz transformation
    static CELER_CONSTEXPR_FUNCTION real_type epsilon()
    {
        return real_type{1e-10};
    }

    //// HELPER FUNCTIONS ////

    //! Calculate the momentum magnitude in the center of mass frame
    inline CELER_FUNCTION real_type calc_cm_p(FourVector const& v) const;
};

//---------------------------------------------------------------------------//
// Inline DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from share data and colliding nucleons in the lab frame.
 */
CELER_FUNCTION
CascadeCollider::CascadeCollider(NeutronInelasticRef const& shared,
                                 CascadeParticle const& bullet,
                                 CascadeParticle const& target)
    : shared_(shared)
    , bullet_(bullet)
    , target_(target)
    , ch_id_(ChannelId{
          static_cast<size_type>((bullet_.type == target_.type) ? 0 : 1)})
    , sample_phi_(0, 2 * constants::pi)
{
    // Initialize the boost velocity and momentum in the center of mass frame
    FourVector sum_four_vec = bullet_.four_vec + target_.four_vec;
    cm_velocity_ = boost_vector(sum_four_vec);
    cm_p_ = calc_cm_p(sum_four_vec);

    // Calculate the kinetic energy in the target rest frame
    FourVector bullet_p = bullet_.four_vec;
    boost(-boost_vector(target_.four_vec), &bullet_p);
    kin_energy_ = bullet_p.energy - norm(bullet_p);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the final state of outgoing particles from the two-body intra-nucleus
 * nucleon-nucleon collision in the center of mass (c.m.) frame and return
 * them after converting momentum to the lab frame. This performs the same
 * sampling routine as in Geant4's G4ElementaryParticleCollider, mainly
 * implemented in collider and generateSCMfinalState methods. The cos\theta
 * distribution in c.m. is inversely sampled using the tabulated cumulative
 * distribution function (c.d.f) data in the kinetic energy and cos\theta bins
 * which are implemented in G4CascadeFinalStateAlgorithm::GenerateTwoBody and
 * G4NumIntTwoBodyAngDst::GetCosTheta methods.
 */
template<class Engine>
CELER_FUNCTION auto CascadeCollider::operator()(Engine& rng) -> FinalState
{
    // Sample cos\theta of outgoing particles in the center of mass frame
    real_type cdf = generate_canonical(rng);
    TwodGridData cdf_grid = shared_.angular_cdf[ch_id_];
    Grid energy_grid(cdf_grid.x, shared_.reals);
    real_type cos_theta{0};

    if (kin_energy_ < energy_grid.back())
    {
        // Find cos\theta from tabulated angular data for a given c.d.f.
        Grid cos_grid(cdf_grid.y, shared_.reals);
        TwodGridCalculator calc_cdf(cdf_grid, shared_.reals);

        size_type idx = cos_grid.size() - 2;
        real_type cdf_upper = 0;
        real_type cdf_lower = 1;

        do
        {
            cdf_upper = cdf_lower;
            cdf_lower = calc_cdf({kin_energy_, cos_grid[idx]});
        } while (cdf_lower > cdf && idx-- > 0);

        real_type frac = (cdf - cdf_lower) / (cdf_upper - cdf_lower);
        cos_theta = fma(frac, cos_grid[idx + 1] - cos_grid[idx], cos_grid[idx]);
    }
    else
    {
        // Sample the angle outside tabulated data (unlikely)
        real_type slope = 2 * energy_grid.back() * ipow<2>(cm_p_);
        cos_theta = std::log(1 + cdf * std::expm1(2 * slope)) / slope - 1;
    }

    // Sample the momentum of outgoing particles in the center of mass frame
    Real3 mom = cm_p_ * from_spherical(cos_theta, sample_phi_(rng));

    // Rotate the momentum along the reference z-axis
    FourVector fv = {mom,
                     std::sqrt(dot_product(mom, mom)
                               + ipow<2>(value_as<MevMass>(bullet_.mass)))};

    // Find the final state of outgoing particles
    FinalState result = {bullet_, target_};

    FourVector cm_momentum = target_.four_vec;
    boost(-cm_velocity_, &cm_momentum);

    Real3 cm_dir = make_unit_vector(-cm_momentum.mom);
    real_type vel_parallel = dot_product(cm_velocity_, cm_dir);

    // Degenerated if velocity perpendicular to the c.m. momentum is small
    bool degenerated
        = (dot_product(cm_velocity_, cm_velocity_) - ipow<2>(vel_parallel)
           < this->epsilon());

    if (!degenerated)
    {
        Real3 vscm = make_unit_vector(cm_velocity_ - vel_parallel * cm_dir);
        Real3 vxcm = make_unit_vector(cross_product(cm_dir, cm_velocity_));
        if (norm(vscm) > this->epsilon() && norm(vxcm) > this->epsilon())
        {
            result[0].four_vec
                = {{fv.mom[0] * vscm + fv.mom[1] * vxcm + fv.mom[2] * cm_dir},
                   fv.energy};
        }
    }

    result[1].four_vec
        = {{-result[0].four_vec.mom},
           std::sqrt(dot_product(mom, mom)
                     + ipow<2>(value_as<MevMass>(target_.mass)))};

    // Convert the final state to the lab frame
    for (auto i : range(2))
    {
        boost(cm_velocity_, &result[i].four_vec);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the momentum magnitude of outgoing particles in the center of mass
 * (c.m.) frame. See Geant4's 4G4VHadDecayAlgorithm::TwoBodyMomentum method.
 */
CELER_FUNCTION real_type CascadeCollider::calc_cm_p(FourVector const& v) const
{
    // The total energy in c.m.
    real_type m0 = norm(v);

    real_type m1 = value_as<MevMass>(bullet_.mass);
    real_type m2 = value_as<MevMass>(target_.mass);

    real_type pc_squre = diffsq(m0, m1 - m2) * diffsq(m0, m1 + m2);

    return std::sqrt(clamp_to_nonneg(pc_squre)) / (2 * m0);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
