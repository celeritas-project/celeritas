//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/InverseCdfFinder.hh"
#include "celeritas/phys/FourVector.hh"

#include "CascadeParticle.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample final state for a nucleon-nucleon intra-nucleus cascade collision.
 *
 * This samples the final state of outgoing particles from the two-body
 * intra-nucleus nucleon-nucleon collision in the center of mass (c.m.) frame
 * and returns them after converting momentum to the lab frame. It performs the
 * same sampling routine as in Geant4's \c G4ElementaryParticleCollider, mainly
 * implemented in collide and generateSCMfinalState methods. The
 * \f$ \cos\theta \f$ distribution in c.m. is inversely sampled using the
 * tabulated cumulative distribution function (c.d.f) data in the kinetic
 * energy and the cosine bins which are implemented in \c
 * G4CascadeFinalStateAlgorithm::GenerateTwoBody and
 * \c  G4NumIntTwoBodyAngDst::GetCosTheta methods.
 */
class CascadeCollider
{
  public:
    //!@{
    //! \name Type aliases
    using FinalState = Array<CascadeParticle, 2>;
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

    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
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
    // Boost vector in the center of mass frame [1/c]
    Real3 cm_velocity_;
    // Momentum magnitude in the center of mass frame [MeV/c]
    real_type cm_p_;
    // Kinetic energy in the target rest frame [MeV]
    real_type kin_energy_;

    // Sampler
    UniformRealDist sample_phi_;

    //// CONSTANTS ////

    //! A criteria [1/c] for coplanarity in the Lorentz transformation
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
    , sample_phi_(0, static_cast<real_type>(2 * constants::pi))
{
    // Initialize the boost velocity and momentum in the center of mass frame
    FourVector sum_four_vec = bullet_.four_vec + target_.four_vec;
    cm_velocity_ = boost_vector(sum_four_vec);
    cm_p_ = this->calc_cm_p(sum_four_vec);

    // Calculate the kinetic energy in the target rest frame
    FourVector bullet_p = bullet_.four_vec;
    boost(-boost_vector(target_.four_vec), &bullet_p);
    kin_energy_ = bullet_p.energy - norm(bullet_p);
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the given RNG.
 */
template<class Engine>
CELER_FUNCTION auto CascadeCollider::operator()(Engine& rng) -> FinalState
{
    // Sample cos\theta of outgoing particles in the center of mass frame
    real_type cdf = generate_canonical(rng);
    TwodGridData const& cdf_grid = shared_.angular_cdf[ch_id_];
    Grid energy_grid(cdf_grid.x, shared_.reals);
    real_type cos_theta{0};

    if (kin_energy_ < energy_grid.back())
    {
        // Find cos\theta from tabulated angular data for a given CDF
        cos_theta = InverseCdfFinder(
            Grid(cdf_grid.y, shared_.reals),
            TwodGridCalculator(cdf_grid, shared_.reals)(kin_energy_))(cdf);
    }
    else
    {
        // Sample the angle outside tabulated data (unlikely)
        real_type slope = 2 * energy_grid.back() * ipow<2>(cm_p_);
        cos_theta = std::log(1 + cdf * std::expm1(2 * slope)) / slope - 1;
    }

    // Sample the momentum of outgoing particles in the center of mass frame
    // Rotate the momentum along the reference z-axis
    auto fv = FourVector::from_mass_momentum(
        bullet_.mass,
        Momentum{cm_p_},
        from_spherical(cos_theta, sample_phi_(rng)));

    // Find the final state of outgoing particles
    FinalState result = {bullet_, target_};

    FourVector cm_momentum = target_.four_vec;
    boost(-cm_velocity_, &cm_momentum);

    Real3 cm_dir = make_unit_vector(-cm_momentum.mom);
    real_type vel_parallel = dot_product(cm_velocity_, cm_dir);

    Real3 vscm = cm_velocity_;
    axpy(-vel_parallel, cm_dir, &vscm);

    if (norm(vscm) > this->epsilon())
    {
        vscm = make_unit_vector(vscm);
        Real3 vxcm = make_unit_vector(cross_product(cm_dir, cm_velocity_));
        if (norm(vxcm) > this->epsilon())
        {
            for (int i = 0; i < 3; ++i)
            {
                result[0].four_vec.mom[i] = fv.mom[0] * vscm[i]
                                            + fv.mom[1] * vxcm[i]
                                            + fv.mom[2] * cm_dir[i];
            }
            result[0].four_vec.energy = fv.energy;
        }
    }
    else
    {
        // Degenerate if velocity perpendicular to the c.m. momentum is small
        result[0].four_vec = fv;
    }

    result[1].four_vec = {{-result[0].four_vec.mom},
                          hypot(cm_p_, value_as<Mass>(target_.mass))};

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
 * (c.m.) frame. See Geant4's G4VHadDecayAlgorithm::TwoBodyMomentum method.
 */
CELER_FUNCTION real_type CascadeCollider::calc_cm_p(FourVector const& v) const
{
    // The total energy in c.m.
    real_type m0 = norm(v);

    real_type m1 = value_as<Mass>(bullet_.mass);
    real_type m2 = value_as<Mass>(target_.mass);

    real_type pc_sq = diffsq(m0, m1 - m2) * diffsq(m0, m1 + m2);

    return std::sqrt(clamp_to_nonneg(pc_sq)) / (2 * m0);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
