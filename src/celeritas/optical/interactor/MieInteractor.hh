//---------------------------------*- C++
//-*----------------------------------//
// Copyright ...
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/MieInteractor.hh
//! \brief Sample optical Mie scattering (Henyey–Greenstein phase function)
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie scattering using the Henyey–Greenstein distribution.
 *
 * Henyey–Greenstein phase function:
 * \f[
 *   P(\cos\theta) \propto \frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^{3/2}}
 * \f]
 *
 * Parameters:
 * - forward_ratio: probability of using forward vs backward lobe
 * - forward_g, backward_g: HG asymmetry parameters for each lobe
 */
class MieInteractor
{
  public:
    struct Params
    {
        real_type forward_g;
        real_type backward_g;
        real_type forward_ratio;
    };

    inline CELER_FUNCTION MieInteractor(ParticleTrackView const& particle,
                                        Real3 const& direction,
                                        Params const& mie);

    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng) const;

  private:
    Real3 const& inc_dir_;  //!< Incident photon direction
    Real3 const& inc_pol_;  //!< Incident polarization
    Params mie_params_;  //!< Mie scattering params
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION
MieInteractor::MieInteractor(ParticleTrackView const& particle,
                             Real3 const& direction,
                             Params const& mie)
    : inc_dir_(direction), inc_pol_(particle.polarization()), mie_params_(mie)
{
    CELER_EXPECT(is_soft_unit_vector(inc_dir_));
    CELER_EXPECT(is_soft_unit_vector(inc_pol_));
    CELER_EXPECT(soft_zero(dot_product(inc_dir_, inc_pol_)));
}

//---------------------------------------------------------------------------//
/*!
 * Sample a single optical Mie scattering event.
 */
template<class Engine>
CELER_FUNCTION Interaction MieInteractor::operator()(Engine& rng) const
{
    Interaction result;
    Real3& new_dir = result.direction;
    Real3& new_pol = result.polarization;

    using UniformRealDist = UniformRealDistribution<real_type>;
    UniformRealDist sample_phi(0, real_type(2 * constants::pi));
    UniformRealDist sample_r(0, 1);
    // --- 1. Choose forward/backward lobe ---
    // real_type g;
    // real_type direction;
    // if (UniformRealDistribution<real_type>{0, 1}(rng) <
    // mie_params_.forward_ratio)
    //{
    //    g = mie_params_.forward_g;
    //    direction = 1;
    //}
    // else
    //{
    //    g = mie_params_.backward_g;
    //    direction = -1;
    //}
    // Select forward/backward g
    UniformRealDist sample_g(0, 1);
    real_type g = (sample_g(rng) < mie_params_.forward_ratio
                       ? mie_params_.forward_g
                       : mie_params_.backward_g);
    // --- 2. Sample cosθ from HG distribution ---
    real_type r = sample_r(rng);

    // real_type costheta = (g != 0)
    //         ? (1 / (2*g)) * (1 + g*g - ipow<2>((1 - g*g) / (1 - g + 2*g*r)))
    //         : (2*r - 1);
    real_type costheta = (g != 0)
                             ? 2 * r * (1 - g * g * r)
                                       * ipow<2>(1 + g / (1 - g + 2 * g * r))
                                   - 1
                             : (2 * r - 1);
    real_type phi = sample_phi(rng);

    // --- 3. Build new direction ---
    new_dir = from_spherical(costheta, phi);

    CELER_ENSURE(is_soft_unit_vector(result.direction));

    // --- 4. Build polarization (rejection sampling) ---
    SoftZero const soft_zero{SoftEqual{}.rel()};
    do
    {
        // Project old polarization onto plane perpendicular to new direction
        new_pol = make_unit_vector(make_orthogonal(inc_pol_, new_dir));

        // Rare degenerate case → reject
    } while (RejectionSampler{std::fabs(dot_product(new_pol, new_dir))}(rng));

    CELER_ENSURE(is_soft_unit_vector(new_pol));
    CELER_ENSURE(soft_zero(dot_product(new_pol, new_dir)));

    return result;
}

}  // namespace optical
}  // namespace celeritas
