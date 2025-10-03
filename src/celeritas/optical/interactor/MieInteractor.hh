//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/MieInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/MieData.hh"
#include "celeritas/optical/ParticleTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie scattering using the Henyey–Greenstein distribution.
 *\f[
 *   P(\cos\theta) \propto \frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^{3/2}}
 * \f].
 * Parameters:
 * - forward_ratio: probability of using forward vs backward lobe
 * - forward_g, backward_g: HG asymmetry parameters for each lobe
 */
class MieInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION MieInteractor(NativeCRef<MieData> const& shared,
                                        ParticleTrackView const& particle,
                                        Real3 const& direction,
                                        OptMatId const& mat_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng) const;

  private:
    Real3 const& inc_dir_;  //!< Incident photon direction
    Real3 const& inc_pol_;  //!< Incident polarization
    MieMaterialData const& mie_params_;  //!< Mie scattering params

    // Choose forward/backward scattering using random generator
    BernoulliDistribution sample_forward_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
MieInteractor::MieInteractor(NativeCRef<MieData> const& shared,
                             ParticleTrackView const& particle,
                             Real3 const& direction,
                             OptMatId const& mat_id)
    : inc_dir_(direction)
    , inc_pol_(particle.polarization())
    , mie_params_(shared.mie_record[mat_id])
    , sample_forward_(mie_params_.forward_g)
{
    CELER_EXPECT(shared);
    CELER_EXPECT(mat_id < shared.mie_record.size());
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
    UniformRealDist sample_r(0, 1);
    do
    {
        real_type r = sample_r(rng);
        bool is_forward = sample_forward_(rng);

        // Select forward or backward lobe
        real_type g
            = (is_forward ? mie_params_.forward_g : mie_params_.backward_g);

        // Compute cos(theta) distribution for optical photons that undergo mie
        // scattering
        // NOTE: floating point cancellation means the expression can result in
        // a cosine slightly greater than unity
        real_type costheta = 2 * r * ipow<2>((1 + g) / (1 - g + 2 * g * r))
                                 * (1 - g + g * r)
                             - 1;
        costheta = celeritas::min(costheta, real_type{1});

        CELER_ASSERT(costheta >= -1 && costheta <= 1);

        // Reverse cos theta if backward_g is chosen
        if (!is_forward)
            costheta = -costheta;

        // Sample azimuthal angle
        UniformRealDist sample_phi(0, real_type(2 * constants::pi));
        real_type phi = sample_phi(rng);

        // Creating new direction
        new_dir = from_spherical(costheta, phi);

        // Project polarization onto plane perpendicular to new direction
        new_pol = make_unit_vector(make_orthogonal(inc_pol_, new_dir));
    } while (CELER_UNLIKELY(!is_soft_orthogonal(new_pol, new_dir)));

    if (!BernoulliDistribution{0.5}(rng))
    {
        // Flip direction with 50% probability: there are two polarizations
        // perpendicular to the new direction and the original polarization
        new_pol = -new_pol;
    }

    CELER_ENSURE(is_soft_unit_vector(new_dir));
    CELER_ENSURE(is_soft_unit_vector(new_pol));
    CELER_ENSURE(soft_zero(dot_product(new_pol, new_dir)));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
