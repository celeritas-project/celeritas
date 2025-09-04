#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"

namespace celeritas
{
namespace optical
{

class MieInteractor
{
  public:
    struct Params
    {
        real_type forward_g;
        real_type backward_g;
        real_type forward_ratio;
    };

    CELER_FUNCTION
    MieInteractor(ParticleTrackView const& particle,
                  Real3 const& direction,
                  Params const& params);

    template<class Engine>
    CELER_FUNCTION Interaction operator()(Engine& rng) const;

  private:
    Real3 const& inc_dir_;
    Real3 const& inc_pol_;
    Params params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION
MieInteractor::MieInteractor(ParticleTrackView const& particle,
                             Real3 const& direction,
                             Params const& params)
    : inc_dir_(direction), inc_pol_(particle.polarization()), params_(params)
{
    CELER_EXPECT(is_soft_unit_vector(inc_dir_));
    CELER_EXPECT(is_soft_unit_vector(inc_pol_));
}

//---------------------------------------------------------------------------//
template<class Engine>
CELER_FUNCTION Interaction MieInteractor::operator()(Engine& rng) const
{
    Interaction result;
    Real3& new_dir = result.direction;
    Real3& new_pol = result.polarization;

    // Pick forward vs backward branch
    real_type g;
    if (UniformRealDistribution<real_type>{0, 1}(rng) <= params_.forward_ratio)
        g = params_.forward_g;
    else
        g = params_.backward_g;

    // Sample scattering angle theta (HG distribution)
    real_type r = UniformRealDistribution<real_type>{0, 1}(rng);
    real_type cos_theta;
    if (std::abs(g) < 1e-12)
        cos_theta = 2 * r - 1;
    else
        cos_theta
            = (1.0 / (2 * g))
              * (1 + g * g - std::pow((1 - g * g) / (1 - g + 2 * g * r), 2));

    real_type phi = UniformRealDistribution<real_type>{0, 2 * pi}(rng);

    // Convert to vector
    new_dir = {std::sqrt(1 - cos_theta * cos_theta) * std::cos(phi),
               std::sqrt(1 - cos_theta * cos_theta) * std::sin(phi),
               cos_theta};
    new_dir.rotateUz(inc_dir_);
    new_dir = make_unit_vector(new_dir);

    // Polarization: pick perpendicular direction
    new_pol = make_unit_vector(make_orthogonal(inc_pol_, new_dir));

    if (UniformRealDistribution<real_type>{0, 1}(rng) < 0.5)
        new_pol = -new_pol;

    return result;
}

}  // namespace optical
}  // namespace celeritas
