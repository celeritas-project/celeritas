//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/optical/MaterialView.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for calculating quantities for Fresnel equations.
 *
 * Incident waves are decomposed into transverse-electric (TE) and
 * transverse-magnetic (TM) polarizations. The interaction plane is defined as
 * the span of the photon direction and the surface normal. The TE (TM)
 * polarization has the electric (magnetic) field polarization transverse to
 * the interaction plane. If the direction is parallel to the surface normal,
 * then interaction plane is degenerate and the incident photon is defined to
 * be entirely in the TE polarization.
 *
 * This calculator helps handle this degenerate case, and also handles
 * calculating reflectivity and transmission in the total internal reflection
 * case.
 *
 * Convention follows \cite{fowles-optics-1975} section 2.6, except that
 * photon phase is not tracked in Celeritas.
 */
class FresnelCalculator
{
  public:
    // Construct from initial state
    explicit inline CELER_FUNCTION
    FresnelCalculator(Real3 const& direction,
                      Real3 const& polarization,
                      Real3 const& normal,
                      real_type relative_r_index);

    // Construct from track views
    explicit inline CELER_FUNCTION
    FresnelCalculator(Real3 const& inc_direction,
                      ParticleTrackView const& photon,
                      Real3 const& normal,
                      MaterialView const& pre_material,
                      MaterialView const& post_material);

    // Whether the interaction will be total internal reflection
    inline CELER_FUNCTION bool is_total_internal_reflection() const;

    // Calculate total reflectivity
    inline CELER_FUNCTION real_type calc_reflectivity() const;

    // Calculate interaction for refracted wave
    inline CELER_FUNCTION SurfaceInteraction refracted_interaction() const;

  private:
    // Get refracted photon direction
    inline CELER_FUNCTION Real3 refracted_direction() const;

    // Calculate transmission coefficients
    inline CELER_FUNCTION real_type calc_transmission_te() const;
    inline CELER_FUNCTION real_type calc_transmission_tm() const;

    // Calculate reflectivity coefficients
    inline CELER_FUNCTION real_type calc_reflectivity_te() const;
    inline CELER_FUNCTION real_type calc_reflectivity_tm() const;

    // Polarization axes
    inline CELER_FUNCTION Real3 const& te_axis() const;
    inline CELER_FUNCTION Real3 tm_axis(Real3 const& direction) const;

    // Incident photon polarization components
    inline CELER_FUNCTION real_type inc_te_component() const;
    inline CELER_FUNCTION real_type inc_tm_component() const;

    Real3 const& direction_;
    Real3 const& polarization_;
    Real3 const& normal_;
    real_type relative_r_index_;

    real_type cos_theta_;
    real_type cosine_ratio_;
    Real3 p_axis_;

    // Helper function for calculating reflectivity
    inline CELER_FUNCTION real_type reflectivity_ratio(real_type x) const;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

inline CELER_FUNCTION real_type
calc_relative_r_index(units::MevEnergy energy,
                      MaterialView const& pre_material,
                      MaterialView const& post_material)
{
    auto calc_r = [energy](MaterialView const& mat) {
        return mat.make_refractive_index_calculator()(
            value_as<units::MevEnergy>(energy));
    };

    return calc_r(post_material) / calc_r(pre_material);
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct calculator from initial photon and surface physics data.
 */
CELER_FUNCTION
FresnelCalculator::FresnelCalculator(Real3 const& direction,
                                     Real3 const& polarization,
                                     Real3 const& normal,
                                     real_type relative_r_index)
    : direction_(direction)
    , polarization_(polarization)
    , normal_(normal)
    , relative_r_index_(relative_r_index)
{
    CELER_EXPECT(is_soft_unit_vector(direction_));
    CELER_EXPECT(is_soft_unit_vector(polarization_));
    CELER_EXPECT(is_soft_unit_vector(normal_));
    CELER_EXPECT(relative_r_index > 0);
    CELER_EXPECT(is_entering_surface(direction_, normal));

    // Sometimes dot product of normalized parallel vectors is the next
    // representation after 1. Round down to exactly 1 to avoid errors.
    CELER_EXPECT(-dot_product(direction_, normal_)
                 <= std::nextafter(real_type{1}, real_type{2}));
    cos_theta_ = min(-dot_product(direction_, normal_), real_type{1});

    // Snell's law
    real_type sin_phi = sqrt(1 - ipow<2>(cos_theta_)) / relative_r_index_;

    // If undergoing total internal reflection, set the cosine ratio to exactly
    // zero. This gives the correct reflectivity and transmission coefficients.
    cosine_ratio_ = sin_phi >= 1 ? 0 : sqrt(1 - ipow<2>(sin_phi)) / cos_theta_;

    Real3 s_axis = make_orthogonal(direction_, normal);
    s_axis = soft_zero(norm(s_axis)) ? cross_product(polarization_, normal)
                                     : make_unit_vector(s_axis);
    p_axis_ = cross_product(normal_, s_axis);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from track views.
 */
CELER_FUNCTION
FresnelCalculator::FresnelCalculator(Real3 const& inc_direction,
                                     ParticleTrackView const& photon,
                                     Real3 const& normal,
                                     MaterialView const& pre_material,
                                     MaterialView const& post_material)
    : FresnelCalculator(
          inc_direction,
          photon.polarization(),
          normal,
          calc_relative_r_index(photon.energy(), pre_material, post_material))
{
}

//---------------------------------------------------------------------------//
/*!
 * Whether the photon is subject to total internal reflection.
 */
CELER_FUNCTION bool FresnelCalculator::is_total_internal_reflection() const
{
    // In the constructor, the cosine ratio is set to exactly zero for total
    // internal reflection.
    return cosine_ratio_ == 0;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate total reflectivity for the incident photon.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_reflectivity() const
{
    real_type te_comp_sq = ipow<2>(this->inc_te_component());
    real_type tm_comp_sq = ipow<2>(this->inc_tm_component());
    real_type total_reflectivity
        = (te_comp_sq * ipow<2>(this->calc_reflectivity_te())
           + tm_comp_sq * ipow<2>(this->calc_reflectivity_tm()))
          / (te_comp_sq + tm_comp_sq);

    CELER_ENSURE(0 <= total_reflectivity && total_reflectivity <= 1);

    return total_reflectivity;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate interaction for refracted wave.
 */
CELER_FUNCTION SurfaceInteraction FresnelCalculator::refracted_interaction() const
{
    CELER_ASSERT(!this->is_total_internal_reflection());

    SurfaceInteraction result;
    result.action = SurfaceInteraction::Action::refracted;
    result.direction = this->refracted_direction();

    result.polarization = {0, 0, 0};
    axpy(this->calc_transmission_te() * this->inc_te_component(),
         this->te_axis(),
         &result.polarization);
    axpy(this->calc_transmission_tm() * this->inc_tm_component(),
         this->tm_axis(result.direction),
         &result.polarization);
    result.polarization = make_unit_vector(result.polarization);

    CELER_ENSURE(result.is_valid());

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate direction of refracted photon.
 */
CELER_FUNCTION Real3 FresnelCalculator::refracted_direction() const
{
    CELER_EXPECT(!this->is_total_internal_reflection());
    Real3 dir = direction_;
    axpy(cos_theta_ * (1 - relative_r_index_ * cosine_ratio_), normal_, &dir);
    return make_unit_vector(dir);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate transmission coefficient of the TE component.
 *
 * Derived from equation 2.52 of \cite{fowles-optics-1975}.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_transmission_te() const
{
    return this->calc_reflectivity_te() + 1;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate transmission coefficient of the TM component.
 *
 * Derived from equation 2.53 of \cite{fowles-optics-1975}.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_transmission_tm() const
{
    return relative_r_index_ * (this->calc_reflectivity_tm() + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate reflectivity coefficient of the TE component.
 *
 * Equivalent to equation 2.54 of \cite{fowles-optics-1975}.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_reflectivity_te() const
{
    return -this->reflectivity_ratio(cosine_ratio_ * relative_r_index_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate reflectivity coefficient of the TM component.
 *
 * Equivalent to equation 2.55 of \cite{fowles-optics-1975}.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_reflectivity_tm() const
{
    return this->reflectivity_ratio(cosine_ratio_ / relative_r_index_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the polarization vector for the TE component.
 */
CELER_FUNCTION Real3 const& FresnelCalculator::te_axis() const
{
    return p_axis_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the polarization vector for the TM component based on the given
 * direction.
 */
CELER_FUNCTION Real3 FresnelCalculator::tm_axis(Real3 const& direction) const
{
    return cross_product(this->te_axis(), direction);
}

//---------------------------------------------------------------------------//
/*!
 * Get the incident photon TE polarization component.
 */
CELER_FUNCTION real_type FresnelCalculator::inc_te_component() const
{
    return dot_product(polarization_, this->te_axis());
}

//---------------------------------------------------------------------------//
/*!
 * Get the incident photon TM polarization component.
 */
CELER_FUNCTION real_type FresnelCalculator::inc_tm_component() const
{
    return dot_product(polarization_, this->tm_axis(direction_));
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for calculating reflectivity coefficients.
 */
CELER_FUNCTION real_type FresnelCalculator::reflectivity_ratio(real_type x) const
{
    return (x - 1) / (x + 1);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
