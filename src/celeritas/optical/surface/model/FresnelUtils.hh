//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for calculating quantities for Fresnel equations.
 *
 * Incident waves are decomposed into transverse-electric (TE) and
 * transverse-magnetic (TM) modes. The interaction plane is defined as the span
 * of the photon direction and the surface normal. The TE (TM) mode has the
 * electric (magnetic) field polarization transverse to the interaction plane.
 * If the direction is parallel to the surface normal, then interaction plane
 * is degenerate and the incident photon is defined to be entirely in the TE
 * mode.
 *
 * This calculator helps handle this degenerate case, and also handles
 * calculating reflectivity and transmission in the total internal reflection
 * case.
 */
class FresnelCalculator
{
  public:
    // Construct from initial state
    CELER_FUNCTION
    FresnelCalculator(PhotonPhasor const& inc_photon,
                      Real3 const& normal,
                      real_type relative_r_index);

    // Whether the interaction will be total internal reflection
    CELER_FUNCTION bool is_total_internal_reflection() const;

    // Get refracted photon direction
    CELER_FUNCTION Real3 refracted_direction() const;

    // Calculate transmission coefficients
    CELER_FUNCTION real_type calc_transmission_te() const;
    CELER_FUNCTION real_type calc_transmission_tm() const;

    // Calculate reflectivity coefficients
    CELER_FUNCTION real_type calc_reflectivity_te() const;
    CELER_FUNCTION real_type calc_reflectivity_tm() const;

    // Polarization axes
    CELER_FUNCTION Real3 const& te_axis() const;
    CELER_FUNCTION Real3 tm_axis(Real3 const& direction) const;

    // Incident photon polarization components
    CELER_FUNCTION real_type inc_te_component() const;
    CELER_FUNCTION real_type inc_tm_component() const;

  private:
    PhotonPhasor const& inc_photon_;
    Real3 const& normal_;
    real_type relative_r_index_;

    real_type cos_theta_;
    real_type cosine_ratio_;
    Real3 p_axis_;

    // Helper function for calculating reflectivity
    CELER_FUNCTION real_type reflectivity_ratio(real_type x) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct calculator from initial photon and surface physics data.
 */
CELER_FUNCTION
FresnelCalculator::FresnelCalculator(PhotonPhasor const& inc_photon,
                                     Real3 const& normal,
                                     real_type relative_r_index)
    : inc_photon_(inc_photon)
    , normal_(normal)
    , relative_r_index_(relative_r_index)
{
    CELER_EXPECT(inc_photon.is_valid());
    CELER_EXPECT(is_soft_unit_vector(normal));
    CELER_EXPECT(relative_r_index > 0);
    CELER_EXPECT(is_entering_surface(inc_photon.direction, normal));

    cos_theta_ = clamp(-dot_product(inc_photon_.direction, normal_),
                       real_type{0},
                       real_type{1});

    real_type sin_phi = sqrt(1 - ipow<2>(cos_theta_)) / relative_r_index_;

    // If undergoing total internal reflection, set the cosine ratio to exactly
    // zero. This gives the correct reflectivity and transmission coefficients.
    cosine_ratio_ = sin_phi >= 1 ? 0 : sqrt(1 - ipow<2>(sin_phi)) / cos_theta_;

    Real3 s_axis = make_orthogonal(inc_photon.direction, normal);
    s_axis = soft_zero(norm(s_axis))
                 ? cross_product(inc_photon.polarization, normal)
                 : make_unit_vector(s_axis);
    p_axis_ = cross_product(normal_, s_axis);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate direction of refracted photon.
 */
CELER_FUNCTION Real3 FresnelCalculator::refracted_direction() const
{
    CELER_EXPECT(!this->is_total_internal_reflection());
    Real3 dir = inc_photon_.direction;
    axpy(cos_theta_ * (1 - relative_r_index_ * cosine_ratio_), normal_, &dir);
    return make_unit_vector(dir);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate transmission coefficient of the TE component.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_transmission_te() const
{
    return this->calc_reflectivity_te() + 1;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate transmission coefficient of the TM component.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_transmission_tm() const
{
    return relative_r_index_ * (this->calc_reflectivity_tm() + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate reflectivity coefficient of the TE component.
 */
CELER_FUNCTION real_type FresnelCalculator::calc_reflectivity_te() const
{
    return this->reflectivity_ratio(cosine_ratio_ * relative_r_index_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate reflectivity coefficient of the TM component.
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
    return dot_product(inc_photon_.polarization, this->te_axis());
}

//---------------------------------------------------------------------------//
/*!
 * Get the incident photon TM polarization component.
 */
CELER_FUNCTION real_type FresnelCalculator::inc_tm_component() const
{
    return dot_product(inc_photon_.polarization,
                       this->tm_axis(inc_photon_.direction));
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
 * Helper function for calculating reflectivity coefficients.
 */
CELER_FUNCTION real_type FresnelCalculator::reflectivity_ratio(real_type x) const
{
    return (x - 1) / (x + 1);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
