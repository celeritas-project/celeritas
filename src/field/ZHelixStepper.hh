//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ZHelixStepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

#include "Types.hh"
#include "UniformZMagField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The Helix (Test) Stepper.
 *
 * The analytical solution for the motion of a charged particle in a uniform
 * magnetic field along the z-direction.
 */
template<class EquationT>
class ZHelixStepper
{
    // Check whether the field_type used in EquationT is UniformZMagField
    static_assert(
        std::is_same<typename EquationT::field_type, UniformZMagField>::value,
        "ZHelix stepper only works with UniformZMagField");

  public:
    //!@{
    //! Type aliases
    using Result = StepperResult;
    //!@}

  public:
    // Construct with the equation of motion
    CELER_FUNCTION
    ZHelixStepper(const EquationT& eq) : equation_(eq) {}

    // Adaptive step size control
    CELER_FUNCTION auto operator()(real_type step, const OdeState& beg_state)
        -> Result;

  private:
    //// DATA ////

    // Equation of the motion
    const EquationT& equation_;

    //// HELPER TYPES ////
    enum class Helicity : bool
    {
        positive,
        negative
    };

    //// HELPER FUNCTIONS ////

    // Analytical solution for a given step along a helix trajectory
    CELER_FUNCTION OdeState move(real_type       step,
                                 real_type       radius,
                                 Helicity        helicity,
                                 const OdeState& beg_state,
                                 const OdeState& rhs);

    //// COMMON PROPERTIES ////

    static CELER_CONSTEXPR_FUNCTION real_type tolerance() { return 1e-10; }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * An explicit helix stepper with analytical solutions at the end and the
 * middle point for a given step.
 *
 * Assuming that the magnetic field is uniform and chosen to be parallel to
 * the z-axis, \f$B = (0, 0, B_z)\f$, without loss of generality, the motion of
 * a charged particle is described by a helix trajectory. For this algorithm,
 * the radius of the helix, \f$R = m gamma v/(qB)\f$ and the helicity, defined
 * as \f$ -sign(q B_z)\f$ are evaluated through the right hand side of the ODE
 * equation where q is the charge of the particle.
 */
template<class E>
CELER_FUNCTION auto
ZHelixStepper<E>::operator()(real_type step, const OdeState& beg_state)
    -> Result
{
    Result result;

    // Evaluate the right hand side of the equation
    OdeState rhs = equation_(beg_state);

    // Calculate the radius of the helix
    real_type radius = std::sqrt(dot_product(beg_state.mom, beg_state.mom)
                                 - ipow<2>(beg_state.mom[2]))
                       / norm(rhs.mom);

    // Set the helicity: 1(-1) for negative(positive) charge with Bz > 0
    Helicity helicity = Helicity(rhs.mom[0] / rhs.pos[1] > 0);

    // State after the half step
    result.mid_state
        = this->move(real_type(0.5) * step, radius, helicity, beg_state, rhs);

    // State after the full step
    result.end_state = this->move(step, radius, helicity, beg_state, rhs);

    // Solution are exact, but assign a tolerance for numerical treatments
    for (auto i : range(3))
    {
        result.err_state.pos[i] += ZHelixStepper::tolerance();
        result.err_state.mom[i] += ZHelixStepper::tolerance();
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Integration for a given step length on a helix.
 *
 * Equations of a charged particle motion in a uniform magnetic field,
 * \f$B(0, 0, B_z)\f$ along the curved trajectory \f$ds = v dt\f$ are
 * \f[
 *  d^2 x/ds^2 =  q/p (dy/ds) B_z
 *  d^2 y/ds^2 = -q/p (dx/ds) B_z
 *  d^2 z/ds^2 =  0
 * \f]
 * where \em q and \em p are the charge and the absolute momentum of the
 * particle, respectively. Since the motion in the perpendicular plane with
 * respected to the to the magnetic field is circular with a constant
 * \f$p_{\perp}\f$, the final ODE state of the perpendicular motion on the
 * circle for a given step length \em s is
 * \f[
 *  (x, y) = M(\phi) (x_0, y_0)^T
 *  (px, py) = M(\phi) (px_0, py_0)^T
 * \f]
 * where \f$\phi = s/R\f$ is the azimuth angle of the particle position between
 * the start and the end position and \f$M(\phi)\f$ is the rotational matrix.
 * The solution for the parallel direction along the field is trivial.
 */
template<class E>
CELER_FUNCTION OdeState ZHelixStepper<E>::move(real_type       step,
                                               real_type       radius,
                                               Helicity        helicity,
                                               const OdeState& beg_state,
                                               const OdeState& rhs)
{
    OdeState end_state;

    // Solution for position and momentum after moving delta_phi on the helix
    real_type del_phi = (helicity == Helicity::positive) ? step / radius
                                                         : -step / radius;
    real_type sin_phi = std::sin(del_phi);
    real_type cos_phi = std::cos(del_phi);

    end_state.pos = {(beg_state.pos[0] * cos_phi - beg_state.pos[1] * sin_phi),
                     (beg_state.pos[0] * sin_phi + beg_state.pos[1] * cos_phi),
                     beg_state.pos[2] + del_phi * radius * rhs.pos[2]};

    end_state.mom = {rhs.pos[0] * cos_phi - rhs.pos[1] * sin_phi,
                     rhs.pos[0] * sin_phi + rhs.pos[1] * cos_phi,
                     rhs.pos[2]};

    real_type momentum = norm(beg_state.mom);
    for (auto i : range(3))
    {
        end_state.mom[i] *= momentum;
    }

    return end_state;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
