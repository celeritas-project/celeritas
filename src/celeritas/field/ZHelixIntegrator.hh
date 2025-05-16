//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/ZHelixIntegrator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "Types.hh"

namespace celeritas
{
class UniformZField;
//---------------------------------------------------------------------------//
/*!
 * Analytically step along a helical path for a uniform Z magnetic field.
 *
 * Given a uniform magnetic field along the *z* axis, \f$B = (0, 0, B_z)\f$,
 * the motion of a charged particle is described by a helix trajectory.
 * For this algorithm, the radius of the helix, \f$R = \frac{m v}{q B_z}\f$ and
 * the helicity, defined as \f$ -\sgn(q B_z)\f$, are evaluated through the
 * right hand side of the ODE equation where \f$ q \f$ is the charge of the
 * particle.
 *
 * The midpoint and endpoint states are calculated analytically.
 */
template<class EquationT>
class ZHelixIntegrator
{
    static_assert(
        std::is_same<std::remove_cv_t<std::remove_reference_t<
                         typename std::remove_reference_t<EquationT>::Field_t>>,
                     UniformZField>::value,
        "ZHelix stepper only works with UniformZField");

  public:
    //!@{
    //! \name Type aliases
    using result_type = FieldIntegration;
    //!@}

  public:
    //! Construct with the equation of motion
    explicit CELER_FUNCTION ZHelixIntegrator(EquationT&& eq)
        : calc_rhs_(::celeritas::forward<EquationT>(eq))
    {
    }

    // Adaptive step size control
    CELER_FUNCTION auto
    operator()(real_type step, OdeState const& beg_state) const -> result_type;

  private:
    //// DATA ////

    // Evaluate the equation of the motion
    EquationT calc_rhs_;

    //// HELPER TYPES ////
    enum class Helicity : bool
    {
        positive,
        negative
    };

    //// HELPER FUNCTIONS ////

    // Analytical solution for a given step along a helix trajectory
    CELER_FUNCTION OdeState move(real_type step,
                                 real_type radius,
                                 Helicity helicity,
                                 OdeState const& beg_state,
                                 OdeState const& rhs) const;

    //// COMMON PROPERTIES ////

    static CELER_CONSTEXPR_FUNCTION real_type tolerance()
    {
        if constexpr (std::is_same_v<real_type, double>)
            return 1e-10;
        else if constexpr (std::is_same_v<real_type, float>)
            return 1e-5f;
    }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EquationT>
CELER_FUNCTION ZHelixIntegrator(EquationT&&) -> ZHelixIntegrator<EquationT>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
template<class E>
CELER_FUNCTION auto
ZHelixIntegrator<E>::operator()(real_type step, OdeState const& beg_state) const
    -> result_type
{
    result_type result;

    // Evaluate the right hand side of the equation
    OdeState rhs = calc_rhs_(beg_state);

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

    // Solutions are exact, but assign a tolerance for numerical treatments
    result.err_state.pos.fill(ZHelixIntegrator::tolerance());
    result.err_state.mom.fill(ZHelixIntegrator::tolerance());

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
CELER_FUNCTION OdeState ZHelixIntegrator<E>::move(real_type step,
                                                  real_type radius,
                                                  Helicity helicity,
                                                  OdeState const& beg_state,
                                                  OdeState const& rhs) const
{
    // Solution for position and momentum after moving delta_phi on the helix
    real_type del_phi = (helicity == Helicity::positive) ? step / radius
                                                         : -step / radius;
    real_type sin_phi = std::sin(del_phi);
    real_type cos_phi = std::cos(del_phi);

    OdeState end_state;
    end_state.pos = {(beg_state.pos[0] * cos_phi - beg_state.pos[1] * sin_phi),
                     (beg_state.pos[0] * sin_phi + beg_state.pos[1] * cos_phi),
                     beg_state.pos[2] + del_phi * radius * rhs.pos[2]};

    end_state.mom = {rhs.pos[0] * cos_phi - rhs.pos[1] * sin_phi,
                     rhs.pos[0] * sin_phi + rhs.pos[1] * cos_phi,
                     rhs.pos[2]};

    real_type momentum = norm(beg_state.mom);
    for (int i = 0; i < 3; ++i)
    {
        end_state.mom[i] *= momentum;
    }

    return end_state;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
