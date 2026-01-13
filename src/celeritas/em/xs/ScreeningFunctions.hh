//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/ScreeningFunctions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/UnitUtils.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bethe-Heitler-Wheeler-Lamb screening factors for use in atomic showers.
 *
 * These are derived from \citet{bethe-stopping-1934,
 * https://doi.org/10.1098/rspa.1934.0140} (Eq. 31) for the \f$ \phi\f$
 * (elastic) components and \citet{wheeler-electrons-1939,
 * https://doi.org/10.1103/PhysRev.55.858} for the \f$ \psi \f$ (inelastic)
 * components.
 */
struct BhwlScreeningFactors
{
    //! Elastic component, to be multiplied into Z^2
    real_type phi1{};
    //! \f$\phi_1 - \phi_2\f$ corrective term for low-energy secondary
    real_type dphi{};
    //! Inelastic component, to be multiplied into Z
    real_type psi1{};
    //! \f$\psi_1 - \psi_2\f$ corrective term for low-energy secondary
    real_type dpsi{};
};

//---------------------------------------------------------------------------//
/*!
 * Thomas-Fermi screening functions from Tsai.
 *
 * This calculates atomic screening factors given by  \citet{tsai-1974,
 * https://doi.org/10.1103/RevModPhys.46.815} Eq. 3.30-31, as part of the
 * relativistic bremsstrahlung cross section calculation.
 * This model is valid for \f$ Z \ge 5 \f$.
 *
 * The calculator argument is the fraction \f[
 * \delta = \frac{k}{E(k - E)} \equiv \frac{2\delta_\mathrm{Tsai}}{m_e}
 * \f]
 * where \f$E\f$ is the kinetic plus rest mass energy of the electron
 * and \f$k\f$ is the photon energy. (For Bremsstrahlung, the electron is
 * incident and photon is exiting.)
 *
 * The calculated screening functions are:
 * \f[
   \begin{aligned}
     \varphi_1(\gamma)
     &= 20.863 - 2 \ln \left( 1 + (0.55846\gamma)^2 \right)
       - 4 \left( 1 - 0.6 \exp(-0.9\gamma) - 0.4 \exp(-1.5\gamma) \right),
   \\
     \varphi_2(\gamma)
     &= \varphi_1(\gamma)
       - \frac{2}{3} \left( 1 + 6.5\gamma + 6\gamma^2 \right)^{-1},
   \\
     \psi_1(\epsilon)
     &= 28.340 - 2 \ln \left( 1 + (3.621\epsilon)^2 \right)
       - 4 \left( 1 - 0.7 \exp(-8\epsilon) - 0.3 \exp(-29.2\epsilon) \right),
   \\
     \psi_2(\epsilon)
     &= \psi_1(\epsilon)
       - \frac{2}{3} \left( 1 + 40\epsilon + 400\epsilon^2 \right)^{-1}.
   \end{aligned}
 * \f]
 *
 * Here,
 * \f[
   \begin{aligned}
   \gamma &= \frac{100 m_e k}{E (k - E) Z^{1/3}} \\
   \epsilon &= \frac{100 m_e k}{E (k - E) Z^{2/3}}
   \end{aligned}
 * \f]
 * from which we extract input factors precalculated in
 * \c celeritas::RelativisticBremModel:
 * \f[
 * f_\gamma = \frac{100 m_e}{Z^{1/3}}
 * \f]
 * and
 * \f[
 * f_\epsilon = \frac{100 m_e}{Z^{2/3}}
 * \f]
 *
 * \sa RBDiffXsCalculator
 */
class TsaiScreeningCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = BhwlScreeningFactors;
    using Mass = units::MevMass;
    using InvEnergy = RealQuantity<UnitInverse<units::Mev>>;
    //!@}

  public:
    // Construct with defaults
    CELER_FUNCTION inline TsaiScreeningCalculator(Mass gamma_factor,
                                                  Mass epsilon_factor);

    // Calculate screening function from energy transfer
    CELER_FUNCTION inline result_type operator()(InvEnergy delta) const;

  private:
    real_type f_gamma_;
    real_type f_epsilon_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with gamma and epsilon factors.
 */
CELER_FUNCTION
TsaiScreeningCalculator::TsaiScreeningCalculator(Mass gamma_factor,
                                                 Mass epsilon_factor)
    : f_gamma_{gamma_factor.value()}, f_epsilon_{epsilon_factor.value()}
{
    CELER_EXPECT(epsilon_factor > zero_quantity());
    CELER_EXPECT(gamma_factor > epsilon_factor);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate screening function from energy transfer.
 */
CELER_FUNCTION auto TsaiScreeningCalculator::operator()(InvEnergy delta) const
    -> result_type
{
    real_type gam = delta.value() * f_gamma_;
    real_type eps = delta.value() * f_epsilon_;

    using PolyQuad = PolyEvaluator<real_type, 2>;
    using R = real_type;
    result_type func;

    func.phi1 = R(20.863 - 4) - 2 * std::log(1 + ipow<2>(R(0.55846) * gam))
                + R(-4 * -0.6) * std::exp(R(-0.9) * gam)
                + R(-4 * -0.4) * std::exp(R(-1.5) * gam);
    func.dphi = (R{2} / R{3}) / PolyQuad{1, 6.5, 6}(gam);

    func.psi1 = R(28.340 - 4) - 2 * std::log(1 + ipow<2>(R(3.621) * eps))
                + R(-4 * -0.7) * std::exp(R(-8) * eps)
                + R(-4 * -0.3) * std::exp(R(-29.2) * eps);
    func.dpsi = (R{2} / R{3}) / PolyQuad{1, 40, 400}(eps);

    return func;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
