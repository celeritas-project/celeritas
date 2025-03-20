//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/ScreeningFunctions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Array.hh"

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
    //! Elastic component, multiplied into Z^2
    real_type phi1{};
    //! \f$\phi_1 - \phi_2\f$ corrective term for low-energy secondary
    real_type dphi{};
    //! Inelastic component, multiplied into Z
    real_type psi1{};
    //! \f$\psi_1 - \psi_2\f$ corrective term for low-energy secondary
    real_type dpsi{};
};

//---------------------------------------------------------------------------//
/*!
 * Thomas-Fermi screening functions from Tsai.
 *
 * This calculates atomic screening factors given by  \citet{tsai-1974,
 * https://doi.org/10.1103/RevModPhys.46.815} Eq. 3.30-31:
 *
 * \f[
 * \varphi_1(\gamma) = 20.863 - 2 \ln \left( 1 + (0.55846\gamma)^2 \right)
 * - 4 \left( 1 - 0.6 \exp(-0.9\gamma) - 0.4 \exp(-1.5\gamma) \right),
 * \f]
 *
 * \f[
 * \varphi_2(\gamma) = \varphi_1(\gamma) - \frac{2}{3} \left( 1 + 6.5\gamma +
 * 6\gamma^2 \right)^{-1}, \f]
 *
 * \f[
 * \psi_1(\epsilon) = 28.340 - 2 \ln \left( 1 + (3.621\epsilon)^2 \right)
 * - 4 \left( 1 - 0.7 \exp(-8\epsilon) - 0.3 \exp(-29.2\epsilon) \right),
 * \f]
 *
 * \f[
 * \psi_2(\epsilon) = \psi_1(\epsilon) - \frac{2}{3} \left( 1 + 40\epsilon +
 * 400\epsilon^2 \right)^{-1}. \f]
 *
 * Here,
 * \f[
 * \gamma = \frac{100 m_e k}{E (k - E) Z^{1/3}}
 * \f]
 * and
 * \f[
 * f_\epsilon = \frac{100 m_e k}{E (k - E) Z^{2/3}}
 * \f]
 * from which we extract input factors preclaculated in
 * celeritas::RelativisticBremModel as
 * \f[
 * f_\gamma = \frac{100 m_e}{Z^{1/3}}
 * \f]
 * and
 * \f[
 * f_\epsilon = \frac{100 m_e}{Z^{2/3}}
 * \f]
 *
 * The calculator argument is the unitless fraction \f[
 * \delta' = \frac{k}{E(k - E)} \equiv \frac{2\delta_\mathrm{Tsai}}{m_e}
 * \f]
 * where \f$k\f$ is the kinetic plus rest mass energy of the incident electron
 * and \f$E\f$ is the exiting gamma energy.
 * This model is valid for \f$Z \ge 5\f$.
 */
class TsaiScreeningCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = BhwlScreeningFactors;
    //!@}

  public:
    // Construct with defaults
    CELER_FUNCTION inline TsaiScreeningCalculator(real_type gamma_factor,
                                                  real_type epsilon_factor);

    // Calculate screening function from energy transfer
    CELER_FUNCTION result_type operator()(real_type delta) const;

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
TsaiScreeningCalculator::TsaiScreeningCalculator(real_type gamma_factor,
                                                 real_type epsilon_factor)
    : f_gamma_{gamma_factor}, f_epsilon_{epsilon_factor}
{
    CELER_EXPECT(epsilon_factor > 0);
    CELER_EXPECT(gamma_factor > epsilon_factor);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate screening function from energy transfer.
 */
CELER_FUNCTION auto TsaiScreeningCalculator::operator()(real_type delta) const
    -> result_type
{
    CELER_EXPECT(delta >= 0 && delta <= 1);
    real_type gam = delta * f_gamma_;
    real_type eps = delta * f_epsilon_;

    using PolyQuad = PolyEvaluator<real_type, 2>;
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
