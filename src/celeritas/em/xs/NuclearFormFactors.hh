//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/NuclearFormFactors.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Helper traits used in nuclear form factors
struct NuclearFormFactorTraits
{
    //!@{
    //! \name Type aliases
    using AtomicMassNumber = AtomicNumber;
    using Momentum = units::MevMomentum;
    using MomentumSq = units::MevMomentumSq;
    using InvMomentum = RealQuantity<UnitInverse<Momentum::unit_type>>;
    using InvMomentumSq = RealQuantity<UnitInverse<MomentumSq::unit_type>>;
    using FFType = NuclearFormFactorType;
    //!@}

    //! Momentum transfer prefactor: 1 fm / hbar
    static CELER_CONSTEXPR_FUNCTION InvMomentum fm_par_hbar()
    {
        return native_value_to<InvMomentum>(units::femtometer
                                            / constants::hbar_planck);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Exponential nuclear form factor.
 *
 * This nuclear form factor corresponds \c NuclearFormFactorType::exponential
 * and assumes the nuclear charge decays exponentially from its center. This
 * assumes a parameterization of the atomic nucleus valid for light and medium
 * atomic nuclei from Eq. 7 of \citet{butkevich-msc-2002,
 * https://doi.org/10.1016/S0168-9002(02)00478-3} : \f[
   R_N = 1.27A^{0.27} \,\mathrm{fm}
 * \f] with a special case for the proton radius, \f$ R_p = 0.85 \f$ fm.
 *
 * See \citet{leroy-2016, https://doi-org.ezproxy.cern.ch/10.1142/9167} Eq.
 * 2.262.
 *
 * \todo Instead of using this coarse parameterization, we should add nuclear
 * radius to the isotope properties for a more accurate treatment, and
 * construct these classes directly from the atomic radius.
 */
class ExpNuclearFormFactor : public NuclearFormFactorTraits
{
  public:
    //! Form factor type corresponding to this distribution
    static CELER_CONSTEXPR_FUNCTION FFType ff_type()
    {
        return FFType::exponential;
    }

    // Construct with atomic mass number
    explicit inline CELER_FUNCTION
    ExpNuclearFormFactor(AtomicMassNumber a_mass);

    // Construct with precalculated form factor
    explicit inline CELER_FUNCTION
    ExpNuclearFormFactor(InvMomentumSq prefactor);

    // Calculate from square of target momentum
    inline CELER_FUNCTION real_type operator()(MomentumSq target_momsq) const;

    // Calculate from target momentum
    inline CELER_FUNCTION real_type operator()(Momentum target_mom) const;

    //! Nuclear form prefactor for the selected isotope
    CELER_FORCEINLINE_FUNCTION InvMomentumSq prefactor() const
    {
        return InvMomentumSq{prefactor_};
    }

  private:
    real_type prefactor_;  // Function of nuclear radius [(MeV/c)^{-2}]
};

//---------------------------------------------------------------------------//
/*!
 * Gaussian nuclear form factor.
 *
 * This nuclear form factor corresponds \c NuclearFormFactorType::gaussian and
 * assumes a Gaussian distribution of nuclear charge: see \citet{leroy-2016,
 * https://doi-org.ezproxy.cern.ch/10.1142/9167} Eq. 2.264. Its
 * prefactor has the same value as the exponential.
 */
class GaussianNuclearFormFactor : public ExpNuclearFormFactor
{
  public:
    //! Form factor type corresponding to this distribution
    static CELER_CONSTEXPR_FUNCTION FFType ff_type()
    {
        return FFType::gaussian;
    }

    using ExpNuclearFormFactor::ExpNuclearFormFactor;

    // Calculate from square of target momentum
    inline CELER_FUNCTION real_type operator()(MomentumSq target_momsq) const;

    // Calculate from target momentum
    inline CELER_FUNCTION real_type operator()(Momentum target_mom) const;
};

//---------------------------------------------------------------------------//
/*!
 * Uniform-uniform folded nuclear form factor.
 *
 * This nuclear form factor corresponds \c NuclearFormFactorType::flat and
 * assumes a uniform nuclear charge at the center with a smoothly decreasing
 * charge at the surface. From \citet{leroy-2016,
 * https://doi-org.ezproxy.cern.ch/10.1142/9167} Eq. 2.265, this leads to a
 * form factor: \f[
 * F(q) = F'(x(R_0, q)) F'(x(R_1, q))
 * \f]
 * where \f$ x \equiv q R / \hbar \f$ uses the effective nuclear radius
 * \f$ R_0 = 1.2 A^{1/3} \,\mathrm{fm} \f$ and nuclear surface skin
 * \f$ R_1 = 2.0 \,\mathrm{fm} \f$,
 * and
 * \f[
 * F'(x) = \frac{3}{x^3} ( \sin x - x \cos x)
 * \f]
 * is the form factor for a uniformly charged sphere.
 *
 * \warning This form factor suffers from catastrophic numerical cancellation
 * for small radii and momenta so should only be used for large nuclei or large
 * momentum transfers.
 *
 * See also
 * \citet{helm-scattering-1956, https://doi.org/10.1103/PhysRev.104.1466},
 * \citet{fernandez-msc-1993, https://doi.org/10.1016/0168-583X(93)95827-R}
 */
class UUNuclearFormFactor : public NuclearFormFactorTraits
{
  public:
    //! Form factor type corresponding to this distribution
    static CELER_CONSTEXPR_FUNCTION FFType ff_type() { return FFType::flat; }

    // Construct with atomic mass number
    explicit inline CELER_FUNCTION UUNuclearFormFactor(AtomicMassNumber a_mass);

    // Calculate from square of target momentum
    inline CELER_FUNCTION real_type operator()(MomentumSq target_momsq) const;

    // Calculate from target momentum
    inline CELER_FUNCTION real_type operator()(Momentum target_mom) const;

  private:
    real_type nucl_radius_fm_;

    // Effective nuclear skin radius: 2 fm
    static CELER_CONSTEXPR_FUNCTION real_type skin_radius_fm() { return 2; }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from atomic mass number.
 */
CELER_FUNCTION
ExpNuclearFormFactor::ExpNuclearFormFactor(AtomicMassNumber a_mass)
{
    CELER_EXPECT(a_mass);
    real_type nucl_radius_fm = [a_mass] {
        if (CELER_UNLIKELY(a_mass == AtomicMassNumber{1}))
        {
            // Special case for proton radius
            return real_type{0.85};
        }
        return real_type{1.27}
               * fastpow(real_type(a_mass.get()), real_type{0.27});
    }();
    prefactor_ = ipow<2>(nucl_radius_fm * value_as<InvMomentum>(fm_par_hbar()))
                 * (real_type{1} / 12);
    CELER_ENSURE(prefactor_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with precalculated form factor.
 */
CELER_FUNCTION
ExpNuclearFormFactor::ExpNuclearFormFactor(InvMomentumSq prefactor)
    : prefactor_{prefactor.value()}
{
    CELER_EXPECT(prefactor_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the exponential folded form factor from the square momentum.
 */
CELER_FUNCTION real_type
ExpNuclearFormFactor::operator()(MomentumSq target_momsq) const
{
    CELER_EXPECT(target_momsq >= zero_quantity());
    return 1 / ipow<2>(1 + prefactor_ * target_momsq.value());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the exponential folded form factor.
 */
CELER_FUNCTION real_type ExpNuclearFormFactor::operator()(Momentum target_mom) const
{
    return (*this)(MomentumSq{ipow<2>(target_mom.value())});
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the gaussian folded form factor.
 */
CELER_FUNCTION real_type
GaussianNuclearFormFactor::operator()(MomentumSq target_momsq) const
{
    CELER_EXPECT(target_momsq >= zero_quantity());
    return std::exp(-2 * value_as<InvMomentumSq>(this->prefactor())
                    * target_momsq.value());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the gaussian folded form factor by squaring the momentum.
 */
CELER_FUNCTION real_type
GaussianNuclearFormFactor::operator()(Momentum target_mom) const
{
    return (*this)(MomentumSq{ipow<2>(target_mom.value())});
}

//---------------------------------------------------------------------------//
/*!
 * Construct from atomic mass number.
 */
CELER_FUNCTION UUNuclearFormFactor::UUNuclearFormFactor(AtomicMassNumber a_mass)
{
    CELER_EXPECT(a_mass);
    nucl_radius_fm_ = real_type{1.2}
                      * fastpow(real_type(a_mass.get()), real_type{1} / 3);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the uniform-uniform folded form factor by calculating momentum.
 */
CELER_FUNCTION real_type
UUNuclearFormFactor::operator()(MomentumSq target_momsq) const
{
    CELER_EXPECT(target_momsq >= zero_quantity());
    return (*this)(Momentum{std::sqrt(target_momsq.value())});
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the uniform-uniform folded form factor.
 */
CELER_FUNCTION real_type UUNuclearFormFactor::operator()(Momentum target_mom) const
{
    auto sphere_ff = [&target_mom](real_type r) {
        // x = q R / hbar
        // r units: fm
        real_type x = value_as<Momentum>(target_mom)
                      * (r * value_as<InvMomentum>(fm_par_hbar()));
        return (3 / ipow<3>(x)) * std::fma(-x, std::cos(x), std::sin(x));
    };

    // Due to catastrophic error for small x, clamp the result to 1
    return min(sphere_ff(nucl_radius_fm_)
                   * sphere_ff(UUNuclearFormFactor::skin_radius_fm()),
               real_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
