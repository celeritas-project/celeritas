//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WentzelXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the ratio of the electron to total Wentzel cross sections for
 * elastic Coulomb scattering.
 */
class WentzelXsCalculator
{
  public:
    // Construct the calculator from the given values
    inline CELER_FUNCTION WentzelXsCalculator(int target_z,
                                              real_type screening_coefficient,
                                              real_type cos_t_max_elec);

    // The ratio of electron to total cross section for Coulomb scattering
    inline CELER_FUNCTION real_type operator()() const;

  private:
    // Target atomic number
    int const target_z_;

    // Moliere screening coefficient
    real_type const screening_coefficient_;

    // Cosine of the maximum scattering angle off of electrons
    real_type const cos_t_max_elec_;

    //! (Reduced) nuclear cross section
    inline CELER_FUNCTION real_type nuclear_xsec() const;

    //! (Reduced) electron cross section
    inline CELER_FUNCTION real_type electron_xsec() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state data
 */
CELER_FUNCTION
WentzelXsCalculator::WentzelXsCalculator(int target_z,
                                         real_type screening_coefficient,
                                         real_type cos_t_max_elec)
    : target_z_(target_z)
    , screening_coefficient_(screening_coefficient)
    , cos_t_max_elec_(cos_t_max_elec)
{
    CELER_EXPECT(target_z_ > 0);
    CELER_EXPECT(screening_coefficient > 0);
    CELER_EXPECT(cos_t_max_elec >= -1 && cos_t_max_elec <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Ratio of electron cross section to the total (nuclear + electron)
 * cross section.
 */
CELER_FUNCTION real_type WentzelXsCalculator::operator()() const
{
    const real_type nuc_xsec = nuclear_xsec();
    const real_type elec_xsec = electron_xsec();

    return elec_xsec / (nuc_xsec + elec_xsec);
}

//---------------------------------------------------------------------------//
/*!
 * Reduced integrated nuclear cross section from theta=0 to pi.
 *
 * Since this is only used in the electric ratio, mutual factors with the
 * electron cross section are dropped.
 */
CELER_FUNCTION real_type WentzelXsCalculator::nuclear_xsec() const
{
    return target_z_ / (1 + screening_coefficient_);
}

//---------------------------------------------------------------------------//
/*!
 * Reduced integrated electron cross section from theta=0 to maximum
 * electron angle.
 *
 * Since this is only used in the electric ratio, mutual factors with the
 * electron cross section are dropped.
 */
CELER_FUNCTION real_type WentzelXsCalculator::electron_xsec() const
{
    return (1 - cos_t_max_elec_)
           / (1 - cos_t_max_elec_ + 2 * screening_coefficient_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
