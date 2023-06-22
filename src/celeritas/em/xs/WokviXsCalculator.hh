//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WokviXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/interactor/detail/WokviStateHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the Coulomb scattering cross section using the Wentzel OK and
 * VI model.
 */
class WokviXsCalculator
{
  public:
    // Construct with state data
    inline CELER_FUNCTION
    WokviXsCalculator(detail::WokviStateHelper const& state);

    // Cross sections
    inline CELER_FUNCTION real_type nuclear_xsec() const;
    inline CELER_FUNCTION real_type electron_xsec() const;

  private:
    detail::WokviStateHelper const& state_;

    // Functional form of the integrated Wentzel OK and VI cross section
    inline CELER_FUNCTION real_type wokvi_xs(real_type cos_t_min,
                                             real_type cos_t_max) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state data
 */
CELER_FUNCTION
WokviXsCalculator::WokviXsCalculator(detail::WokviStateHelper const& state)
    : state_(state)
{
}

//---------------------------------------------------------------------------//
/*!
 * Integrated nuclear cross section
 */
CELER_FUNCTION real_type WokviXsCalculator::nuclear_xsec() const
{
    return state_.target_Z()
           * wokvi_xs(state_.cos_t_min_nuc(), state_.cos_t_max_nuc());
}

//---------------------------------------------------------------------------//
/*!
 * Integrated electron cross section
 */
CELER_FUNCTION real_type WokviXsCalculator::electron_xsec() const
{
    return wokvi_xs(state_.cos_t_min_elec(), state_.cos_t_max_elec());
}

//---------------------------------------------------------------------------//
/*!
 * Integrated cross section based on the Wentzel OK and VI model [PRM 8.16]
 */
CELER_FUNCTION real_type WokviXsCalculator::wokvi_xs(real_type cos_t_min,
                                                     real_type cos_t_max) const
{
    if (cos_t_max < cos_t_min)
    {
        const real_type w1 = state_.w_term(cos_t_min);
        const real_type w2 = state_.w_term(cos_t_max);
        return state_.kinetic_factor * state_.mott_factor() * (w2 - w1) / (w1 * w2);
    }
    return 0.0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
