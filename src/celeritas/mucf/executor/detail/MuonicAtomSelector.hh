//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/MuonicAtomSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/Types.hh"
#include "celeritas/mucf/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a muonic atom given the material information.
 *
 * This class uses the \f$ q_\text{1S} \f$ formula
 * \citet{bom-experimentaldt-2005, https://doi.org/10.1134/1.1926428}
 * \f[
 * q_\text{1s} = \frac{1}{1 + 2.9 C_t},
 * \f]
 * where \f$ C_t \f$  is the relative tritium isotope concentration and
 * \f$ q_\text{1s} \f$ is the fraction muonic deuterium atoms in the ground
 * state. This expression allows calculating the probability of forming a
 * muonic deuterium atom via
 * \f[
 * P_\text{d} = C_d \times q\text{1s}.
 * \f]
 *
 * If a selected uniform random number is \f$ x \leq P_\text{d} \f$, a muonic
 * deuterium is formed. Otherwside, a muonic tritium is selected.
 */
class MuonicAtomSelector
{
  public:
    //! Construct with material information
    inline CELER_FUNCTION MuonicAtomSelector(real_type deuterium_fraction,
                                             real_type tritium_fraction);

    // Select muonic atom
    template<class Engine>
    inline CELER_FUNCTION MucfMuonicAtom operator()(Engine& rng);

  private:
    real_type deuterium_probability_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with material information.
 */
CELER_FUNCTION
MuonicAtomSelector::MuonicAtomSelector(real_type deuterium_fraction,
                                       real_type tritium_fraction)
{
    CELER_EXPECT(deuterium_fraction >= 0);
    CELER_EXPECT(tritium_fraction >= 0);
    CELER_EXPECT(deuterium_fraction + tritium_fraction > 0);
    CELER_EXPECT(deuterium_fraction + tritium_fraction <= 1);

    real_type const q1s = real_type{1}
                          / (real_type{1} + real_type{2.9} * tritium_fraction);
    deuterium_probability_ = deuterium_fraction * q1s;

    CELER_ENSURE(deuterium_probability_ >= 0 && deuterium_probability_ <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Select a muonic atom.
 */
template<class Engine>
CELER_FUNCTION MucfMuonicAtom MuonicAtomSelector::operator()(Engine& rng)
{
    return BernoulliDistribution(deuterium_probability_)(rng)
               ? MucfMuonicAtom::deuterium
               : MucfMuonicAtom::tritium;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
