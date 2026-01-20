//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/MuonicAtomSpinSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select muonic atom spin, in units of \f$ \frac{\hbar}{2} \f$.
 *
 * Applicable to \f$ (p)_\mu \f$, \f$ (d)_\mu \f$, and \f$ (t)_\mu \f$.
 */
class MuonicAtomSpinSelector
{
  public:
    // Construct with muonic atom type
    inline CELER_FUNCTION MuonicAtomSpinSelector(MucfMuonicAtom atom);

    // Sample and return a spin value in units of hbar / 2
    template<class Engine>
    inline CELER_FUNCTION size_type operator()(Engine&);

  private:
    MucfMuonicAtom atom_;
    //! \todo Add constant atom spin sampling rejection fractions
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with muonic atom.
 */
CELER_FUNCTION
MuonicAtomSpinSelector::MuonicAtomSpinSelector(MucfMuonicAtom atom)
    : atom_(atom)
{
    CELER_EXPECT(atom_ < MucfMuonicAtom::size_);
    CELER_NOT_IMPLEMENTED("Mucf muonic atom spin selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return selected spin, in units of \f$ \hbar / 2 \f$.
 */
template<class Engine>
CELER_FUNCTION size_type MuonicAtomSpinSelector::operator()(Engine&)
{
    //! \todo switch on atom_
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
