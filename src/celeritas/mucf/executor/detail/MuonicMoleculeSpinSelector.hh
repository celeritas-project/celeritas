//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/MuonicMoleculeSpinSelector.hh
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
 * Select muonic molecule spin, in units of \f$ \frac{\hbar}{2} \f$.
 *
 * Applicable to \f$ (dd)_\mu \f$, * \f$ (dt)_\mu \f$ , and \f$ (tt)_\mu \f$.
 */
class MuonicMoleculeSpinSelector
{
  public:
    // Construct with muonic molecule type
    inline CELER_FUNCTION
    MuonicMoleculeSpinSelector(MucfMuonicMolecule molecule);

    // Sample and return a spin value in units of hbar / 2
    template<class Engine>
    inline CELER_FUNCTION size_type operator()(Engine&);

  private:
    MucfMuonicMolecule molecule_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with muonic molecule.
 */
CELER_FUNCTION MuonicMoleculeSpinSelector::MuonicMoleculeSpinSelector(
    MucfMuonicMolecule molecule)
    : molecule_(molecule)
{
    CELER_EXPECT(molecule_ < MucfMuonicMolecule::size_);
    CELER_NOT_IMPLEMENTED("Mucf muonic molecule spin selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return selected spin, in units of \f$ \hbar / 2 \f$.
 */
template<class Engine>
CELER_FUNCTION size_type MuonicMoleculeSpinSelector::operator()(Engine&)
{
    //! \todo switch on molecule_
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
