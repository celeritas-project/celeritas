//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DTMixMuonicMoleculeSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a muonic molecule by calculating the interaction lengths of the
 * possible molecule formations.
 *
 * This is the equivalent of Geant4's
 * \c G4VRestProcess::AtRestGetPhysicalInteractionLength
 */
class DTMixMuonicMoleculeSelector
{
  public:
    //! Construct with args; \todo Update documentation
    inline CELER_FUNCTION DTMixMuonicMoleculeSelector(/* args */);

    // Select muonic molecule
    template<class Engine>
    inline CELER_FUNCTION MucfMuonicMolecule operator()(Engine&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with args.
 *
 * \todo Update documentation
 */
CELER_FUNCTION
DTMixMuonicMoleculeSelector::DTMixMuonicMoleculeSelector(/* args */)
{
    //! \todo Implement
    CELER_NOT_IMPLEMENTED("Mucf muonic molecule selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return selected muonic molecule.
 */
template<class Engine>
CELER_FUNCTION MucfMuonicMolecule
DTMixMuonicMoleculeSelector::operator()(Engine&)
{
    MucfMuonicMolecule result{MucfMuonicMolecule::size_};

    //! \todo Implement

    CELER_ENSURE(result < MucfMuonicMolecule::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
