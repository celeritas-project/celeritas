//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DTMixMuonicAtomSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a muonic atom given the material information.
 */
class DTMixMuonicAtomSelector
{
  public:
    //! Construct with args; \todo Update documentation
    inline CELER_FUNCTION DTMixMuonicAtomSelector(/* args */);

    // Select muonic atom
    template<class Engine>
    inline CELER_FUNCTION MucfMuonicAtom operator()(Engine&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with args.

 * \todo Update documentation
 */
CELER_FUNCTION DTMixMuonicAtomSelector::DTMixMuonicAtomSelector(/* args */)
{
    CELER_NOT_IMPLEMENTED("Mucf muonic atom selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return selected muonic atom.
 */
template<class Engine>
CELER_FUNCTION MucfMuonicAtom DTMixMuonicAtomSelector::operator()(Engine&)
{
    MucfMuonicAtom result{MucfMuonicAtom::size_};

    //! \todo Implement

    CELER_ENSURE(result < MucfMuonicAtom::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
