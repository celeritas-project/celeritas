//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/ScopedVgNavState.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "geocel/vg/VecgeomTypes.hh"

#include "VgNavStateWrapper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! When using the path navigator, just return a reference
class ReferencedVgNavState
{
  public:
    // Construct with reference to nav wrapper, and construct temp via copy
    CELER_FORCEINLINE_FUNCTION ReferencedVgNavState(VgNavState& src)
        : src_{src}
    {
    }

    //! Implicit cast to reference type for use in VecGeom function calls
    CELER_FORCEINLINE_FUNCTION
    operator VgNavState&() { return src_; }

  private:
    VgNavState& src_;
};

//---------------------------------------------------------------------------//
/*!
 * RAII temporary navigator that copies back to the original on exit.
 *
 * This is necessary to interface with VecGeom's navigation methods using a
 * trimmed-down state.
 */
class ScopedTempVgNavState
{
  public:
    //!@{
    //! \name Type aliases
    using VgNavWrapper = VgNavStateWrapper;
    using VgNavState = VgNavStateWrapper::VgNavState;
    //!@}

  public:
    // Construct with reference to nav wrapper, and construct temp via copy
    inline CELER_FUNCTION ScopedTempVgNavState(VgNavWrapper const& src)
        : src_{src}, tmp_{src}
    {
    }

    // Copy back to nav wrapper on destruction
    CELER_FORCEINLINE_FUNCTION ~ScopedTempVgNavState() { src_ = tmp_; }

    //! Implicit cast to reference type for use in VecGeom function calls
    CELER_FORCEINLINE_FUNCTION
    operator VgNavState&() { return tmp_; }

  private:
    VgNavWrapper src_;
    VgNavState tmp_;
};

//---------------------------------------------------------------------------//

#if CELER_VGNAV == CELER_VGNAV_PATH
using ScopedVgNavState = ReferencedVgNavState;
#else
using ScopedVgNavState = ScopedTempVgNavState;
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
