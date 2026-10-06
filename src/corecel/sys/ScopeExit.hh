//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopeExit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call a function when this object is destroyed (at end of scope).
 */
template<class F>
class ScopeExit
{
  public:
    //! Construct with functor
    ScopeExit(F func) : func_{std::forward<F>(func)} {}

    //! Call functor on destruction
    ~ScopeExit() { func_(); }

    CELER_DELETE_COPY_MOVE(ScopeExit);

  private:
    F func_;
};

template<class F>
ScopeExit(F&& func) -> ScopeExit<F>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
