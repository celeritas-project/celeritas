//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ScopedRootErrorHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Install a ROOT Error Handler to redirect the message toward the
 * Celeritas Logger.
 */
class ScopedRootErrorHandler
{
  public:
    // Clear ROOT's signal handlers that get installed on startup/activation
    static void disable_signal_handler();

    // Install the error handler
    ScopedRootErrorHandler();

    // Raise an exception if at least one error has been logged
    void throw_if_errors() const;

    // Return to the previous error handler.
    ~ScopedRootErrorHandler();
    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedRootErrorHandler);
    //!@}

  private:
    using ErrorHandlerFuncPtr = void (*)(int, bool, char const*, char const*);

    ErrorHandlerFuncPtr previous_{nullptr};
    bool prev_errored_{false};
};

#if !CELERITAS_USE_ROOT
//!@{
//! Do nothing if ROOT is disabled (source file will not be compiled)
inline void ScopedRootErrorHandler::disable_signal_handler() {}
inline ScopedRootErrorHandler::ScopedRootErrorHandler()
{
    CELER_DISCARD(previous_);
    CELER_DISCARD(prev_errored_);
}
inline ScopedRootErrorHandler::~ScopedRootErrorHandler() {}
inline void ScopedRootErrorHandler::throw_if_errors() const {}
//!@}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
