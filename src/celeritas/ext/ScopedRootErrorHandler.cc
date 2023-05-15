//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ScopedRootErrorHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedRootErrorHandler.hh"

#include <TError.h>
#include <TSystem.h>

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
bool has_root_errored{false};

//---------------------------------------------------------------------------//
/*!
 * Actual ROOT Error Handler function for Celeritas
 */
void RootErrorHandler(Int_t rootlevel,
                      Bool_t abort_bool,
                      char const* location,
                      char const* msg)
{
    if (rootlevel < gErrorIgnoreLevel)
        return;

    LogLevel level = LogLevel::status;

    if (rootlevel >= kInfo)
        level = LogLevel::info;
    if (rootlevel >= kWarning)
        level = LogLevel::warning;
    if (rootlevel >= kError)
    {
        level = LogLevel::error;
        has_root_errored = true;
    }
    if (rootlevel >= kBreak)
        level = LogLevel::critical;
    if (rootlevel >= kSysError)
        level = LogLevel::critical;
    if (rootlevel >= kFatal)
        level = LogLevel::critical;

    if (abort_bool)
    {
        throw RuntimeError::from_root_error(location, msg);
    }
    else
    {
        // Print log statement
        auto log_msg = ::celeritas::world_logger()({"ROOT", 0}, level);
        if (location)
        {
            log_msg << color_code('x') << location << color_code(' ') << ": ";
        }
        log_msg << msg;
    }
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas ROOT error handler
 */
ScopedRootErrorHandler::ScopedRootErrorHandler()
    : previous_(SetErrorHandler(RootErrorHandler))
    , prev_errored_{has_root_errored}
{
    // Disable ROOT interception of system signals the first time we run
    [[maybe_unused]] static bool const disabled_root_backtrace = [] {
        gSystem->ResetSignals();
        return true;
    }();
}

//---------------------------------------------------------------------------//
/*!
 * Raise an exception if at least one error has been logged.
 * 
 * Clear the error flag while throwing.
 */
void ScopedRootErrorHandler::throw_if_errors() const
{
    bool prev_errored = has_root_errored;
    has_root_errored = false;
    CELER_VALIDATE(!prev_errored,
                   << "ROOT encountered non-fatal errors: see log messages "
                      "above");
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous ROOT error handler
 */
ScopedRootErrorHandler::~ScopedRootErrorHandler()
{
    SetErrorHandler(previous_);
    has_root_errored = prev_errored_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
