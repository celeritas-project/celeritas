//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ScopedRootErrorHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedRootErrorHandler.hh"

#include <cstring>
#include <TError.h>
#include <TSystem.h>

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"

#include "RootFileManager.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
bool g_has_root_errored_{false};

//---------------------------------------------------------------------------//
/*!
 * Actual ROOT Error Handler function for Celeritas
 */
void RootErrorHandler(Int_t rootlevel,
                      Bool_t must_abort,
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
        g_has_root_errored_ = true;
        if (std::strcmp(location, "TCling::LoadPCM") == 0)
            must_abort = true;
    }
    if (rootlevel >= kBreak)
        level = LogLevel::critical;
    if (rootlevel >= kSysError)
        level = LogLevel::critical;
    if (rootlevel >= kFatal)
        level = LogLevel::critical;

    if (must_abort)
    {
        throw RuntimeError::from_root_error(location, msg);
    }
    else
    {
        // Print log statement
        auto log_msg = ::celeritas::world_logger()({"ROOT", 0}, level);
        if (location)
        {
            log_msg << color_code('x') << location << color_code(' ');
        }
        if (msg && msg[0] != '\0')
        {
            log_msg << ": " << msg;
        }
    }
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Clear ROOT's signal handlers that get installed on startup/activation.
 */
void ScopedRootErrorHandler::disable_signal_handler()
{
    gSystem->ResetSignals();
}

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas ROOT error handler.
 */
ScopedRootErrorHandler::ScopedRootErrorHandler()
{
    if (RootFileManager::use_root())
    {
        previous_ = SetErrorHandler(RootErrorHandler);
        prev_errored_ = g_has_root_errored_;
        ScopedRootErrorHandler::disable_signal_handler();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Raise an exception if at least one error has been logged.
 *
 * Clear the error flag while throwing.
 */
void ScopedRootErrorHandler::throw_if_errors() const
{
    bool prev_errored = g_has_root_errored_;
    g_has_root_errored_ = false;
    CELER_VALIDATE(!prev_errored,
                   << "ROOT encountered non-fatal errors: see log messages "
                      "above");
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous ROOT error handler.
 */
ScopedRootErrorHandler::~ScopedRootErrorHandler()
{
    if (RootFileManager::use_root())
    {
        SetErrorHandler(previous_);
        g_has_root_errored_ = prev_errored_;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
