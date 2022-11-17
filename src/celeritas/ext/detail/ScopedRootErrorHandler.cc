//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/RootErrorHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedRootErrorHandler.hh"

#include <TError.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Actual ROOT Error Handler function for Celeritas
 */
void RootErrorHandler(Int_t rootlevel, Bool_t abort_bool, const char *location,
                          const char *msg)
{
   if (rootlevel < gErrorIgnoreLevel)
      return;

   LogLevel level = LogLevel::status;

   if (rootlevel >= kInfo)
      level = LogLevel::info;
   if (rootlevel >= kWarning)
      level = LogLevel::warning;
   if (rootlevel >= kError)
      level = LogLevel::error;
   if (rootlevel >= kBreak)
      level = LogLevel::critical;
   if (rootlevel >= kSysError)
      level = LogLevel::critical;
   if (rootlevel >= kFatal)
      level = LogLevel::critical;

   ::celeritas::world_logger()({"ROOT" , 0}, level)
      << "<" << (location ? location : "unspecified location")
      << ">:  " << msg;

   if (abort_bool) {
      auto err = RuntimeError::from_root_error(
        location, msg);
      throw err;
   }
}

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas ROOT error handler
 */
ScopedRootErrorHandler::ScopedRootErrorHandler() :
   previous_(SetErrorHandler(RootErrorHandler))
{
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous ROOT error handler
 */
ScopedRootErrorHandler::~ScopedRootErrorHandler()
{
   SetErrorHandler(previous_);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
