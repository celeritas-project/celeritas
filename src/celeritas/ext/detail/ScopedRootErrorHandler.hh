//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/RootErrorHandler.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 *  ROOT Error Handler for Celeritas.
 *
 */

void RootErrorHandler(int rootlevel, bool abort_bool, const char *location,
                          const char *msg);

//---------------------------------------------------------------------------//
/*!
 * Install a ROOT Error Handler to redirect the message toward the
 * Celeritas Logger.
 *
 */

class ScopedRootErrorHandler
{
   decltype(&RootErrorHandler) previous_;

public:
    // Install the error handler
    ScopedRootErrorHandler();

    // Return to the previous error handler.
    ~ScopedRootErrorHandler();
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
