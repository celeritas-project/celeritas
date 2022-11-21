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
 * Install a ROOT Error Handler to redirect the message toward the
 * Celeritas Logger.
 *
 */

class ScopedRootErrorHandler
{
public:
    // Install the error handler
    ScopedRootErrorHandler();

    // Return to the previous error handler.
    ~ScopedRootErrorHandler();

private:
   using ErrorHandlerFuncPtr = void (*)(int, bool, const char*, const char*);
   ErrorHandlerFuncPtr previous_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
