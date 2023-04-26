//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LogContextException.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Log any RichContextException and rethrow the embedded pointer.
 *
 * This is useful for unit tests and other situations where the nearest `catch`
 * does not use `std::rethrow_if_nested`.
 *
 * \code
 CELER_TRY_HANDLE(step(), log_context_exception);
   \endcode
 */
void log_context_exception(std::exception_ptr p);

//---------------------------------------------------------------------------//
}  // namespace celeritas
