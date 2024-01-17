//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LogContextException.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>

namespace celeritas
{
class OutputRegistry;

//---------------------------------------------------------------------------//
/*!
 * Log an exception's context and optionally save to an output registry.
 *
 * Example:
 * \code
    CELER_TRY_HANDLE(step(), LogContextException{this->output_reg().get()});
   \endcode
 */
struct LogContextException
{
    void operator()(std::exception_ptr p);

    OutputRegistry* out{nullptr};
};

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
inline void log_context_exception(std::exception_ptr p)
{
    return LogContextException{}(p);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
