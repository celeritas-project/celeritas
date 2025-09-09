//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ExceptionConverter.cu
//---------------------------------------------------------------------------//
#include "ExceptionConverter.hh"

#include <G4Exception.hh>
#include <thrust/system/system_error.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Catch exceptions that require CUDA headers.
 *
 * This should be called as a fallback after Celeritas exceptions are caught.
 * It can't be part of the main "catch" clause when building with HIP because
 * of how the HIP-thrust fork is implemented.
 */
void ExceptionConverter::convert_device_exceptions(std::exception_ptr eptr) const
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (thrust::system_error const& e)
    {
        G4Exception("Thrust GPU library", err_code_, FatalException, e.what());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
