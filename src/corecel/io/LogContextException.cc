//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LogContextException.cc
//---------------------------------------------------------------------------//
#include "LogContextException.hh"

#include "corecel/Assert.hh"

#include "Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void log_context_exception(std::exception_ptr eptr)
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (RichContextException const& e)
    {
        CELER_LOG_LOCAL(critical)
            << "The following error is from: " << e.what();
        try
        {
            std::rethrow_if_nested(e);
        }
        catch (...)
        {
            return log_context_exception(std::current_exception());
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
