//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LogContextException.cc
//---------------------------------------------------------------------------//
#include "LogContextException.hh"

#include "corecel/Assert.hh"

#include "ExceptionOutput.hh"
#include "Logger.hh"
#include "OutputRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void LogContextException::operator()(std::exception_ptr eptr)
{
    if (this->out)
    {
        this->out->insert(std::make_shared<ExceptionOutput>(eptr));
    }
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (RichContextException const& e)
    {
        CELER_LOG_LOCAL(critical) << "The following error is from " << e.what();
        try
        {
            std::rethrow_if_nested(e);
        }
        catch (...)
        {
            // Prevent reregistration of the exception
            this->out = nullptr;
            return (*this)(std::current_exception());
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
