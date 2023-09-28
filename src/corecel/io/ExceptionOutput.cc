//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ExceptionOutput.cc
//---------------------------------------------------------------------------//
#include "ExceptionOutput.hh"

#include <string>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "JsonPimpl.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/AssertIO.json.hh"
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_JSON
void eptr_to_json(nlohmann::json&, std::exception_ptr const&);

void try_nested_to_json(nlohmann::json& j, std::exception const& e)
{
    try
    {
        std::rethrow_if_nested(e);
    }
    catch (...)
    {
        // Replace the output with the embedded (lower-level) exception
        auto orig = std::move(j);
        eptr_to_json(j, std::current_exception());
        // Save the lower-level exception as context
        j["context"] = std::move(orig);
    }
}

void eptr_to_json(nlohmann::json& j, std::exception_ptr const& eptr)
{
    // Process the error pointer by rethrowing it and catching possible types
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (RuntimeError const& e)
    {
        j = e.details();
        j["type"] = "RuntimeError";
    }
    catch (DebugError const& e)
    {
        j = e.details();
        j["type"] = "DebugError";
    }
    catch (RichContextException const& e)
    {
        // Construct detailed info from a rich exception
        j = e;
        j["type"] = e.type();
    }
    catch (std::exception const& e)
    {
        // Save unknown exception info
        TypeDemangler<std::exception> demangle;
        j = {{"type", demangle(e)}, {"what", e.what()}};
    }
    catch (...)
    {
        j = {{"type", "unknown"}};
    }

    // Rethrow to process any chained exceptions
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (std::exception const& e)
    {
        try_nested_to_json(j, e);
    }
    catch (...)
    {
    }
}
#endif
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with an exception pointer.
 */
ExceptionOutput::ExceptionOutput(std::exception_ptr eptr)
{
    CELER_EXPECT(eptr);
#if CELERITAS_USE_JSON
    output_ = std::make_unique<JsonPimpl>();
    eptr_to_json(output_->obj, eptr);
#endif
}

//---------------------------------------------------------------------------//
//! Default destructor
ExceptionOutput::~ExceptionOutput() = default;

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void ExceptionOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    CELER_EXPECT(output_);
    j->obj = output_->obj;
#else
    CELER_DISCARD(j);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
