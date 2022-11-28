//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ExceptionOutput.cc
//---------------------------------------------------------------------------//
#include "ExceptionOutput.hh"

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "JsonPimpl.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
#if CELERITAS_USE_JSON
namespace
{
//---------------------------------------------------------------------------//
template<class T>
void details_to_json(nlohmann::json& j, const T& d)
{
    j["which"] = to_cstring(d.which);
    if (d.condition)
    {
        j["condition"] = d.condition;
    }
    if (d.file && d.file[0] != '\0')
    {
        j["file"] = d.file;
    }
    if (d.line != 0)
    {
        j["line"] = d.line;
    }
}
//---------------------------------------------------------------------------//
} // namespace

void to_json(nlohmann::json& j, const DebugErrorDetails& d)
{
    details_to_json(j, d);
}

void to_json(nlohmann::json& j, const RuntimeErrorDetails& d)
{
    j["what"] = d.what;
    details_to_json(j, d);
}

void json_from_eptr(nlohmann::json& j, const std::exception_ptr& eptr)
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (const RuntimeError& e)
    {
        j         = e.details();
        j["type"] = "RuntimeError";
    }
    catch (const DebugError& e)
    {
        j         = e.details();
        j["type"] = "DebugError";
    }
    catch (const std::exception& e)
    {
        // Save unknown exception info
        TypeDemangler<std::exception> demangle;
        j = {{"type", demangle(e)}, {"what", e.what()}};
    }
    catch (...)
    {
        j = {{"type", "unknown"}};
    }
}

//---------------------------------------------------------------------------//
#endif

//---------------------------------------------------------------------------//
/*!
 * Construct with an exception pointer.
 */
ExceptionOutput::ExceptionOutput(std::exception_ptr eptr)
{
    CELER_EXPECT(eptr);
#if CELERITAS_USE_JSON
    output_ = std::make_unique<JsonPimpl>();
    json_from_eptr(output_->obj, eptr);
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
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
