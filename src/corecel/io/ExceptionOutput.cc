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
    j["what"]      = d.what;
    details_to_json(j, d);
}

#endif

//---------------------------------------------------------------------------//
/*!
 * Construct with an exception object.
 *
 * This saves the type information and message for later output.
 */
ExceptionOutput::ExceptionOutput(const std::exception& e)
{
#if CELERITAS_USE_JSON
    output_ = std::make_unique<JsonPimpl>();
    if (auto* d = dynamic_cast<const DebugError*>(&e))
    {
        output_->obj         = d->details();
        output_->obj["type"] = "DebugError";
    }
    else if (auto* d = dynamic_cast<const RuntimeError*>(&e))
    {
        output_->obj         = d->details();
        output_->obj["type"] = "RuntimeError";
    }
    else
    {
        // Save unknown exception info
        TypeDemangler<std::exception> demangle;
        output_->obj = {{"type", demangle(e)}, {"what", e.what()}};
    }
#else
    (void)sizeof(e);
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
