//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/AssertIO.json.cc
//---------------------------------------------------------------------------//
#include "AssertIO.json.hh"

#include "io/JsonPimpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write details of a debug error to JSON output.
 */
void to_json(nlohmann::json& j, DebugErrorDetails const& d)
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
/*!
 * Write details of a runtime error to JSON output.
 */
void to_json(nlohmann::json& j, RuntimeErrorDetails const& d)
{
    j["what"] = d.what;
    j["which"] = to_cstring(d.which);
    if (!d.condition.empty())
    {
        j["condition"] = d.condition;
    }
    if (!d.file.empty())
    {
        j["file"] = d.file;
    }
    if (d.line != 0)
    {
        j["line"] = d.line;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write properties of a rich context exception.
 */
void to_json(nlohmann::json& j, RichContextException const& e)
{
    JsonPimpl temp{std::move(j)};
    e.output(&temp);
    j = std::move(temp.obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
