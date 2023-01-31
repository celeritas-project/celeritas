//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/AssertIO.json.cc
//---------------------------------------------------------------------------//
#include "AssertIO.json.hh"

#include "io/JsonPimpl.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
void details_to_json(nlohmann::json& j, T const& d)
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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Write details of a debug error to JSON output.
 */
void to_json(nlohmann::json& j, DebugErrorDetails const& d)
{
    details_to_json(j, d);
}

//---------------------------------------------------------------------------//
/*!
 * Write details of a runtime error to JSON output.
 */
void to_json(nlohmann::json& j, RuntimeErrorDetails const& d)
{
    j["what"] = d.what;
    details_to_json(j, d);
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
} // namespace celeritas
