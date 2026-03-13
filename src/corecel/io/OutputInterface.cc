//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputInterface.cc
//---------------------------------------------------------------------------//
#include "OutputInterface.hh"

#include <iostream>
#include <nlohmann/json.hpp>

#include "EnumStringMapper.hh"
#include "JsonPimpl.hh"
#include "StreamToString.hh"

using Category = celeritas::OutputInterface::Category;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a category.
 */
char const* to_cstring(Category value)
{
    static EnumStringMapper<Category> const to_cstring_impl{
        "input",
        "result",
        "system",
        "internal",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the JSON representation of a single output.
 *
 * This is used mostly for testing, but it can also be useful for quickly
 * generating useful JSON output from applications, e.g. with exception output.
 */
std::string to_string(OutputInterface const& output)
{
    return stream_to_string(output);
}

//---------------------------------------------------------------------------//
/*!
 * Stream the JSON representation of a single output.
 */
std::ostream& operator<<(std::ostream& os, OutputInterface const& output)
{
    JsonPimpl json_wrap;
    output.output(&json_wrap);
    json_wrap.obj["_category"] = to_cstring(output.category());
    json_wrap.obj["_label"] = output.label();
    os << json_wrap.obj;
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
