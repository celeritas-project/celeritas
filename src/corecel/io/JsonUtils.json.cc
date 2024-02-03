//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/JsonUtils.json.cc
//---------------------------------------------------------------------------//

#include "JsonUtils.json.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/Version.hh"

#include "Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Print a warning about a deprecated input option.
 */
void warn_deprecated_json_option(char const* old_name, char const* new_name)
{
    CELER_LOG(warning) << "Deprecated option '" << old_name << "': use '"
                       << new_name << "' instead";
}

//---------------------------------------------------------------------------//
/*!
 * Save a format and version marker.
 */
void save_format(nlohmann::json& j, std::string const& format)
{
    j["_format"] = format;
    j["_version"] = to_string(celer_version());
}

//---------------------------------------------------------------------------//
/*!
 * Load and check for a format and compatible version marker.
 */
void check_format(nlohmann::json const& j, std::string const& format)
{
    if (auto iter = j.find("_version"); iter != j.end())
    {
        auto version = Version::from_string(iter->get<std::string>());
        if (version > celer_version())
        {
            CELER_LOG(warning)
                << "Input version " << version
                << " is newer than the current Celeritas version "
                << celer_version()
                << ": options may be missing or incompatible";
        }
    }
    if (auto iter = j.find("_format"); iter != j.end())
    {
        auto format_str = iter->get<std::string>();
        CELER_VALIDATE(format_str == format,
                       << "invalid format for \"" << format << "\" input: \""
                       << format_str << "\"");
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
