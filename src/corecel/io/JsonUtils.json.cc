//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/JsonUtils.json.cc
//---------------------------------------------------------------------------//

#include "JsonUtils.json.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/sys/Version.hh"

#include "Logger.hh"
#include "StreamToString.hh"

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
 *
 * This should be only used for JSON structs intended for \em input in addition
 * to output. Format strings should be all lowercase with hyphens for
 * consistency.
 */
void save_format(nlohmann::json& j, std::string const& format)
{
    CELER_EXPECT(j.is_object());
    j["_format"] = format;
    j["_version"] = stream_to_string(celer_version());
}

//---------------------------------------------------------------------------//
/*!
 * Save units for provenance/reproducibility.
 */
void save_units(nlohmann::json& j)
{
    CELER_EXPECT(j.is_object());
    j["_units"] = to_cstring(UnitSystem::native);
}

//---------------------------------------------------------------------------//
/*!
 * Load and check for a format and compatible version marker.
 */
void check_format(nlohmann::json const& j, std::string_view format)
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
    else
    {
        CELER_LOG(debug) << "Missing validation for JSON '" << format
                         << "' format";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Check units for consistency.
 */
void check_units(nlohmann::json const& j, std::string_view format)
{
    if (auto iter = j.find("_units"); iter != j.end())
    {
        CELER_VALIDATE(
            to_unit_system(iter->get<std::string>()) == UnitSystem::native,
            << "incompatible unit system in " << format
            << " JSON file: constructed with " << iter->get<std::string>()
            << " units, but current executable requires "
            << UnitSystem::native);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
