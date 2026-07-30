//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Version.cc
//---------------------------------------------------------------------------//
#include "Version.hh"

#include <cstdlib>
#include <iostream>
#include <regex>

#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Join.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a string "1.2.3".
 */
Version Version::from_string(std::string_view sv)
{
    static std::regex const version_regex{
        R"re(^(\d+)(?:\.(\d+)(?:\.(\d+)(?:\.\d+)*)?)?(?:-.*)?)re"};
    std::match_results<std::string_view::iterator> version_match;
    bool matched
        = std::regex_match(sv.begin(), sv.end(), version_match, version_regex);
    CELER_VALIDATE(matched, << "failed to parse version '" << sv << "'");

    auto match_to_int = [](auto const& submatch) {
        if (submatch.length() == 0)
        {
            // No version component given
            return size_type{0};
        }
        int result = std::atoi(&(*submatch.first));
        return static_cast<size_type>(result);
    };

    return Version{match_to_int(version_match[1]),
                   match_to_int(version_match[2]),
                   match_to_int(version_match[3])};
}

//---------------------------------------------------------------------------//
/*!
 * Write to a stream.
 */
std::ostream& operator<<(std::ostream& os, Version const& v)
{
    os << join(v.value().begin(), v.value().end(), '.');
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Celeritas version.
 */
Version celer_version()
{
    return {celeritas::version_major,
            celeritas::version_minor,
            celeritas::version_patch};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
