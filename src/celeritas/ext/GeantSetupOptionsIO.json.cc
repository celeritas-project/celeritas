//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetupOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "GeantSetupOptionsIO.json.hh"

#include "corecel/Assert.hh"
#include "corecel/io/StringEnumMap.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list value.
 */
const char* to_cstring(GeantSetupPhysicsList value)
{
    CELER_EXPECT(value != GeantSetupPhysicsList::size_);

    static const char* const strings[] = {
        "none",
        "em_basic",
        "em_standard",
        "ftfp_bert",
    };
    static_assert(
        static_cast<int>(GeantSetupPhysicsList::size_) * sizeof(const char*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<int>(value)];
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(const nlohmann::json& j, GeantSetupOptions& opts)
{
    static auto gspl_from_string
        = StringEnumMap<GeantSetupPhysicsList>::from_cstring_func(
            to_cstring, "physics list");
    opts.physics = gspl_from_string(j.at("physics").get<std::string>());
    j.at("em_bins_per_decade").get_to(opts.em_bins_per_decade);
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, const GeantSetupOptions& opts)
{
    j = nlohmann::json{{"physics", std::string{to_cstring(opts.physics)}},
                       {"em_bins_per_decade", opts.em_bins_per_decade}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
