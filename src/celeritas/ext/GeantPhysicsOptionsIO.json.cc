//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsOptionsIO.json.hh"

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
const char* to_cstring(BremsModelSelection value)
{
    CELER_EXPECT(value != BremsModelSelection::size_);

    static const char* const strings[] = {
        "seltzer_berger",
        "relativistic",
        "all",
    };
    static_assert(
        static_cast<int>(BremsModelSelection::size_) * sizeof(const char*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list value.
 */
const char* to_cstring(MscModelSelection value)
{
    CELER_EXPECT(value != MscModelSelection::size_);

    static const char* const strings[] = {
        "none",
        "urban",
        "wentzel_vi",
    };
    static_assert(
        static_cast<int>(MscModelSelection::size_) * sizeof(const char*)
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
void from_json(const nlohmann::json& j, GeantPhysicsOptions& opts)
{
    static auto brems_from_string
        = StringEnumMap<BremsModelSelection>::from_cstring_func(to_cstring,
                                                                "brems model");
    static auto msc_from_string
        = StringEnumMap<MscModelSelection>::from_cstring_func(to_cstring,
                                                              "msc model");

    j.at("coulomb_scattering").get_to(opts.coulomb_scattering);
    j.at("rayleigh_scattering").get_to(opts.rayleigh_scattering);
    opts.brems = brems_from_string(j.at("brems").get<std::string>());
    opts.msc   = msc_from_string(j.at("msc").get<std::string>());
    j.at("em_bins_per_decade").get_to(opts.em_bins_per_decade);
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, const GeantPhysicsOptions& opts)
{
    j = nlohmann::json{{"coulomb_scattering", opts.coulomb_scattering},
                       {"rayleigh_scattering", opts.rayleigh_scattering},
                       {"brems", std::string{to_cstring(opts.brems)}},
                       {"msc", std::string{to_cstring(opts.msc)}},
                       {"em_bins_per_decade", opts.em_bins_per_decade}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
