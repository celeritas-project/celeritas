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
#include "corecel/math/QuantityIO.json.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the Bremsstrahlung model selection.
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
 * Get a string corresponding to the multiple scattering model selection.
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
/*!
 * Get a string corresponding to the atomic relaxation option.
 */
const char* to_cstring(RelaxationSelection value)
{
    CELER_EXPECT(value != RelaxationSelection::size_);

    static const char* const strings[] = {
        "none",
        "radiative",
        "all",
    };
    static_assert(
        static_cast<int>(RelaxationSelection::size_) * sizeof(const char*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// JSON serializers
//---------------------------------------------------------------------------//
void from_json(const nlohmann::json& j, MscModelSelection& value)
{
    static const auto from_string
        = StringEnumMap<MscModelSelection>::from_cstring_func(to_cstring,
                                                              "msc model");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, const MscModelSelection& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(const nlohmann::json& j, BremsModelSelection& value)
{
    static const auto from_string
        = StringEnumMap<BremsModelSelection>::from_cstring_func(to_cstring,
                                                                "brems model");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, const BremsModelSelection& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(const nlohmann::json& j, RelaxationSelection& value)
{
    static const auto from_string
        = StringEnumMap<RelaxationSelection>::from_cstring_func(
            to_cstring, "atomic relaxation");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, const RelaxationSelection& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(const nlohmann::json& j, GeantPhysicsOptions& options)
{
    options = {};
#define GPO_LOAD_OPTION(NAME)                 \
    do                                        \
    {                                         \
        if (j.count(#NAME))                   \
            j.at(#NAME).get_to(options.NAME); \
    } while (0)
    GPO_LOAD_OPTION(coulomb_scattering);
    GPO_LOAD_OPTION(rayleigh_scattering);
    GPO_LOAD_OPTION(eloss_fluctuation);
    GPO_LOAD_OPTION(lpm);
    GPO_LOAD_OPTION(integral_approach);
    GPO_LOAD_OPTION(brems);
    GPO_LOAD_OPTION(msc);
    GPO_LOAD_OPTION(relaxation);
    GPO_LOAD_OPTION(em_bins_per_decade);
    GPO_LOAD_OPTION(min_energy);
    GPO_LOAD_OPTION(max_energy);
    GPO_LOAD_OPTION(linear_loss_limit);
#undef GPO_LOAD_OPTION
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, const GeantPhysicsOptions& options)
{
    j = nlohmann::json::object();
#define GPO_SAVE_OPTION(NAME) j[#NAME] = options.NAME
    GPO_SAVE_OPTION(coulomb_scattering);
    GPO_SAVE_OPTION(rayleigh_scattering);
    GPO_SAVE_OPTION(eloss_fluctuation);
    GPO_SAVE_OPTION(lpm);
    GPO_SAVE_OPTION(integral_approach);
    GPO_SAVE_OPTION(brems);
    GPO_SAVE_OPTION(msc);
    GPO_SAVE_OPTION(relaxation);
    GPO_SAVE_OPTION(em_bins_per_decade);
    GPO_SAVE_OPTION(min_energy);
    GPO_SAVE_OPTION(max_energy);
    GPO_SAVE_OPTION(linear_loss_limit);
#undef GPO_SAVE_OPTION
}

//---------------------------------------------------------------------------//
} // namespace celeritas
