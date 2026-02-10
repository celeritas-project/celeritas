//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/inp/IO.json.cc
//---------------------------------------------------------------------------//
#include "IO.json.hh"

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "orange/OrangeTypesIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
static char const format_str[] = "g4org-options";

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to an inline singletons option.
 */
char const* to_cstring(InlineSingletons value)
{
    static EnumStringMapper<InlineSingletons> const to_cstring_impl{
        "none",
        "untransformed",
        "unrotated",
        "all",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, InlineSingletons const& v)
{
    j = std::string{to_cstring(v)};
}

void from_json(nlohmann::json const& j, InlineSingletons& v)
{
    static auto const from_string
        = StringEnumMapper<InlineSingletons>::from_cstring_func(
            to_cstring, "inline singletons");
    v = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, OrangeGeoFromGeant const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, unit_length),
        CELER_JSON_PAIR(v, explicit_interior_threshold),
        CELER_JSON_PAIR(v, inline_childless),
        CELER_JSON_PAIR(v, inline_singletons),
        CELER_JSON_PAIR(v, inline_unions),
        CELER_JSON_PAIR(v, implicit_parent_boundary),
        CELER_JSON_PAIR(v, logic),
        CELER_JSON_PAIR(v, verbose_volumes),
        CELER_JSON_PAIR(v, verbose_structure),
        CELER_JSON_PAIR(v, tol),
        CELER_JSON_PAIR(v, objects_output_file),
        CELER_JSON_PAIR(v, csg_output_file),
        CELER_JSON_PAIR(v, org_output_file),
    };

#undef OPT_JSON_STRING

    save_format(j, format_str);
}

void from_json(nlohmann::json const& j, OrangeGeoFromGeant& v)
{
    check_format(j, format_str);

    for (auto s : {"remove_interior", "remove_negated_join"})
    {
        CELER_VALIDATE(!j.contains(s),
                       << "deleted geo conversion option '" << s << '\'');
    }

#define OPT_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, v, NAME)

    OPT_LOAD_OPTION(unit_length);
    OPT_LOAD_OPTION(tol);
    OPT_LOAD_OPTION(explicit_interior_threshold);
    OPT_LOAD_OPTION(inline_childless);
    OPT_LOAD_OPTION(inline_singletons);
    OPT_LOAD_OPTION(inline_unions);
    OPT_LOAD_OPTION(implicit_parent_boundary);
    OPT_LOAD_OPTION(logic);
    OPT_LOAD_OPTION(verbose_volumes);
    OPT_LOAD_OPTION(verbose_structure);
    OPT_LOAD_OPTION(objects_output_file);
    OPT_LOAD_OPTION(csg_output_file);
    OPT_LOAD_OPTION(org_output_file);

#undef OPT_LOAD_OPTION
}

//!@}

//---------------------------------------------------------------------------//
/*!
 * Helper to read the import options from a file or stream.
 *
 * Example to read from a file:
 * \code
   Options inp;
   std::ifstream("foo.json") >> inp;
 * \endcode
 */
std::istream& operator>>(std::istream& is, OrangeGeoFromGeant& inp)
{
    auto j = nlohmann::json::parse(is);
    j.get_to(inp);
    return is;
}

//---------------------------------------------------------------------------//
/*!
 * Helper to write the options to a file or stream.
 */
std::ostream& operator<<(std::ostream& os, OrangeGeoFromGeant const& inp)
{
    nlohmann::json j = inp;
    os << j.dump();
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
