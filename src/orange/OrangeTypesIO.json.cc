//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypesIO.json.cc
//---------------------------------------------------------------------------//
#include "OrangeTypesIO.json.hh"

#include "corecel/io/StringEnumMapper.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read tolerances.
 */
template<class T>
void from_json(nlohmann::json const& j, Tolerance<T>& value)
{
    j.at("rel").get_to(value.rel);
    CELER_VALIDATE(value.rel > 0 && value.rel < 1,
                   << "tolerance " << value.rel
                   << " is out of range [must be in (0,1)]");

    j.at("abs").get_to(value.abs);
    CELER_VALIDATE(value.abs > 0,
                   << "tolerance " << value.abs
                   << " is out of range [must be greater than zero]");
}

template void from_json(nlohmann::json const&, Tolerance<real_type>&);

//---------------------------------------------------------------------------//
/*!
 * Write tolerances.
 */
template<class T>
void to_json(nlohmann::json& j, Tolerance<T> const& value)
{
    CELER_EXPECT(value);

    j = {
        {"rel", value.rel},
        {"abs", value.abs},
    };
}

template void to_json(nlohmann::json&, Tolerance<real_type> const&);

//---------------------------------------------------------------------------//
/*!
 * Read logic notation.
 */
void from_json(nlohmann::json const& j, LogicNotation& value)
{
    static auto const from_string
        = StringEnumMapper<LogicNotation>::from_cstring_func(to_cstring,
                                                             "logic");
    value = from_string(j.get<std::string>());
}

//---------------------------------------------------------------------------//
/*!
 * Write logic notation.
 */
void to_json(nlohmann::json& j, LogicNotation const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
