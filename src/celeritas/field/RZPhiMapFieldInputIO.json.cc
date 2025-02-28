//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RZPhiMapFieldInputIO.json.hh"

#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"

#include "FieldDriverOptionsIO.json.hh"
#include "RZPhiMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
static char const format_str[] = "rzphi-map-field";

//---------------------------------------------------------------------------//
/*!
 * Read field from JSON.
 */
void from_json(nlohmann::json const& j, RZPhiMapFieldInput& inp)
{
#define RZPFI_LOAD(NAME) j.at(#NAME).get_to(inp.NAME)
    using namespace celeritas::units;

    check_format(j, format_str);
    check_units(j, format_str);

    RZPFI_LOAD(num_grid_z);
    RZPFI_LOAD(num_grid_r);
    RZPFI_LOAD(num_grid_phi);
    RZPFI_LOAD(min_z);
    RZPFI_LOAD(min_r);
    RZPFI_LOAD(min_phi);
    RZPFI_LOAD(max_z);
    RZPFI_LOAD(max_r);
    RZPFI_LOAD(max_phi);
    RZPFI_LOAD(field_z);
    RZPFI_LOAD(field_r);
    RZPFI_LOAD(field_phi);
    if (j.contains("driver_options"))
    {
        RZPFI_LOAD(driver_options);
    }

    // Convert unit systems based on input
    UnitSystem length_units{UnitSystem::cgs};  // cm
    UnitSystem field_units{UnitSystem::si};  // tesla
    if (auto iter = j.find("_units"); iter != j.end())
    {
        auto const& ustr = iter->get<std::string>();
        if (ustr == "tesla" || ustr == "T")
        {
            CELER_LOG(warning)
                << "Deprecated RZPhi field input units '" << ustr
                << "': use SI units for length (m) and field (T) "
                   "and set units to 'si'";
            field_units = UnitSystem::si;
        }
        else if (ustr == "gauss" || ustr == Gauss::label() || ustr == "native")
        {
            //! \todo Remove in 1.0
            CELER_LOG(warning) << "Deprecated RZPhi field input units '"
                               << ustr << "': replace with 'cgs' (Gauss + cm)";
            field_units = UnitSystem::cgs;
        }
        else
        {
            try
            {
                // Input should be si/cgs/clhep
                length_units = to_unit_system(ustr);
                field_units = length_units;
            }
            catch (RuntimeError const& e)
            {
                CELER_VALIDATE(false,
                               << "unrecognized value '" << ustr
                               << "' for \"_units\" field: " << e.what());
            }
        }
    }
    else
    {
        auto msg = CELER_LOG(warning);
        msg << "No units given in RZPhi field input: assuming CGS for length "
               "(cm) and SI for strength (T)";
    }

    if (field_units != UnitSystem::native)
    {
        CELER_LOG(info) << "Converting magnetic field input strength from "
                        << to_cstring(field_units) << " to ["
                        << NativeTraits::BField::label() << "]";

        double field_scale = visit_unit_system(
            [](auto traits) {
                using Unit = typename decltype(traits)::BField;
                return native_value_from(Quantity<Unit, double>{1});
            },
            field_units);

        CELER_LOG(debug) << "Scaling input magnetic field by " << field_scale;

        // Convert units from JSON tesla to input native
        for (auto* f : {&inp.field_z, &inp.field_r, &inp.field_phi})
        {
            for (double& v : *f)
            {
                v *= field_scale;
            }
        }
    }

    if (length_units != UnitSystem::native)
    {
        CELER_LOG(info) << "Converting magnetic field input positions from "
                        << to_cstring(length_units) << " to ["
                        << NativeTraits::Length::label() << "]";

        double length_scale = visit_unit_system(
            [](auto traits) {
                using Unit = typename decltype(traits)::Length;
                return native_value_from(Quantity<Unit, double>{1});
            },
            length_units);

        CELER_LOG(debug) << "Scaling input lengths by " << length_scale;

        // Convert units from JSON tesla to input native
        for (auto* v : {&inp.min_z, &inp.max_z, &inp.min_r, &inp.max_r})
        {
            *v *= length_scale;
        }
        // Note: min_phi and max_phi are in radians, which don't need scaling
    }
#undef RZPFI_LOAD
}

//---------------------------------------------------------------------------//
/*!
 * Write field to JSON.
 */
void to_json(nlohmann::json& j, RZPhiMapFieldInput const& inp)
{
    j = {
        {"_format", "RZPhiMapField"},
        {"_version", 0},
        CELER_JSON_PAIR(inp, num_grid_z),
        CELER_JSON_PAIR(inp, num_grid_r),
        CELER_JSON_PAIR(inp, num_grid_phi),
        CELER_JSON_PAIR(inp, min_z),
        CELER_JSON_PAIR(inp, min_r),
        CELER_JSON_PAIR(inp, min_phi),
        CELER_JSON_PAIR(inp, max_z),
        CELER_JSON_PAIR(inp, max_r),
        CELER_JSON_PAIR(inp, max_phi),
        CELER_JSON_PAIR(inp, field_z),
        CELER_JSON_PAIR(inp, field_r),
        CELER_JSON_PAIR(inp, field_phi),
        CELER_JSON_PAIR(inp, driver_options),
    };
    save_format(j, format_str);
    save_units(j);
}

//---------------------------------------------------------------------------//
// Helper to read the field from a file or stream.
std::istream& operator>>(std::istream& is, RZPhiMapFieldInput& inp)
{
    auto j = nlohmann::json::parse(is);
    j.get_to(inp);
    return is;
}

//---------------------------------------------------------------------------//
// Helper to write the field to a file or stream.
std::ostream& operator<<(std::ostream& os, RZPhiMapFieldInput const& inp)
{
    nlohmann::json j = inp;
    os << j.dump(0);
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
