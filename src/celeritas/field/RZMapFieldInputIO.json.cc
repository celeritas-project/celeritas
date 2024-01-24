//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RZMapFieldInputIO.json.hh"

#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"

#include "FieldDriverOptionsIO.json.hh"
#include "RZMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read field from JSON.
 */
void from_json(nlohmann::json const& j, RZMapFieldInput& inp)
{
#define RZFI_LOAD(NAME) j.at(#NAME).get_to(inp.NAME)
    using namespace celeritas::units;

    RZFI_LOAD(num_grid_z);
    RZFI_LOAD(num_grid_r);
    RZFI_LOAD(min_z);
    RZFI_LOAD(min_r);
    RZFI_LOAD(max_z);
    RZFI_LOAD(max_r);
    RZFI_LOAD(field_z);
    RZFI_LOAD(field_r);
    if (j.contains("driver_options"))
    {
        RZFI_LOAD(driver_options);
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
                << "Deprecated RZ field input units '" << ustr
                << "': use SI units for length (m) and field (T) "
                   "and set units to 'si'";
            field_units = UnitSystem::si;
        }
        else if (ustr == "gauss" || ustr == Gauss::label() || ustr == "native")
        {
            // TODO: Remove in 1.0
            CELER_LOG(warning) << "Deprecated RZ field input units '" << ustr
                               << "': replace with 'cgs' (Gauss + cm)";
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
        msg << "No units given in RZ field input: assuming CGS for length "
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
        for (auto* f : {&inp.field_z, &inp.field_r})
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
    }
#undef RZFI_LOAD
}

//---------------------------------------------------------------------------//
/*!
 * Write field to JSON.
 */
void to_json(nlohmann::json& j, RZMapFieldInput const& inp)
{
#define RZFI_KEY_VALUE(NAME) {#NAME, inp.NAME}
    j = {
        {"_units", units::NativeTraits::label()},
        RZFI_KEY_VALUE(num_grid_z),
        RZFI_KEY_VALUE(num_grid_r),
        RZFI_KEY_VALUE(min_z),
        RZFI_KEY_VALUE(min_r),
        RZFI_KEY_VALUE(max_z),
        RZFI_KEY_VALUE(max_r),
        RZFI_KEY_VALUE(field_z),
        RZFI_KEY_VALUE(field_r),
        RZFI_KEY_VALUE(driver_options),
    };
#undef RZFI_KEY_VALUE
}

//---------------------------------------------------------------------------//
// Helper to read the field from a file or stream.
std::istream& operator>>(std::istream& is, RZMapFieldInput& inp)
{
    auto j = nlohmann::json::parse(is);
    j.get_to(inp);
    return is;
}

//---------------------------------------------------------------------------//
// Helper to write the field to a file or stream.
std::ostream& operator<<(std::ostream& os, RZMapFieldInput const& inp)
{
    nlohmann::json j = inp;
    os << j.dump(0);
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
