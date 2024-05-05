//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportData.cc
//---------------------------------------------------------------------------//
#include "ImportData.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/UnitTypes.hh"

#include "detail/ImportDataConverter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Recursively convert imported data to the native unit type.
 */
void convert_to_native(ImportData* data)
{
    CELER_EXPECT(data);

    // Convert data
    if (data->units.empty())
    {
        CELER_LOG(debug) << "Unit system missing from import data: assuming "
                            "CGS";
        data->units = to_cstring(UnitSystem::cgs);
    }

    // Convert string to unit system enum
    UnitSystem const usys = to_unit_system(data->units);
    CELER_VALIDATE(usys != UnitSystem::none, << "invalid unit system");

    if (usys == UnitSystem::native)
    {
        // No unit conversion needed
        return;
    }
    CELER_LOG(info) << "Converting imported units from '" << to_cstring(usys)
                    << "' to '" << to_cstring(UnitSystem::native) << "'";

    detail::ImportDataConverter convert{usys};
    convert(data);

    CELER_ENSURE(data->units == units::NativeTraits::label());
}

//---------------------------------------------------------------------------//
/*!
 * Return a vector containing all imported models of the given class.
 */
std::vector<ImportModel const*>
find_models(ImportData const& data, ImportModelClass model_class)
{
    std::vector<ImportModel const*> result;
    for (ImportProcess const& process : data.processes)
    {
        for (ImportModel const& model : process.models)
        {
            if (model.model_class == model_class)
            {
                result.push_back(&model);
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return a vector containing all imported MSC models of the given class.
 */
std::vector<ImportMscModel const*>
find_msc_models(ImportData const& data, ImportModelClass model_class)
{
    std::vector<ImportMscModel const*> result;
    for (ImportMscModel const& model : data.msc_models)
    {
        if (model.model_class == model_class)
        {
            result.push_back(&model);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
