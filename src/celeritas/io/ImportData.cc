//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportData.cc
//---------------------------------------------------------------------------//
#include "ImportData.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/EnumStringMapper.hh"
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
        CELER_LOG(warning)
            << R"(Unit system missing from import data: assuming CGS)";
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
    CELER_LOG(info) << "Converting imported units from '" << usys << "' to '"
                    << UnitSystem::native << "'";

    detail::ImportDataConverter convert{usys};
    convert(data);

    CELER_ENSURE(data->units == units::NativeTraits::label());
}

//---------------------------------------------------------------------------//
/*!
 * Whether an imported model of the given class is present.
 */
bool has_model(ImportData const& data, ImportModelClass model_class)
{
    for (ImportProcess const& process : data.processes)
    {
        for (ImportModel const& model : process.models)
        {
            if (model.model_class == model_class)
            {
                return true;
            }
        }
    }
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Whether an imported MSC model of the given class is present.
 */
bool has_msc_model(ImportData const& data, ImportModelClass model_class)
{
    return std::any_of(
        data.msc_models.begin(),
        data.msc_models.end(),
        [&](ImportMscModel const& m) { return m.model_class == model_class; });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
