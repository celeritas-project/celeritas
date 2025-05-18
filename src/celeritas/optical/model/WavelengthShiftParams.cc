//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftParams.cc
//---------------------------------------------------------------------------//
#include "WavelengthShiftParams.hh"

#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/PdfUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct wavelength shift (WLS) data with imported data.
 */
std::shared_ptr<WavelengthShiftParams>
WavelengthShiftParams::from_import(ImportData const& data)
{
    CELER_EXPECT(!data.optical_materials.empty());

    if (!std::any_of(
            data.optical_materials.begin(),
            data.optical_materials.end(),
            [](auto const& iter) { return static_cast<bool>(iter.wls); }))
    {
        // No wavelength shift data present
        return nullptr;
    }

    Input input;
    for (auto const& mat : data.optical_materials)
    {
        input.data.push_back(mat.wls);
    }
    return std::make_shared<WavelengthShiftParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with wavelength shift (WLS) input data.
 */
WavelengthShiftParams::WavelengthShiftParams(Input const& input)
{
    CELER_EXPECT(input.data.size() > 0);

    SegmentIntegrator integrate_emission{TrapezoidSegmentIntegrator{}};

    HostVal<WavelengthShiftData> data;
    CollectionBuilder wls_record{&data.wls_record};
    NonuniformGridInserter insert_energy_cdf(&data.reals, &data.energy_cdf);
    for (auto const& wls : input.data)
    {
        if (!wls)
        {
            // No WLS data for this material
            wls_record.push_back({});
            insert_energy_cdf();
            continue;
        }

        // WLS material properties
        WlsMaterialRecord record;
        record.mean_num_photons = wls.mean_num_photons;
        record.time_constant = wls.time_constant;
        wls_record.push_back(record);

        // Calculate the WLS cumulative probability of the emission spectrum
        inp::Grid grid;
        grid.x = wls.component.x;
        grid.y.resize(grid.x.size());
        integrate_emission(make_span(wls.component.x),
                           make_span(wls.component.y),
                           make_span(grid.y));
        normalize_cdf(make_span(grid.y));

        // Insert energy -> CDF grid
        insert_energy_cdf(grid);
    }
    CELER_ASSERT(data.energy_cdf.size() == input.data.size());
    CELER_ASSERT(data.wls_record.size() == data.energy_cdf.size());

    data_ = CollectionMirror<WavelengthShiftData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
