//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/WavelengthShiftParams.cc
//---------------------------------------------------------------------------//
#include "WavelengthShiftParams.hh"

#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridBuilder.hh"
#include "celeritas/grid/GenericGridInserter.hh"
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
    HostVal<WavelengthShiftData> data;

    CollectionBuilder wls_record{&data.wls_record};
    GenericGridInserter insert_energy_cdf(&data.reals, &data.energy_cdf);
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
        // Store WLS component tabulated as a function of photon energy
        auto const& comp_vec = wls.component;
        std::vector<double> cdf(comp_vec.x.size());

        CELER_ASSERT(comp_vec.y[0] > 0);
        // The value of cdf at the low edge is zero by default
        cdf[0] = 0;
        for (size_type i = 1; i < comp_vec.x.size(); ++i)
        {
            // TODO: use trapezoidal integrator helper class
            cdf[i] = cdf[i - 1]
                     + 0.5 * (comp_vec.x[i] - comp_vec.x[i - 1])
                           * (comp_vec.y[i] + comp_vec.y[i - 1]);
        }

        // Normalize for the cdf probability
        for (size_type i = 1; i < comp_vec.x.size(); ++i)
        {
            cdf[i] = cdf[i] / cdf.back();
        }
        // Note that energy and cdf are swapped for the inverse sampling
        insert_energy_cdf(make_span(cdf), make_span(comp_vec.x));
    }
    CELER_ASSERT(data.energy_cdf.size() == input.data.size());
    CELER_ASSERT(data.wls_record.size() == data.energy_cdf.size());

    data_ = CollectionMirror<WavelengthShiftData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
