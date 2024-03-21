//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionParams.cc
//---------------------------------------------------------------------------//
#include "AbsorptionParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<AbsorptionParams>
AbsorptionParams::from_import(ImportData const& data)
{
    CELER_EXPECT(!data.optical.empty());

    if (!std::any_of(
                data.optical.begin(), data.optical.end(), [](auto const& iter) {
                    return static_cast<bool>(iter.second.absorption);
                }))
    {
        // No absorption data present
        return nullptr;
    }

    Input input;
    for (auto const& mat : data.optical)
    {
        input.data.push_back(mat.second.absorption);
    }
    return std::make_shared<AbsorptionParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scintillation input data.
 */
AbsorptionParams::AbsorptionParams(Input const& input)
{
    // Construct data on CPU
    HostVal<AbsorptionData> host_data;

    CollectionBuilder absorption_length{&host_data.absorption_length};
    GenericGridBuilder build_grid(&data.reals);

    for (auto const& mat : input.data)
    {
        // Store absorption length tabulated as a function of photon energy
        auto const& abs_vec = mat.absorption_length;
        if (abs_vec.x.empty())
        {
            // No absorption length for this material
            absorption_length.push_back({});
        }

        CELER_VALIDATE(is_monotonic_increasing(make_span(abs_vec.x)),
                       << "absorption length energy grid values are not "
                          "monotonically increasing");

        absorption_length.push_back(build_grid(abs_vec));
    }
    CELER_ASSERT(absorption_length.size() == input.data.size());

    data_ = CollectionMirror<AbsorptionData>{std::move(data)};
    CELER_ENSURE(data_ || input.data.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
