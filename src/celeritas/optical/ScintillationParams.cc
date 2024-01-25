//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationParams.cc
//---------------------------------------------------------------------------//
#include "ScintillationParams.hh"

#include <numeric>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with scintillation input data.
 */
ScintillationParams::ScintillationParams(ScintillationInput const& input)
{
    CELER_EXPECT(input.data.size() > 0);
    HostVal<ScintillationData> host_data;

    CollectionBuilder spectra(&host_data.spectra);
    CollectionBuilder components(&host_data.components);
    for (auto const& spec : input.data)
    {
        CELER_ASSERT(spec.size() > 0);

        // Validity of scintillation component data
        real_type total_yield{0};
        for (auto const& comp : spec)
        {
            CELER_ASSERT(comp);
            total_yield += comp.yield_prob;
        }
        CELER_ASSERT(soft_equal(real_type(1), total_yield));

        ScintillationSpectrum spectrum;
        spectrum.components = components.insert_back(spec.begin(), spec.end());
        spectra.push_back(spectrum);
    }

    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
