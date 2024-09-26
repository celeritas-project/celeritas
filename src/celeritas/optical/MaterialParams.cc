//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialParams.cc
//---------------------------------------------------------------------------//
#include "MaterialParams.hh"

#include <algorithm>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/grid/GenericGridBuilder.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data and material/volume data.
 */
std::shared_ptr<MaterialParams>
MaterialParams::from_import(ImportData const& data,
                            ::celeritas::GeoMaterialParams const& geo_mat,
                            ::celeritas::MaterialParams const& mat)
{
    CELER_EXPECT(!data.optical_materials.empty());
    CELER_EXPECT(geo_mat.num_volumes() > 0);

    CELER_VALIDATE(std::all_of(data.optical_materials.begin(),
                               data.optical_materials.end(),
                               [](ImportOpticalMaterial const& m) {
                                   return static_cast<bool>(m);
                               }),
                   << "one or more optical materials lack required data");

    Input inp;

    // Extract optical material properties
    inp.properties.reserve(data.optical_materials.size());
    for (ImportOpticalMaterial const& opt_mat : data.optical_materials)
    {
        inp.properties.push_back(opt_mat.properties);
    }

    // Construct volume-to-optical mapping
    inp.volume_to_mat.reserve(geo_mat.num_volumes());
    bool has_opt_mat{false};
    for (auto vid : range(VolumeId{geo_mat.num_volumes()}))
    {
        OpticalMaterialId optmat;
        if (auto matid = geo_mat.material_id(vid))
        {
            auto mat_view = mat.get(matid);
            optmat = mat_view.optical_material_id();
            if (optmat)
            {
                has_opt_mat = true;
            }
        }
        inp.volume_to_mat.push_back(optmat);
    }
    CELER_VALIDATE(has_opt_mat,
                   << "no volumes have associated optical materials");

    CELER_ENSURE(inp.volume_to_mat.size() == geo_mat.num_volumes());
    return std::make_shared<MaterialParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
MaterialParams::MaterialParams(Input const& inp)
    : num_materials_(inp.properties.size())
{
    CELER_EXPECT(!inp.properties.empty());
    CELER_EXPECT(!inp.volume_to_mat.empty());

    HostVal<MaterialParamsData> data;
    CollectionBuilder refractive_index{&data.refractive_index};
    GenericGridBuilder build_grid(&data.reals);
    for (auto opt_mat_idx : range(inp.properties.size()))
    {
        auto const& mat = inp.properties[opt_mat_idx];

        // Store refractive index tabulated as a function of photon energy.
        // In a dispersive medium, the index of refraction is an increasing
        // function of photon energy
        auto const& ri_vec = mat.refractive_index;
        CELER_VALIDATE(ri_vec,
                       << "no refractive index data is defined for optical "
                          "material "
                       << opt_mat_idx);
        CELER_VALIDATE(is_monotonic_increasing(make_span(ri_vec.x)),
                       << "refractive index energy grid values are not "
                          "monotonically increasing");
        CELER_VALIDATE(is_monotonic_increasing(make_span(ri_vec.y)),
                       << "refractive index values are not monotonically "
                          "increasing");
        if (ri_vec.y.front() < 1)
        {
            CELER_LOG(warning) << "Encountered refractive index below unity "
                                  "for optical material "
                               << opt_mat_idx;
        }

        refractive_index.push_back(build_grid(ri_vec));
    }
    CELER_ASSERT(refractive_index.size() == inp.properties.size());

    for (auto optmat : inp.volume_to_mat)
    {
        CELER_VALIDATE(!optmat || optmat < inp.properties.size(),
                       << "optical material ID " << optmat.unchecked_get()
                       << " provided to material params is out of range");
    }
    CollectionBuilder{&data.optical_id}.insert_back(inp.volume_to_mat.begin(),
                                                    inp.volume_to_mat.end());

    data_ = CollectionMirror<MaterialParamsData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
