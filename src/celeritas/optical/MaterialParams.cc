//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialParams.cc
//---------------------------------------------------------------------------//
#include "MaterialParams.hh"

#include <algorithm>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
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
                               Identity{}),
                   << "one or more optical materials lack required data");

    Input inp;

    // Extract optical material properties
    inp.properties.reserve(data.optical_materials.size());
    for (ImportOpticalMaterial const& opt_mat : data.optical_materials)
    {
        inp.properties.push_back(opt_mat.properties);
    }

    // Construct impl-volume-to-optical mapping
    inp.volume_to_mat.assign(geo_mat.num_volumes(), OptMatId{});
    inp.optical_to_core.assign(inp.properties.size(), PhysMatId{});

    std::unordered_set<OptMatId> all_optmat;
    for (auto iv_id : range(ImplVolumeId{geo_mat.num_volumes()}))
    {
        if (PhysMatId matid = geo_mat.material_id(iv_id))
        {
            auto mat_view = mat.get(matid);
            OptMatId optmat = mat_view.optical_material_id();
            if (optmat)
            {
                CELER_ASSERT(optmat < inp.optical_to_core.size());
                all_optmat.insert(optmat);
                inp.optical_to_core[optmat.get()] = matid;
                inp.volume_to_mat[iv_id.get()] = optmat;
            }
        }
    }
    CELER_VALIDATE(!all_optmat.empty(),
                   << "no volumes have associated optical materials");

    CELER_LOG(info) << "Constructed " << inp.properties.size()
                    << " optical materials with " << all_optmat.size()
                    << " present in the geometry";

    return std::make_shared<MaterialParams>(std::move(inp));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
MaterialParams::MaterialParams(Input const& inp)
{
    CELER_EXPECT(!inp.properties.empty());
    CELER_EXPECT(!inp.volume_to_mat.empty());
    CELER_EXPECT(inp.optical_to_core.size() == inp.properties.size());

    HostVal<MaterialParamsData> data;
    NonuniformGridInserter insert_grid(&data.reals, &data.refractive_index);
    for (auto opt_mat_idx : range(inp.properties.size()))
    {
        auto const& mat = inp.properties[opt_mat_idx];

        // Store refractive index tabulated as a function of photon energy.
        // In a dispersive medium, the index of refraction is an increasing
        // function of photon energy
        auto const& ri = mat.refractive_index;
        CELER_VALIDATE(ri,
                       << "no refractive index data is defined for optical "
                          "material "
                       << opt_mat_idx);
        CELER_VALIDATE(is_monotonic_increasing(make_span(ri.x)),
                       << "refractive index energy grid values are not "
                          "monotonically increasing");
        CELER_VALIDATE(is_monotonic_nondecreasing(make_span(ri.y)),
                       << "refractive index values are not constant or "
                          "increasing");
        if (ri.y.front() < 1)
        {
            CELER_LOG(warning) << "Encountered refractive index below unity "
                                  "for optical material "
                               << opt_mat_idx;
        }
        insert_grid(ri);
    }
    CELER_ASSERT(data.refractive_index.size() == inp.properties.size());

    for (auto optmat : inp.volume_to_mat)
    {
        CELER_VALIDATE(!optmat || optmat < inp.properties.size(),
                       << "optical material ID " << optmat.unchecked_get()
                       << " provided to material params is out of range");
    }
    CollectionBuilder{&data.optical_id}.insert_back(inp.volume_to_mat.begin(),
                                                    inp.volume_to_mat.end());

    CollectionBuilder{&data.core_material_id}.insert_back(
        inp.optical_to_core.begin(), inp.optical_to_core.end());

    data_ = ParamsDataStore<MaterialParamsData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a material view for the given identifier.
 */
MaterialView MaterialParams::get(OptMatId mat) const
{
    return MaterialView(this->host_ref(), mat);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
